#!/usr/bin/env python3
# sag_d4rl_antmaze.py
# Inference-only AntMaze pipeline with SAGE factorized selection:
#   - sample C candidate state-only plans from planner
#   - infer prefix actions via diffusion inverse dynamics (policy)
#   - compute SAGE prefix energy E
#   - keep lowest-energy subset (keep_p), then select by (J - lambda * E)
#
# Assumptions (enforced):
#   - guidance_type == "MCSS"   (we use the horizon critic as utility J)
#   - pipeline_type == "separate" and use_diffusion_invdyn == True (diffusion inverse dynamics)

from __future__ import annotations

import os
import sys
import uuid

import d4rl  # noqa: F401
import gym
import hydra
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf

from cleandiffuser.dataset.d4rl_antmaze_dataset import DV_D4RLAntmazeSeqDataset
from cleandiffuser.diffusion import ContinuousDiffusionSDE, DiscreteDiffusionSDE
from cleandiffuser.nn_condition import IdentityCondition
from cleandiffuser.nn_diffusion import DiT1d, DVInvMlp, JannerUNet1d
from cleandiffuser.utils import DVHorizonCritic, report_parameters
from utils import set_seed

# ---- allow "import energy", "import jepa" from repo root ----
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from energy import (  # noqa: E402
    SAGEEnergyScorer,
    _infer_prefix_actions_for_state_only_plans,
    _select_with_sage,
)


@hydra.main(config_path="../configs/veteran/antmaze", config_name="antmaze", version_base=None)
def pipeline(args):
    # ------------------------ setup ------------------------
    args.device = args.device if torch.cuda.is_available() else "cpu"
    assert str(args.mode) == "inference", "This script is inference-only. Set mode=inference."
    assert str(args.guidance_type) == "MCSS", "This script supports MCSS only (critic utility J)."
    assert str(args.pipeline_type) == "separate", "This script is for separate (state-only planner) pipeline."
    assert bool(args.use_diffusion_invdyn), "This script assumes diffusion inverse dynamics (use_diffusion_invdyn=True)."

    if bool(getattr(args, "enable_wandb", False)):
        wandb.require("core")
        wandb.init(
            reinit=True,
            id=str(uuid.uuid4()),
            project=str(args.project),
            group=str(args.group),
            name=str(args.name),
            config=OmegaConf.to_container(args, resolve=True),
        )

    set_seed(int(args.seed))

    # ------------------------ paths ------------------------
    base_path = f"{args.pipeline_name}_H{args.task.planner_horizon}_Jump{args.task.stride}"
    base_path += f"_next{args.planner_next_obs_loss_weight}"
    base_path += f"_{args.guidance_type}"
    base_path += f"_{args.planner_net}"
    if args.planner_net == "transformer":
        base_path += f"_d{args.planner_depth}"
        base_path += f"_width{args.planner_d_model}"
    elif args.planner_net == "unet":
        base_path += f"_width{args.unet_dim}"
    if not args.planner_predict_noise:
        base_path += "_pred_x0"
    base_path += f"_{args.pipeline_type}"
    base_path += f"_dp{args.use_diffusion_invdyn}"
    base_path += f"/{args.task.env_name}/"

    save_path = f"{args.save_dir}/" + base_path
    os.makedirs(save_path, exist_ok=True)

    # ------------------------ dataset/env ------------------------
    env = gym.make(args.task.env_name)
    env_dataset = env.get_dataset()

    # NOTE: AntMaze uses reward_mode in your training code; keep it here.
    planner_dataset = DV_D4RLAntmazeSeqDataset(
        env_dataset,
        horizon=args.task.planner_horizon,
        discount=args.reward_mode.discount,
        continous_reward_at_done=args.reward_mode.continous_reward_at_done,
        reward_tune=args.reward_mode.reward_tune,
        stride=args.task.stride,
        learn_policy=False,
        center_mapping=True,  # MCSS
    )
    normalizer = planner_dataset.get_normalizer()
    obs_dim, act_dim = planner_dataset.o_dim, planner_dataset.a_dim
    planner_dim = obs_dim  # separate pipeline => state-only planner

    # ------------------------ networks ------------------------
    if args.planner_net == "transformer":
        nn_diffusion_planner = DiT1d(
            planner_dim,
            emb_dim=args.planner_emb_dim,
            d_model=args.planner_d_model,
            n_heads=max(1, args.planner_d_model // 64),
            depth=args.planner_depth,
            timestep_emb_type="fourier",
        )
    elif args.planner_net == "unet":
        nn_diffusion_planner = JannerUNet1d(
            planner_dim,
            model_dim=args.unet_dim,
            emb_dim=args.unet_dim,
            timestep_emb_type="positional",
            attention=False,
            kernel_size=5,
        )
    else:
        raise ValueError(f"Invalid planner_net: {args.planner_net}")

    critic = DVHorizonCritic(
        planner_dim,
        emb_dim=args.planner_emb_dim,
        d_model=args.planner_d_model,
        n_heads=max(1, args.planner_d_model // 64),
        depth=2,
        norm_type="pre",
    ).to(args.device)

    print("=============== Parameter Report of Value ====================================")
    report_parameters(critic)
    print("==============================================================================")

    print("=============== Parameter Report of Planner ==================================")
    report_parameters(nn_diffusion_planner)
    print("==============================================================================")

    # ------------------------ diffusion planner ------------------------
    fix_mask = torch.zeros((int(args.task.planner_horizon), planner_dim))
    fix_mask[0, :obs_dim] = 1.0
    loss_weight = torch.ones((int(args.task.planner_horizon), planner_dim))
    loss_weight[1] = args.planner_next_obs_loss_weight

    planner = ContinuousDiffusionSDE(
        nn_diffusion_planner,
        nn_condition=None,
        fix_mask=fix_mask,
        loss_weight=loss_weight,
        classifier=None,
        ema_rate=args.planner_ema_rate,
        device=args.device,
        predict_noise=args.planner_predict_noise,
        noise_schedule="linear",
    )

    # ------------------------ diffusion inverse dynamics policy ------------------------
    nn_diffusion_invdyn = DVInvMlp(
        obs_dim,
        act_dim,
        emb_dim=64,
        hidden_dim=args.policy_hidden_dim,
        timestep_emb_type="positional",
    ).to(args.device)
    nn_condition_invdyn = IdentityCondition(dropout=0.0).to(args.device)

    print("=============== Parameter Report of Policy ===================================")
    report_parameters(nn_diffusion_invdyn)
    print("==============================================================================")

    policy = DiscreteDiffusionSDE(
        nn_diffusion_invdyn,
        nn_condition_invdyn,
        predict_noise=args.policy_predict_noise,
        optim_params={"lr": args.policy_learning_rate},
        x_max=+1.0 * torch.ones((1, act_dim), device=args.device),
        x_min=-1.0 * torch.ones((1, act_dim), device=args.device),
        diffusion_steps=args.policy_diffusion_steps,
        ema_rate=args.policy_ema_rate,
        device=args.device,
    )

    # ------------------------ load ckpts ------------------------
    planner.load(save_path + f"planner_ckpt_{args.planner_ckpt}.pt")
    planner.eval()

    critic_ckpt = torch.load(save_path + f"critic_ckpt_{args.critic_ckpt}.pt", map_location=args.device)
    critic.load_state_dict(critic_ckpt["critic"])
    critic.eval()

    policy.load(save_path + f"policy_ckpt_{args.policy_ckpt}.pt")
    policy.eval()

    # ------------------------ SAGE init (ALWAYS ON) ------------------------
    sage_prefix = int(getattr(args, "sage_prefix", 10))
    sage_keep_p = float(getattr(args, "sage_keep_p", 0.8))
    sage_lambda = float(getattr(args, "sage_lambda", 0.1))

    dataset_obs_np = env_dataset["observations"].astype(np.float32)
    dataset_act_np = env_dataset["actions"].astype(np.float32)

    sage = SAGEEnergyScorer(
        device=torch.device(args.device),
        obs_dim=obs_dim,
        act_dim=act_dim,
        encoder_ckpt=str(getattr(args, "sage_encoder_ckpt")),
        state_stats_path=str(getattr(args, "sage_state_stats")),
        ac_ckpt=str(getattr(args, "sage_ac_ckpt")),
        # AntMaze actions from diffusion policy are already in [-1,1]; don't squash again
        actions_tanh=bool(getattr(args, "sage_actions_tanh", False)),
        apply_state_stats=bool(getattr(args, "sage_apply_state_stats", True)),
        apply_action_stats=bool(getattr(args, "sage_apply_action_stats", True)),
        dataset_actions_np=dataset_act_np,
        dataset_obs_np=dataset_obs_np,
        # planner operates in normalized state space => unnormalize before JEPA stats
        input_state_normalizer=normalizer,
        input_states_are_normalized=True,
    )

    # ------------------------ eval loop ------------------------
    env_eval = gym.vector.make(args.task.env_name, int(args.num_envs))
    episode_rewards = []

    for _ in range(int(args.num_episodes)):
        obs = env_eval.reset()
        ep_reward = np.zeros(int(args.num_envs), dtype=np.float32)
        cum_done = np.zeros(int(args.num_envs), dtype=bool)
        t = 0

        while (not np.all(cum_done)) and (t < int(args.task.max_path_length) + 1):
            # ---- 1) sample C candidates from planner (state-only) ----
            C = int(args.planner_num_candidates)
            planner_prior = torch.zeros(
                (int(args.num_envs) * C, int(args.task.planner_horizon), planner_dim),
                device=args.device,
            )

            obs_t = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)
            obs_repeat = obs_t.unsqueeze(1).repeat(1, C, 1).view(-1, obs_dim)
            planner_prior[:, 0, :obs_dim] = obs_repeat

            traj_flat, _ = planner.sample(
                planner_prior,
                solver=args.planner_solver,
                n_samples=int(args.num_envs) * C,
                sample_steps=args.planner_sampling_steps,
                use_ema=args.planner_use_ema,
                condition_cfg=None,
                w_cfg=1.0,
                temperature=args.task.planner_temperature,
            )  # [N*C, H, obs_dim]

            traj_cand = traj_flat.view(int(args.num_envs), C, int(args.task.planner_horizon), planner_dim)

            # ---- 2) utility J from critic ----
            with torch.no_grad():
                J_flat = critic(traj_flat)
                if J_flat.ndim > 1:
                    J_flat = J_flat.view(J_flat.shape[0], -1).mean(dim=1)
                J = J_flat.view(int(args.num_envs), C)

            # ---- 3) prefix actions via diffusion inverse dynamics (policy) ----
            K = sage_prefix
            a_hat = _infer_prefix_actions_for_state_only_plans(
                traj_flat=traj_flat[..., :obs_dim],  # normalized
                K=K,
                obs_dim=obs_dim,
                act_dim=act_dim,
                args=args,
                policy=policy,  # ALWAYS diffusion inverse dynamics here
                invdyn=None,
            )  # [N*C, K, act_dim]

            # ---- 4) prefix energy E + SAGE selection ----
            E_flat = sage.compute_energy_from_traj(
                traj_flat,
                K=K,
                obs_dim=obs_dim,
                planner_dim=planner_dim,
                actions_override=a_hat,
            )  # [N*C]
            E = E_flat.view(int(args.num_envs), C)

            traj_sel = _select_with_sage(
                J=J,
                E=E,
                traj=traj_cand,
                keep_p=sage_keep_p,
                lam=sage_lambda,
            )  # [N, H, obs_dim]

            # ---- 5) execute first action via diffusion inverse dynamics (policy) ----
            next_obs_plan = traj_sel[:, 1, :obs_dim]
            obs_policy = obs_t.clone()
            next_obs_policy = next_obs_plan.clone()

            if bool(getattr(args, "rebase_policy", False)):
                next_obs_policy[:, :2] -= obs_policy[:, :2]
                obs_policy[:, :2] = 0.0

            prior = torch.zeros((int(args.num_envs), act_dim), device=args.device)
            act_t, _ = policy.sample(
                prior,
                solver=args.policy_solver,
                n_samples=int(args.num_envs),
                sample_steps=args.policy_sampling_steps,
                condition_cfg=torch.cat([obs_policy, next_obs_policy], dim=-1),
                w_cfg=1.0,
                use_ema=args.policy_use_ema,
                temperature=args.policy_temperature,
            )
            act = act_t.cpu().numpy()

            # ---- env step ----
            obs, rew, done, info = env_eval.step(act)

            t += 1
            cum_done = np.logical_or(cum_done, done)
            ep_reward += rew.astype(np.float32)

            print(f"[t={t}] rew: {np.around(rew, 3)}")

        # AntMaze: clip to [0,1] like your baseline
        episode_rewards.append(np.clip(ep_reward, 0.0, 1.0))

    # normalize (match your baseline style)
    episode_rewards = [list(map(lambda x: env.get_normalized_score(x), r)) for r in episode_rewards]
    episode_rewards = np.array(episode_rewards).reshape(-1) * 100.0
    mean = float(np.mean(episode_rewards))
    err = float(np.std(episode_rewards) / np.sqrt(len(episode_rewards)))
    print(mean, err)

    if bool(getattr(args, "enable_wandb", False)):
        wandb.log(
            {
                "Mean Reward": mean,
                "Error": err,
                "sage/enabled": 1,
                "sage/prefix": sage_prefix,
                "sage/keep_p": sage_keep_p,
                "sage/lambda": sage_lambda,
            }
        )
        wandb.finish()


if __name__ == "__main__":
    pipeline()
