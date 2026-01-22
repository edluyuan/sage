# sage_d4rl_mujoco.py
import os
import math
import uuid
from typing import Optional, Tuple

import d4rl
import gym
import hydra
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf

from cleandiffuser.classifier import CumRewClassifier
from cleandiffuser.dataset.d4rl_mujoco_dataset import DV_D4RLMuJoCoSeqDataset
from cleandiffuser.diffusion import ContinuousDiffusionSDE, DiscreteDiffusionSDE
from cleandiffuser.invdynamic import MlpInvDynamic
from cleandiffuser.nn_condition import IdentityCondition, MLPCondition
from cleandiffuser.nn_classifier import HalfJannerUNet1d
from cleandiffuser.nn_diffusion import DiT1d, DVInvMlp, JannerUNet1d
from cleandiffuser.utils import DVHorizonCritic, IDQLVNet, report_parameters
from utils import set_seed
import os, sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


from energy import SAGEEnergyScorer, _select_with_sage, _infer_prefix_actions_for_state_only_plans

# ------------------------ SAGE (JEPA) utils ------------------------
from jepa.utils import Stats, load_stats, Encoder, ACTinyTransformer, compute_latent_stats

@hydra.main(config_path="../configs/veteran/mujoco", config_name="mujoco", version_base=None)
def pipeline(args):
    # ------------------------ setup ------------------------
    args.device = args.device if torch.cuda.is_available() else "cpu"
    assert str(args.mode) == "inference"
    assert str(args.guidance_type) == "MCSS"
    assert str(args.pipeline_type) == "separate"
    assert bool(args.use_diffusion_invdyn), "inverse dynamics"


    if bool(args.enable_wandb):
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

    # base config/path (kept identical style)
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
    base_path += f"_penalty{args.terminal_penalty}"
    base_path += f"_bonus{args.full_traj_bonus}"
    base_path += f"_gamma{args.discount}"
    base_path += f"_adv{args.use_weighted_regression}"
    base_path += f"_weight{args.weight_factor}"
    base_path += f"/{args.task.env_name}/"

    save_path = f"{args.save_dir}/" + base_path
    os.makedirs(save_path, exist_ok=True)

    # ------------------------ dataset/env ------------------------
    env = gym.make(args.task.env_name)
    env_dataset = env.get_dataset()

    planner_dataset = DV_D4RLMuJoCoSeqDataset(
        env_dataset,
        horizon=args.task.planner_horizon,
        discount=args.discount,
        stride=args.task.stride,
        center_mapping=(args.guidance_type != "cfg"),
        terminal_penalty=args.terminal_penalty,
        full_traj_bonus=args.full_traj_bonus,
    )
    obs_dim, act_dim = planner_dataset.o_dim, planner_dataset.a_dim
    planner_dim = obs_dim if args.pipeline_type == "separate" else obs_dim + act_dim

    # ------------------------ networks ------------------------
    if args.planner_net == "transformer":
        nn_diffusion_planner = DiT1d(
            planner_dim,
            emb_dim=args.planner_emb_dim,
            d_model=args.planner_d_model,
            n_heads=args.planner_d_model // 32,
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

    nn_condition_planner = None
    classifier = None

    critic = None
    if args.guidance_type == "MCSS":
        critic = DVHorizonCritic(
            planner_dim,
            emb_dim=args.planner_emb_dim,
            d_model=args.planner_d_model,
            n_heads=args.planner_d_model // 32,
            depth=2,
            norm_type="pre",
        ).to(args.device)
        print("=============== Parameter Report of Value ====================================")
        report_parameters(critic)
        print("==============================================================================")
    elif args.guidance_type == "cfg":
        if args.planner_net == "transformer":
            nn_condition_planner = MLPCondition(
                in_dim=1, out_dim=args.planner_emb_dim, hidden_dims=[args.planner_emb_dim], act=torch.nn.SiLU(), dropout=0.25
            )
        else:
            nn_condition_planner = MLPCondition(
                in_dim=1, out_dim=args.unet_dim, hidden_dims=[args.unet_dim], act=torch.nn.SiLU(), dropout=0.25
            )
    elif args.guidance_type == "cg":
        nn_classifier = HalfJannerUNet1d(
            args.task.planner_horizon,
            planner_dim,
            out_dim=1,
            model_dim=args.unet_dim,
            emb_dim=args.unet_dim,
            timestep_emb_type="positional",
            kernel_size=3,
        )
        classifier = CumRewClassifier(nn_classifier, device=args.device)
        print("=============== Parameter Report of Classifier ===============================")
        report_parameters(nn_classifier)
        print("==============================================================================")
    else:
        raise ValueError(f"Invalid guidance_type: {args.guidance_type}")

    print("=============== Parameter Report of Planner ==================================")
    report_parameters(nn_diffusion_planner)
    print("==============================================================================")

    fix_mask = torch.zeros((args.task.planner_horizon, planner_dim))
    fix_mask[0, :obs_dim] = 1.0
    loss_weight = torch.ones((args.task.planner_horizon, planner_dim))
    loss_weight[1] = args.planner_next_obs_loss_weight

    planner = ContinuousDiffusionSDE(
        nn_diffusion_planner,
        nn_condition=nn_condition_planner,
        fix_mask=fix_mask,
        loss_weight=loss_weight,
        classifier=classifier,
        ema_rate=args.planner_ema_rate,
        device=args.device,
        predict_noise=args.planner_predict_noise,
        noise_schedule="linear",
    )

    # policy / invdyn (used for execution AND for SAGE action-inference on state-only plans)
    policy = None
    invdyn = None
    if args.pipeline_type == "separate":
        if args.use_diffusion_invdyn:
            nn_diffusion_invdyn = DVInvMlp(
                obs_dim, act_dim, emb_dim=64, hidden_dim=args.policy_hidden_dim, timestep_emb_type="positional"
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
        else:
            invdyn = MlpInvDynamic(obs_dim, act_dim, 512, torch.nn.Tanh(), {"lr": 2e-4}, device=args.device)

    # ------------------------ load ckpts ------------------------
    planner.load(save_path + f"planner_ckpt_{args.planner_ckpt}.pt")
    planner.eval()

    if args.guidance_type == "MCSS":
        assert critic is not None
        critic_ckpt = torch.load(save_path + f"critic_ckpt_{args.critic_ckpt}.pt", map_location=args.device)
        critic.load_state_dict(critic_ckpt["critic"])
        critic.eval()

    if args.guidance_type == "cg":
        planner.classifier.load(save_path + f"classifier_ckpt_{args.planner_ckpt}.pt")
        planner.eval()

    if args.pipeline_type == "separate":
        if args.use_diffusion_invdyn:
            assert policy is not None
            policy.load(save_path + f"policy_ckpt_{args.policy_ckpt}.pt")
            policy.eval()
        else:
            assert invdyn is not None
            invdyn.load(save_path + f"invdyn_ckpt_{args.invdyn_ckpt}.pt")
            invdyn.eval()

    # EV (used for MCSS selection score)
    MAX_VALUE_STEPS = 1_000_000
    EV = IDQLVNet(obs_dim, hidden_dim=256).to(args.device)
    ev_ckpt = torch.load(save_path + f"EV_ckpt_{MAX_VALUE_STEPS}.pt", map_location=args.device)
    EV.load_state_dict(ev_ckpt["ev"])
    EV.eval()

    # ------------------------ SAGE init (ALWAYS ON for inference) ------------------------
    # You can still set these in config; defaults are chosen to match your earlier code.
    sage_prefix = int(getattr(args, "sage_prefix", 4))
    sage_keep_p = float(getattr(args, "sage_keep_p", 0.9))
    sage_lambda = float(getattr(args, "sage_lambda", 1.0))

    # Representation note:
    # - If JEPA/AC were trained on the same *normalized* state representation as planner/policy, keep as-is.
    # - If not, you should adapt dataset_obs_np / traj states consistently.
    dataset_obs_np = env_dataset["observations"].astype(np.float32)
    dataset_act_np = env_dataset["actions"].astype(np.float32)

    sage = SAGEEnergyScorer(
        device=torch.device(args.device),
        obs_dim=obs_dim,
        act_dim=act_dim,
        encoder_ckpt=str(getattr(args, "sage_encoder_ckpt")),
        state_stats_path=str(getattr(args, "sage_state_stats")),
        ac_ckpt=str(getattr(args, "sage_ac_ckpt")),
        actions_tanh=bool(getattr(args, "sage_actions_tanh", True)),
        apply_state_stats=bool(getattr(args, "sage_apply_state_stats", True)),
        apply_action_stats=bool(getattr(args, "sage_apply_action_stats", True)),
        dataset_actions_np=dataset_act_np,
        dataset_obs_np=dataset_obs_np,
    )

    # ------------------------ eval loop ------------------------
    env_eval = gym.vector.make(args.task.env_name, args.num_envs)
    normalizer = planner_dataset.get_normalizer()

    episode_rewards = []

    for i in range(int(args.num_episodes)):
        obs, ep_reward, cum_done, t = env_eval.reset(), 0.0, 0.0, 0

        while not np.all(cum_done) and t < args.task.max_path_length + 1:

            # -------- 1) generate candidates / plan --------
            if args.guidance_type in ["MCSS", "cg"]:
                C = int(args.planner_num_candidates)
                planner_prior = torch.zeros(
                    (args.num_envs * C, args.task.planner_horizon, planner_dim), device=args.device
                )

                obs_t = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)
                obs_repeat = obs_t.unsqueeze(1).repeat(1, C, 1).view(-1, obs_dim)
                planner_prior[:, 0, :obs_dim] = obs_repeat

                if args.guidance_type == "MCSS":
                    traj_flat, _ = planner.sample(
                        planner_prior,
                        solver=args.planner_solver,
                        n_samples=args.num_envs * C,
                        sample_steps=args.planner_sampling_steps,
                        use_ema=args.planner_use_ema,
                        condition_cfg=None,
                        w_cfg=1.0,
                        temperature=args.task.planner_temperature,
                    )
                    # baseline score J: EV-sum over horizon (use only obs part)
                    v_td = EV(traj_flat[..., :obs_dim])[:, 1:]     # [N*C, H-1]
                    J_flat = v_td.sum(dim=1)                      # [N*C]
                    J = J_flat.view(args.num_envs, C)

                else:  # cg
                    traj_flat, log = planner.sample(
                        planner_prior,
                        solver=args.planner_solver,
                        n_samples=args.num_envs * C,
                        sample_steps=args.planner_sampling_steps,
                        use_ema=args.planner_use_ema,
                        w_cg=args.task.planner_w_cfg,
                        temperature=args.task.planner_temperature,
                    )
                    J = log["log_p"].view(args.num_envs, C)

                traj = traj_flat.view(args.num_envs, C, args.task.planner_horizon, planner_dim)

                # -------- 1b) compute SAGE energy E for every candidate (ALWAYS) --------
                K = sage_prefix
                if planner_dim > obs_dim:
                    E_flat = sage.compute_energy_from_traj(
                        traj_flat, K=K, obs_dim=obs_dim, planner_dim=planner_dim
                    )
                else:
                    # state-only planner => infer actions via invdyn/policy for each prefix transition
                    a_hat = _infer_prefix_actions_for_state_only_plans(
                        traj_flat=traj_flat[..., :obs_dim],
                        K=K,
                        obs_dim=obs_dim,
                        act_dim=act_dim,
                        args=args,
                        policy=policy,
                        invdyn=invdyn,
                    )
                    E_flat = sage.compute_energy_from_traj(
                        traj_flat, K=K, obs_dim=obs_dim, planner_dim=planner_dim, actions_override=a_hat
                    )

                E = E_flat.view(args.num_envs, C)

                # -------- 1c) SAGE selection --------
                traj_sel = _select_with_sage(
                    J=J, E=E, traj=traj, keep_p=sage_keep_p, lam=sage_lambda
                )

            elif args.guidance_type == "cfg":
                # cfg samples one per env; SAGE doesn’t change anything unless you turn cfg into multi-candidate sampling.
                planner_prior = torch.zeros((args.num_envs, args.task.planner_horizon, planner_dim), device=args.device)
                condition = torch.ones((args.num_envs, 1), device=args.device) * args.task.planner_target_return

                obs_t = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)
                planner_prior[:, 0, :obs_dim] = obs_t

                traj_sel, _ = planner.sample(
                    planner_prior,
                    solver=args.planner_solver,
                    n_samples=args.num_envs,
                    sample_steps=args.planner_sampling_steps,
                    use_ema=args.planner_use_ema,
                    condition_cfg=condition,
                    w_cfg=args.task.planner_w_cfg,
                    temperature=args.task.planner_temperature,
                )
            else:
                raise ValueError(f"Invalid guidance_type: {args.guidance_type}")

            # -------- 2) execute first action --------
            if args.pipeline_type == "separate":
                # policy acts on (obs_t, next_obs_plan) in the same representation as planning
                next_obs_plan = traj_sel[:, 1, :obs_dim]
                obs_policy = obs_t.clone()
                next_obs_policy = next_obs_plan.clone()

                if bool(getattr(args, "rebase_policy", False)):
                    next_obs_policy[:, :2] -= obs_policy[:, :2]
                    obs_policy[:, :2] = 0.0

                if args.use_diffusion_invdyn:
                    assert policy is not None
                    prior = torch.zeros((args.num_envs, act_dim), device=args.device)
                    act_t, _ = policy.sample(
                        prior,
                        solver=args.policy_solver,
                        n_samples=args.num_envs,
                        sample_steps=args.policy_sampling_steps,
                        condition_cfg=torch.cat([obs_policy, next_obs_policy], dim=-1),
                        w_cfg=1.0,
                        use_ema=args.policy_use_ema,
                        temperature=args.policy_temperature,
                    )
                    act = act_t.cpu().numpy()
                else:
                    assert invdyn is not None
                    act = invdyn.predict(obs_policy, next_obs_policy).cpu().numpy()
            else:
                # joint pipeline: planner directly outputs action at step 0
                act_t = traj_sel[:, 0, obs_dim : obs_dim + act_dim]
                act = act_t.cpu().numpy()
                if bool(getattr(args, "joint_actions_tanh", False)):
                    act = np.tanh(act)

            # env step
            obs, rew, done, info = env_eval.step(act)

            t += 1
            cum_done = done if cum_done is None else np.logical_or(cum_done, done)
            ep_reward += (rew * (1 - cum_done)) if t < args.task.max_path_length else rew
            print(f"[t={t}] rew: {np.around((rew * (1 - cum_done)), 2)}")

        episode_rewards.append(ep_reward)

    # normalize exactly like baseline code
    episode_rewards = [list(map(lambda x: env.get_normalized_score(x), r)) for r in episode_rewards]
    episode_rewards = np.array(episode_rewards).reshape(-1) * 100
    mean = float(np.mean(episode_rewards))
    err = float(np.std(episode_rewards) / np.sqrt(len(episode_rewards)))
    print(mean, err)

    if bool(args.enable_wandb):
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
