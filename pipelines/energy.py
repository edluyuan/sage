from __future__ import annotations

# energy.py
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

# ------------------------ SAGE (JEPA) utils ------------------------
from jepa.utils import Stats, load_stats, Encoder, ACTinyTransformer, compute_latent_stats

def _torch_unnormalize_states(x: torch.Tensor, normalizer: Any) -> torch.Tensor:
    """
    Best-effort inverse of planner_dataset normalizer, but in torch.
    Supports:
      - normalizer.unnormalize(x) if it works on torch
      - otherwise uses (mean,std) / (mu,sigma) / (obs_mean,obs_std) attributes
    """
    # 1) try method
    if hasattr(normalizer, "unnormalize"):
        try:
            y = normalizer.unnormalize(x)
            if isinstance(y, np.ndarray):
                return torch.as_tensor(y, device=x.device, dtype=x.dtype)
            if torch.is_tensor(y):
                return y.to(device=x.device, dtype=x.dtype)
        except Exception:
            pass

    # 2) attribute-based fallback
    mean = None
    std = None
    for m_name in ["mean", "mu", "obs_mean"]:
        if hasattr(normalizer, m_name):
            mean = getattr(normalizer, m_name)
            break
    for s_name in ["std", "sigma", "obs_std"]:
        if hasattr(normalizer, s_name):
            std = getattr(normalizer, s_name)
            break

    if mean is None or std is None:
        raise ValueError(
            "Cannot unnormalize states: normalizer must provide `.unnormalize()` "
            "or attributes (mean,std)/(mu,sigma)/(obs_mean,obs_std)."
        )

    mean_t = torch.as_tensor(mean, device=x.device, dtype=x.dtype)
    std_t = torch.as_tensor(std, device=x.device, dtype=x.dtype)
    return x * std_t + mean_t


class SAGEEnergyScorer:
    """
    SAGE prefix energy:
      E(tau) = (1/K) * sum_{k=0..K-1} || f(z_k, a_k) - z_{k+1} ||_1

    Key compatibility feature:
    - If your planner operates in a normalized state space but JEPA was trained on raw states,
      pass `input_state_normalizer=planner_dataset.get_normalizer()` and keep
      `input_states_are_normalized=True`. We will unnormalize internally before applying JEPA stats.
    """
    def __init__(
        self,
        device: torch.device,
        obs_dim: int,
        act_dim: int,
        encoder_ckpt: str,
        state_stats_path: str,
        ac_ckpt: str,
        actions_tanh: bool = True,
        apply_state_stats: bool = True,
        apply_action_stats: bool = True,
        dataset_actions_np: Optional[np.ndarray] = None,
        dataset_obs_np: Optional[np.ndarray] = None,
        # ---- NEW: state representation bridge ----
        input_state_normalizer: Optional[Any] = None,
        input_states_are_normalized: bool = True,
    ):
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.actions_tanh = actions_tanh
        self.apply_state_stats = apply_state_stats
        self.apply_action_stats = apply_action_stats

        # NEW
        self.input_state_normalizer = input_state_normalizer
        self.input_states_are_normalized = bool(input_states_are_normalized)

        assert os.path.isfile(encoder_ckpt), f"Missing encoder_ckpt: {encoder_ckpt}"
        assert os.path.isfile(state_stats_path), f"Missing state_stats: {state_stats_path}"
        assert os.path.isfile(ac_ckpt), f"Missing ac_ckpt: {ac_ckpt}"

        # ---- stats (as used during JEPA training) ----
        self.s_stats: Stats = load_stats(state_stats_path)
        self.m_s, self.s_s = self.s_stats.to_torch(device)

        # ---- load AC ckpt (predictor + args) ----
        ac_payload = torch.load(ac_ckpt, map_location=device)
        assert isinstance(ac_payload, dict) and "predictor" in ac_payload, \
            "AC ckpt must be dict with key 'predictor'"
        self.ac_args = ac_payload.get("args", {}) if isinstance(ac_payload, dict) else {}

        # ---- encoder ----
        embed_dim = int(self.ac_args.get("embed_dim", 256))
        enc_hidden = int(self.ac_args.get("enc_hidden", 512))
        enc_layers = int(self.ac_args.get("enc_layers", 3))

        self.encoder = Encoder(obs_dim, embed_dim=embed_dim, hidden=enc_hidden, layers=enc_layers).to(device)
        enc_sd = torch.load(encoder_ckpt, map_location=device)  # encoder_ema.pt often bare state_dict
        self.encoder.load_state_dict(enc_sd)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        # ---- predictor ----
        hidden = int(self.ac_args.get("hidden", 256))
        layers = int(self.ac_args.get("layers", 2))
        nhead = int(self.ac_args.get("nhead", 4))
        dropout = float(self.ac_args.get("dropout", 0.0))
        use_s_token = bool(self.ac_args.get("use_s_token", False))
        delta_pred = bool(self.ac_args.get("delta_pred", True))

        self.predictor = ACTinyTransformer(
            z_dim=embed_dim,
            s_dim=obs_dim,
            a_dim=act_dim,
            hidden=hidden,
            layers=layers,
            nhead=nhead,
            use_s_token=use_s_token,
            delta_pred=delta_pred,
            dropout=dropout,
            max_T=1024,
        ).to(device)
        self.predictor.load_state_dict(ac_payload["predictor"])
        self.predictor.eval()
        for p in self.predictor.parameters():
            p.requires_grad_(False)

        # ---- latent whitening (match ac.py) ----
        self.latent_whiten = bool(self.ac_args.get("latent_whiten", True))
        if self.latent_whiten:
            assert dataset_obs_np is not None, "Need dataset_obs_np to compute latent whitening stats"
            z_mu, z_std = compute_latent_stats(self.encoder, dataset_obs_np, self.s_stats, device)
            self.z_mu = z_mu
            self.z_std = z_std
        else:
            self.z_mu = torch.zeros(embed_dim, device=device)
            self.z_std = torch.ones(embed_dim, device=device)

        # ---- action whitening (match ac.py) ----
        # IMPORTANT FIX:
        # stats must be computed in the same action space used to feed predictor.
        # If actions_tanh=True, predictor sees tanh(atanh(a)) -> [-1,1], so whiten on tanh(actions_np).
        self.action_whiten = bool(self.ac_args.get("action_whiten", True))
        if self.action_whiten and self.apply_action_stats:
            assert dataset_actions_np is not None, "Need dataset_actions_np to compute action whitening stats"
            acts = dataset_actions_np.astype(np.float32)
            if self.actions_tanh:
                acts = np.tanh(acts)
            self.a_stats = Stats.from_array(acts)
            self.m_a, self.s_a = self.a_stats.to_torch(device)
        else:
            self.a_stats = None
            self.m_a, self.s_a = None, None

    def set_input_state_normalizer(self, normalizer: Any, input_states_are_normalized: bool = True) -> None:
        self.input_state_normalizer = normalizer
        self.input_states_are_normalized = bool(input_states_are_normalized)

    @torch.no_grad()
    def compute_energy_from_traj(
        self,
        traj: torch.Tensor,          # [B, H, planner_dim]
        K: int,
        obs_dim: int,
        planner_dim: int,
        actions_override: Optional[torch.Tensor] = None,  # [B, K, act_dim]
        states_override: Optional[torch.Tensor] = None,   # [B, K+1, obs_dim]
        # NEW: per-call override if you ever need it
        states_are_normalized: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Returns E: [B]
        """
        K = int(K)
        assert K >= 1, "K must be >= 1"

        # states: [B, K+1, obs_dim]
        if states_override is None:
            s = traj[:, : K + 1, :obs_dim]
        else:
            s = states_override[:, : K + 1, :obs_dim]

        # ---- CENTRAL FIX: unnormalize planner-space -> raw-space (if configured) ----
        norm_flag = self.input_states_are_normalized if states_are_normalized is None else bool(states_are_normalized)
        if norm_flag and (self.input_state_normalizer is not None):
            s = _torch_unnormalize_states(s, self.input_state_normalizer)

        # actions: [B, K, act_dim]
        if actions_override is not None:
            a = actions_override[:, :K, :]
        else:
            assert planner_dim > obs_dim, "No actions in traj; pass actions_override for state-only planner."
            a = traj[:, :K, obs_dim : obs_dim + self.act_dim]

        # if planner stores actions in atanh-space, map back
        if self.actions_tanh:
            a = torch.tanh(a)

        # apply stats (as in JEPA/AC training)
        s_in = (s - self.m_s) / self.s_s if self.apply_state_stats else s

        if self.a_stats is not None and self.apply_action_stats:
            a_in = (a - self.m_a) / self.s_a
        else:
            a_in = a

        # encode latents z: [B, K+1, Dz]
        B, Tp1, Ds = s_in.shape
        z = self.encoder(s_in.reshape(B * Tp1, Ds)).view(B, Tp1, -1)
        z = (z - self.z_mu) / self.z_std

        # predict next latents: [B, K, Dz]
        if bool(self.ac_args.get("use_s_token", False)):
            z_pred = self.predictor.forward_teacher(z, a_in, s_in)
        else:
            z_pred = self.predictor.forward_teacher(z, a_in, None)

        step_err = (z_pred - z[:, 1:, :]).abs().mean(dim=-1)  # [B, K]
        return step_err.mean(dim=-1)                          # [B]


def _select_with_sage(J: torch.Tensor, E: torch.Tensor, traj: torch.Tensor, keep_p: float, lam: float) -> torch.Tensor:
    Nenv, C = J.shape
    keep = max(1, int(math.ceil(float(keep_p) * C)))

    _, idx_sorted = torch.sort(E, dim=1, descending=False)
    idx_keep = idx_sorted[:, :keep]

    J_keep = J.gather(1, idx_keep)
    E_keep = E.gather(1, idx_keep)

    score = J_keep - float(lam) * E_keep
    best_in_keep = torch.argmax(score, dim=1)
    best_idx = idx_keep.gather(1, best_in_keep.unsqueeze(1)).squeeze(1)

    return traj[torch.arange(Nenv, device=traj.device), best_idx]
                         # [B]


def _select_with_sage(
    J: torch.Tensor,        # [Nenv, C]
    E: torch.Tensor,        # [Nenv, C]
    traj: torch.Tensor,     # [Nenv, C, H, D]
    keep_p: float,
    lam: float,
) -> torch.Tensor:
    """
    SAGE selection: keep lowest-energy subset then select best by (J - lam*E).
    Returns: [Nenv, H, D]
    """
    Nenv, C = J.shape
    keep = max(1, int(math.ceil(float(keep_p) * C)))

    _, idx_sorted = torch.sort(E, dim=1, descending=False)  # [Nenv, C]
    idx_keep = idx_sorted[:, :keep]                         # [Nenv, keep]

    J_keep = J.gather(1, idx_keep)
    E_keep = E.gather(1, idx_keep)

    score = J_keep - float(lam) * E_keep
    best_in_keep = torch.argmax(score, dim=1)               # [Nenv]
    best_idx = idx_keep.gather(1, best_in_keep.unsqueeze(1)).squeeze(1)

    return traj[torch.arange(Nenv, device=traj.device), best_idx]


@torch.no_grad()
def _infer_prefix_actions_for_state_only_plans(
    traj_flat: torch.Tensor,     # [B, H, obs_dim]
    K: int,
    obs_dim: int,
    act_dim: int,
    args,
    policy: Optional[DiscreteDiffusionSDE],
    invdyn: Optional[MlpInvDynamic],
) -> torch.Tensor:
    """
    Given state-only plan (normalized representation used by planner/policy),
    infer actions a_k that take s_k -> s_{k+1}, for k=0..K-1.

    Returns: [B, K, act_dim]
    """
    B, H, D = traj_flat.shape
    assert D == obs_dim, "traj_flat must be state-only: [B, H, obs_dim]"
    assert H >= K + 1

    s0 = traj_flat[:, :K, :]       # [B, K, obs_dim]
    s1 = traj_flat[:, 1 : K + 1, :]  # [B, K, obs_dim]

    # optional rebase (match action generation path)
    if bool(getattr(args, "rebase_policy", False)):
        s1 = s1.clone()
        s0 = s0.clone()
        s1[..., :2] -= s0[..., :2]
        s0[..., :2] = 0.0

    cond = torch.cat([s0.reshape(-1, obs_dim), s1.reshape(-1, obs_dim)], dim=-1)  # [B*K, 2*obs_dim]

    if policy is not None:
        prior = torch.zeros((cond.shape[0], act_dim), device=args.device)
        a_hat, _ = policy.sample(
            prior,
            solver=args.policy_solver,
            n_samples=cond.shape[0],
            sample_steps=args.policy_sampling_steps,
            condition_cfg=cond,
            w_cfg=1.0,
            use_ema=args.policy_use_ema,
            temperature=args.policy_temperature,
        )
        return a_hat.view(B, K, act_dim)

    assert invdyn is not None, "Need either diffusion policy or deterministic invdyn to infer actions."
    a_hat = invdyn.predict(cond[:, :obs_dim], cond[:, obs_dim:])  # typically predict(s, s_next)
    return a_hat.view(B, K, act_dim)