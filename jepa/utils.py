# utils.py
import os
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# -----------------------------
# Repro / perf
# -----------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def worker_init_fn(worker_id: int):
    # make each worker deterministic but different
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)

def enable_torch_perf():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


# -----------------------------
# LR schedule
# -----------------------------

def cosine_warmup_lr(step: int, total_steps: int, base_lr: float, min_lr: float, warmup_steps: int) -> float:
    warmup_steps = max(1, warmup_steps)
    if step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# -----------------------------
# D4RL loading + episodes
# -----------------------------

def load_d4rl(env_id: str, need_actions: bool = False):
    import gym
    import d4rl  # noqa: F401

    env = gym.make(env_id)
    ds = env.get_dataset() if hasattr(env, "get_dataset") else __import__("d4rl").qlearning_dataset(env)

    obs = ds["observations"].astype(np.float32)

    terminals = ds.get("terminals")
    if terminals is None:
        terminals = ds.get("dones")
    if terminals is None:
        terminals = np.zeros((len(obs),), dtype=np.bool_)
    terminals = np.asarray(terminals, dtype=np.bool_).reshape(-1)

    timeouts = ds.get("timeouts")
    if timeouts is None:
        timeouts = np.zeros_like(terminals, dtype=np.bool_)
    timeouts = np.asarray(timeouts, dtype=np.bool_).reshape(-1)

    dones = np.logical_or(terminals, timeouts).astype(np.bool_)

    episode_bounds: List[Tuple[int, int]] = []
    start = 0
    N = len(obs)
    for i in range(N - 1):
        if dones[i]:
            episode_bounds.append((start, i))
            start = i + 1
    if start < N:
        episode_bounds.append((start, N - 1))

    if not need_actions:
        return env, obs, dones, episode_bounds

    actions = ds["actions"].astype(np.float32)
    return env, obs, actions, dones, episode_bounds


# -----------------------------
# Normalization / whitening
# -----------------------------

@dataclass
class Stats:
    mean: np.ndarray
    std: np.ndarray

    @staticmethod
    def from_array(x: np.ndarray, eps: float = 1e-6) -> "Stats":
        mean = x.mean(axis=0, keepdims=True).astype(np.float32)
        std = x.std(axis=0, keepdims=True).astype(np.float32) + eps
        return Stats(mean=mean, std=std)

    def normalize_np(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def to_torch(self, device: torch.device):
        m = torch.from_numpy(self.mean).to(device)
        s = torch.from_numpy(self.std).to(device)
        return m, s

def save_stats(path: str, stats: Stats):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, mean=stats.mean, std=stats.std)

def load_stats(path: str) -> Stats:
    d = np.load(path)
    return Stats(mean=d["mean"], std=d["std"])

@torch.no_grad()
def compute_latent_stats(encoder: nn.Module, obs: np.ndarray, s_stats: Stats, device: torch.device, batch: int = 4096):
    N = obs.shape[0]
    zs = []
    m, s = s_stats.to_torch(device)
    for i in range(0, N, batch):
        x = torch.from_numpy(obs[i:i+batch]).float().to(device)
        x = (x - m) / s
        z = encoder(x)
        zs.append(z.detach().cpu())
    z_all = torch.cat(zs, dim=0)
    mu = z_all.mean(dim=0)
    std = z_all.std(dim=0)
    std[std < 1e-6] = 1e-6
    return mu.to(device), std.to(device)


# -----------------------------
# Datasets
# -----------------------------

class StateJEPADataset(Dataset):
    """
    Returns:
      ctx1: [W, Ds]   masked view 1
      ctx2: [W, Ds]   masked view 2
      targets: [M, Ds]  future state targets (unmasked)
      ks: [M]         integer steps ahead
    """
    def __init__(
        self,
        obs_norm: np.ndarray,
        episode_bounds: List[Tuple[int, int]],
        window: int = 16,
        k_max: int = 5,
        num_mask: int = 3,
        feature_mask_ratio: float = 0.3,
        time_mask_ratio: float = 0.1,
        dual_view_noise_std: float = 0.0,
    ):
        self.obs = obs_norm.astype(np.float32)
        self.window = int(window)
        self.k_max = int(k_max)
        self.num_mask = int(num_mask)
        self.feature_mask_ratio = float(feature_mask_ratio)
        self.time_mask_ratio = float(time_mask_ratio)
        self.dual_view_noise_std = float(dual_view_noise_std)

        self.starts: List[int] = []
        for (s, e) in episode_bounds:
            T = e - s + 1
            max_start = T - (self.window + self.k_max)
            if max_start < 0:
                continue
            for off in range(max_start + 1):
                self.starts.append(s + off)

    def __len__(self):
        return len(self.starts)

    def _mask_ctx(self, ctx: np.ndarray) -> np.ndarray:
        x = ctx.copy()
        W, D = x.shape

        if self.feature_mask_ratio > 0:
            drop_n = max(1, int(D * self.feature_mask_ratio))
            drop = np.random.choice(D, size=min(D, drop_n), replace=False)
            x[:, drop] = 0.0

        if self.time_mask_ratio > 0:
            drop_n = max(1, int(W * self.time_mask_ratio))
            drop = np.random.choice(W, size=min(W, drop_n), replace=False)
            x[drop, :] = 0.0

        return x

    def __getitem__(self, idx: int):
        start = self.starts[idx]
        W = self.window

        ctx = self.obs[start:start + W]  # [W, Ds]

        # sample ks
        if self.num_mask <= self.k_max:
            ks = np.random.choice(np.arange(1, self.k_max + 1), size=self.num_mask, replace=False)
        else:
            ks = np.random.choice(np.arange(1, self.k_max + 1), size=self.num_mask, replace=True)
        ks = np.sort(ks).astype(np.int64)

        targets = np.stack([self.obs[start + W - 1 + int(k)] for k in ks], axis=0)  # [M, Ds]

        ctx1 = self._mask_ctx(ctx)
        ctx2 = self._mask_ctx(ctx)

        if self.dual_view_noise_std > 0:
            nstd = self.dual_view_noise_std
            ctx1 = ctx1 + np.random.randn(*ctx1.shape).astype(np.float32) * nstd
            ctx2 = ctx2 + np.random.randn(*ctx2.shape).astype(np.float32) * nstd

        return (
            torch.from_numpy(ctx1),            # [W, Ds]
            torch.from_numpy(ctx2),            # [W, Ds]
            torch.from_numpy(targets),         # [M, Ds]
            torch.from_numpy(ks),              # [M]
        )

class ACWindowDataset(Dataset):
    """
    Returns:
      s: [W+1, Ds], a: [W, Da]
    """
    def __init__(self, obs: np.ndarray, actions: np.ndarray, episode_bounds: List[Tuple[int, int]], window: int = 16):
        self.obs = obs.astype(np.float32)
        self.actions = actions.astype(np.float32)
        self.window = int(window)

        self.starts: List[int] = []
        for (s, e) in episode_bounds:
            T = e - s + 1
            max_start = T - (self.window + 1)
            if max_start < 0:
                continue
            for off in range(max_start + 1):
                self.starts.append(s + off)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx: int):
        t0 = self.starts[idx]
        s = torch.from_numpy(self.obs[t0:t0 + self.window + 1]).float()      # [W+1, Ds]
        a = torch.from_numpy(self.actions[t0:t0 + self.window]).float()      # [W, Da]
        return s, a


# -----------------------------
# Models: shared MLP + Encoder
# -----------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, layers: int = 2, act=nn.GELU):
        super().__init__()
        dims = [in_dim] + [hidden] * (layers - 1) + [out_dim]
        mods = []
        for i in range(len(dims) - 2):
            mods += [nn.Linear(dims[i], dims[i + 1]), act(), nn.LayerNorm(dims[i + 1])]
        mods += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)

class Encoder(nn.Module):
    def __init__(self, state_dim: int, embed_dim: int = 256, hidden: int = 512, layers: int = 3):
        super().__init__()
        self.proj = MLP(state_dim, embed_dim, hidden=hidden, layers=layers)

    def forward(self, s):
        return self.proj(s)


# -----------------------------
# JEPA predictor (mask tokens)
# -----------------------------

class MaskedTokenPredictor(nn.Module):
    """
    Input: h_ctx [B,W,d], ks [B,M] -> Output: preds [B,M,d]
    """
    def __init__(self, d: int, nhead: int = 4, layers: int = 2, ff_mult: int = 4, max_pos: int = 4096, dropout: float = 0.0):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=nhead,
            dim_feedforward=ff_mult * d,
            batch_first=True,
            activation="gelu",
            norm_first=True,
            dropout=dropout,
        )
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.pos = nn.Embedding(max_pos, d)
        self.k_embed = nn.Embedding(max_pos, d)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d))

        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.k_embed.weight, std=0.02)

        self.head = nn.Sequential(
            nn.LayerNorm(d * 2),
            nn.Linear(d * 2, d * 2),
            nn.GELU(),
            nn.Linear(d * 2, d),
        )
        self.max_pos = max_pos

    def forward(self, h_ctx: torch.Tensor, ks: torch.Tensor) -> torch.Tensor:
        B, W, d = h_ctx.shape
        M = ks.shape[1]
        device = h_ctx.device

        pos_ctx = torch.arange(W, device=device).unsqueeze(0).expand(B, W)
        h_ctx = h_ctx + self.pos(pos_ctx)

        pos_mask = torch.clamp((W - 1) + ks, max=self.max_pos - 1)
        mask_tok = self.mask_token.expand(B, M, d) + self.pos(pos_mask)

        seq = torch.cat([h_ctx, mask_tok], dim=1)  # [B, W+M, d]
        out = self.tr(seq)
        mask_out = out[:, W:, :]                  # [B, M, d]
        k_emb = self.k_embed(ks)                  # [B, M, d]

        pred = self.head(torch.cat([mask_out, k_emb], dim=-1))
        return pred


class JEPAStateModel(nn.Module):
    def __init__(
        self,
        state_dim: int,
        embed_dim: int = 256,
        enc_hidden: int = 512,
        enc_layers: int = 3,
        ema_decay: float = 0.99,
        use_mask_token: bool = True,
        tr_dropout: float = 0.0,
        pred_nhead: int = 4,
        pred_layers: int = 2,
        pred_ff_mult: int = 4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_mask_token = use_mask_token

        self.encoder = Encoder(state_dim, embed_dim, hidden=enc_hidden, layers=enc_layers)
        self.encoder_ema = Encoder(state_dim, embed_dim, hidden=enc_hidden, layers=enc_layers)

        if use_mask_token:
            self.predictor = MaskedTokenPredictor(
                embed_dim,
                nhead=pred_nhead,
                layers=pred_layers,
                ff_mult=pred_ff_mult,
                max_pos=4096,
                dropout=tr_dropout,
            )
        else:
            self.predictor = MLP(embed_dim, embed_dim, hidden=enc_hidden, layers=2)

        self._ema_m = ema_decay
        self._init_ema()

    @torch.no_grad()
    def _init_ema(self):
        for p, p_ema in zip(self.encoder.parameters(), self.encoder_ema.parameters()):
            p_ema.data.copy_(p.data)
            p_ema.requires_grad_(False)

    @torch.no_grad()
    def update_ema(self, momentum: float):
        self._ema_m = momentum
        for p, p_ema in zip(self.encoder.parameters(), self.encoder_ema.parameters()):
            p_ema.data.mul_(momentum).add_(p.data, alpha=1.0 - momentum)

    def forward(self, ctx1, ctx2, targets, ks):
        # ctx*: [B,W,Ds], targets: [B,M,Ds], ks: [B,M]
        h1 = self.encoder(ctx1)  # [B,W,d]
        h2 = self.encoder(ctx2)

        if self.use_mask_token:
            pred1 = self.predictor(h1, ks)  # [B,M,d]
            pred2 = self.predictor(h2, ks)
        else:
            c1 = h1.mean(dim=1)
            c2 = h2.mean(dim=1)
            pred1 = self.predictor(c1).unsqueeze(1).expand(-1, ks.shape[1], -1)
            pred2 = self.predictor(c2).unsqueeze(1).expand(-1, ks.shape[1], -1)

        with torch.no_grad():
            B, M, Ds = targets.shape
            targ = self.encoder_ema(targets.view(B * M, Ds)).view(B, M, -1)

        return pred1, pred2, targ


# -----------------------------
# JEPA loss (BYOL/VICReg-L-ish)
# -----------------------------

def jepa_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    sim_coef=1.0,
    var_coef=1.0,
    cov_coef=0.1,
    norm_coef=0.05,
    eps=1e-4,
    var_upper=1.0,
    reg_on_target: bool = False,
    cov_reduce: str = "mean",
):
    """
    pred/target: [B,M,d] or [B,d]
    Returns: (total, sim, var_low, var_up, cov_pred, norm_pen, t_low, t_up, t_cov)
    """
    if pred.dim() == 3:
        B, M, d = pred.shape
        pred = pred.reshape(B * M, d)
        target = target.detach().reshape(B * M, d)
    else:
        target = target.detach()

    pred_n = F.normalize(pred, dim=-1)
    targ_n = F.normalize(target, dim=-1)
    sim = F.mse_loss(pred_n, targ_n)

    def variance_terms(x):
        std = torch.sqrt(x.var(dim=0) + eps)
        lower = F.relu(1.0 - std).pow(2).mean()
        upper = F.relu(std - var_upper).pow(2).mean() if var_upper is not None else torch.zeros((), device=x.device)
        return lower, upper

    def covariance_term(x):
        Bn, d = x.shape
        x = x - x.mean(dim=0)
        cov = (x.T @ x) / max(1, (Bn - 1))
        off_diag = cov - torch.diag(torch.diag(cov))
        if cov_reduce == "mean":
            return off_diag.pow(2).mean()
        return off_diag.pow(2).sum() / d

    var_low, var_up = variance_terms(pred)
    cov_pred = covariance_term(pred)

    energy = pred.pow(2).mean(dim=1).mean()
    norm_pen = (energy - 1.0).pow(2)

    if reg_on_target:
        t_low, t_up = variance_terms(target)
        t_cov = covariance_term(target)
    else:
        t_low = torch.zeros((), device=pred.device)
        t_up = torch.zeros((), device=pred.device)
        t_cov = torch.zeros((), device=pred.device)

    total = sim_coef * sim + var_coef * (var_low + var_up) + cov_coef * cov_pred + norm_coef * norm_pen
    return total, sim.detach(), var_low.detach(), var_up.detach(), cov_pred.detach(), norm_pen.detach(), t_low.detach(), t_up.detach(), t_cov.detach()


# -----------------------------
# AC Transformer (refactored + scaled defaults)
# -----------------------------

class ACTinyTransformer(nn.Module):
    """
    Block-causal transformer over time with per-time-step tokens:
      bundle=2: [Z_t, A_t]
      bundle=3: [Z_t, A_t, S_t] (optional)

    Predict next latent from Z-slot outputs.
    Supports delta prediction: z_{t+1} = z_t + Δ
    """
    def __init__(
        self,
        z_dim: int,
        s_dim: int,
        a_dim: int,
        hidden: int = 256,
        layers: int = 2,
        nhead: int = 4,
        use_s_token: bool = False,
        delta_pred: bool = True,
        dropout: float = 0.0,
        max_T: int = 1024,
    ):
        super().__init__()
        assert hidden % nhead == 0, "hidden must be divisible by nhead"
        self.use_s_token = use_s_token
        self.delta_pred = delta_pred
        self.bundle = 3 if use_s_token else 2
        self.max_T = max_T

        self.z_in = nn.Linear(z_dim, hidden)
        self.a_in = nn.Linear(a_dim, hidden)
        self.s_in = nn.Linear(s_dim, hidden) if use_s_token else None

        # LN + action gain helps prevent “ignore actions”
        self.z_ln = nn.LayerNorm(hidden)
        self.a_ln = nn.LayerNorm(hidden)
        self.s_ln = nn.LayerNorm(hidden) if use_s_token else None
        self.a_gain = nn.Parameter(torch.tensor(1.0))

        self.type_z = nn.Parameter(torch.zeros(1, 1, hidden))
        self.type_a = nn.Parameter(torch.zeros(1, 1, hidden))
        self.type_s = nn.Parameter(torch.zeros(1, 1, hidden)) if use_s_token else None
        nn.init.trunc_normal_(self.type_z, std=0.02)
        nn.init.trunc_normal_(self.type_a, std=0.02)
        if self.type_s is not None:
            nn.init.trunc_normal_(self.type_s, std=0.02)

        self.time_pos = nn.Embedding(max_T, hidden)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=nhead,
            dim_feedforward=4 * hidden,
            batch_first=True,
            activation="gelu",
            norm_first=True,
            dropout=dropout,
        )
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=layers)

        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 2 * hidden),
            nn.GELU(),
            nn.Linear(2 * hidden, z_dim),
        )

    def _block_causal_mask(self, T: int, device: torch.device):
        # time-level future mask [T,T]
        time_mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        tok_mask = time_mask.repeat_interleave(self.bundle, dim=0).repeat_interleave(self.bundle, dim=1)
        return tok_mask.float() * -1e9  # [S,S]

    def _build_tokens(self, z: torch.Tensor, a: torch.Tensor, s: Optional[torch.Tensor]):
        """
        z: [B,T,Dz], a: [B,T-1,Da], s: [B,T,Ds] if use_s_token
        """
        B, T, _ = z.shape
        assert T <= self.max_T, f"T={T} exceeds max_T={self.max_T}"
        W = T - 1

        z_proj = self.z_ln(self.z_in(z))                       # [B,T,H]
        a_pad = torch.zeros(B, 1, a.shape[-1], device=a.device, dtype=a.dtype)
        a_full = torch.cat([a, a_pad], dim=1)                  # [B,T,Da]
        a_proj = self.a_ln(self.a_in(a_full) * self.a_gain)    # [B,T,H]

        time = torch.arange(T, device=z.device)
        time_emb = self.time_pos(time).view(1, T, -1)          # [1,T,H]

        z_tok = z_proj + self.type_z + time_emb
        a_tok = a_proj + self.type_a + time_emb

        if self.use_s_token:
            assert s is not None
            s_proj = self.s_ln(self.s_in(s))
            s_tok = s_proj + self.type_s + time_emb
            toks = torch.stack([z_tok, a_tok, s_tok], dim=2)   # [B,T,3,H]
        else:
            toks = torch.stack([z_tok, a_tok], dim=2)          # [B,T,2,H]

        seq = toks.view(B, T * self.bundle, -1)                # [B,S,H]
        z_slots = (torch.arange(W, device=z.device) * self.bundle).long()  # [W]
        return seq, z_slots

    def forward_teacher(self, z: torch.Tensor, a: torch.Tensor, s: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Predict z_{t+1} for t=0..T-2. Returns [B,T-1,Dz]
        """
        B, T, Dz = z.shape
        W = T - 1
        seq, z_slots = self._build_tokens(z, a, s)
        attn_mask = self._block_causal_mask(T, seq.device)

        out = self.tr(seq, mask=attn_mask)  # [B,S,H]
        z_repr = out.gather(dim=1, index=z_slots.view(1, -1, 1).expand(B, -1, out.size(-1)))  # [B,W,H]
        dz = self.head(z_repr)  # [B,W,Dz]
        if self.delta_pred:
            return z[:, :W, :] + dz
        return dz

    @torch.no_grad()
    def forward_rollout(self, z: torch.Tensor, a: torch.Tensor, s: Optional[torch.Tensor], horizon: int = 4) -> torch.Tensor:
        """
        Autoregressive rollout from t=0 to t=horizon. Returns z_hat_horizon: [B,Dz]
        """
        B, T, Dz = z.shape
        assert horizon < T, "horizon must be < T"
        z_roll = z.clone()
        for t in range(horizon):
            z_pred = self.forward_teacher(
                z_roll[:, :t+2, :],
                a[:, :t+1, :],
                (s[:, :t+2, :] if (self.use_s_token and s is not None) else None),
            )
            z_roll[:, t+1, :] = z_pred[:, -1, :]
        return z_roll[:, horizon, :]
