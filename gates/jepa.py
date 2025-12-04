import os
import math
import time
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import argparse
import wandb
# d4rl datasets
import gym
import d4rl

# -----------------------------
# D4RL dataset loading helpers
# -----------------------------

def load_d4rl_dataset(env_id: str):

    """try:
        import gym
    except Exception:
        import gymnasium as gym  # type: ignore
    import d4rl  # noqa: F401"""

    env = gym.make(env_id)
    ds = env.get_dataset() if hasattr(env, "get_dataset") else __import__('d4rl').qlearning_dataset(env)

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

    N = len(obs)
    episode_bounds: List[Tuple[int, int]] = []
    start = 0
    for i in range(N - 1):
        if dones[i]:
            episode_bounds.append((start, i))
            start = i + 1
    if start < N:
        episode_bounds.append((start, N - 1))
    return obs, dones, episode_bounds

# -----------------------------
# Dataset builder with dual-view masking and multi-step targets
# -----------------------------

class StateJEPADataset(Dataset):
    def __init__(self,
                 obs: np.ndarray,
                 episode_bounds: List[Tuple[int, int]],
                 window: int = 16,
                 k_max: int = 5,
                 num_mask: int = 3,
                 feature_mask_ratio: float = 0.3,
                 time_mask_ratio: float = 0.1,
                 dual_view_noise_std: float = 0.0,
                 normalize: bool = True):
        self.obs = obs.copy().astype(np.float32)
        self.window = window
        self.k_max = k_max
        self.num_mask = num_mask
        self.feature_mask_ratio = feature_mask_ratio
        self.time_mask_ratio = time_mask_ratio
        self.dual_view_noise_std = dual_view_noise_std
        self.starts: List[int] = []

        if normalize:
            self.mean = self.obs.mean(axis=0, keepdims=True)
            self.std = self.obs.std(axis=0, keepdims=True) + 1e-6
            self.obs = (self.obs - self.mean) / self.std

        for (s, e) in episode_bounds:
            T = e - s + 1
            max_start = T - (window + self.k_max)
            if max_start < 0:
                continue
            for offset in range(max_start + 1):
                self.starts.append(s + offset)

    def __len__(self):
        return len(self.starts)

    def _mask_ctx(self, ctx: np.ndarray) -> np.ndarray:
        ctx_masked = ctx.copy()
        w, D = ctx_masked.shape
        if self.feature_mask_ratio > 0:
            m = np.ones(D, dtype=np.float32)
            drop = np.random.choice(D, size=max(1, int(D * self.feature_mask_ratio)), replace=False)
            m[drop] = 0.0
            ctx_masked *= m[None, :]
        if self.time_mask_ratio > 0:
            t_mask = np.ones(w, dtype=np.float32)
            num_t_drop = max(1, int(w * self.time_mask_ratio))
            drop_t = np.random.choice(w, size=num_t_drop, replace=False)
            t_mask[drop_t] = 0.0
            ctx_masked *= t_mask[:, None]
        return ctx_masked

    def __getitem__(self, idx):
        start = self.starts[idx]
        w = self.window
        ctx = self.obs[start:start + w]                  # [W, D]
        ks = np.random.choice(np.arange(1, self.k_max + 1), size=self.num_mask, replace=False)
        ks.sort()
        targets = np.stack([self.obs[start + w - 1 + int(k)] for k in ks], axis=0)  # [M, D]

        ctx1 = self._mask_ctx(ctx)
        ctx2 = self._mask_ctx(ctx)
        if self.dual_view_noise_std > 0:
            ctx1 += np.random.randn(*ctx1.shape).astype(np.float32) * self.dual_view_noise_std
            ctx2 += np.random.randn(*ctx2.shape).astype(np.float32) * self.dual_view_noise_std

        return (
            torch.from_numpy(ctx1),             # [W, D]
            torch.from_numpy(ctx2),             # [W, D]
            torch.from_numpy(targets),          # [M, D]
            torch.from_numpy(ks.astype(np.int64)),  # [M]
        )

# -----------------------------
# Model components
# -----------------------------

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256, layers=2, act=nn.GELU):
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
    def __init__(self, state_dim: int, embed_dim: int = 256):
        super().__init__()
        self.proj = MLP(state_dim, embed_dim, hidden=512, layers=3)
    def forward(self, s):
        return self.proj(s)

class MaskedTokenPredictor(nn.Module):
    """Transformer predictor with multiple MASK tokens for multiple future steps.
    Input: h_ctx [B,W,d], ks [B,M] → Output: preds [B,M,d]."""
    def __init__(self, d: int, nhead: int = 4, layers: int = 2, ff_mult: int = 4, max_pos: int = 4096, dropout: float = 0.0):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, 
            nhead=nhead, 
            dim_feedforward=ff_mult*d,
            batch_first=True, 
            activation='gelu', 
            norm_first=True, 
            dropout=dropout
            )

        self.tr = nn.TransformerEncoder(enc_layer, num_layers=layers)

        self.pos = nn.Embedding(max_pos, d)
        self.k_embed = nn.Embedding(max_pos, d)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d))

        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.k_embed.weight, std=0.02)
        
        self.head = nn.Sequential(
            nn.LayerNorm(d*2), 
            nn.Linear(d*2, d*2), 
            nn.GELU(), 
            nn.Linear(d*2, d)
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
        mask_out = out[:, W:, :]                   # [B, M, d]
        k_emb = self.k_embed(ks)                   # [B, M, d]
        pred = self.head(torch.cat([mask_out, k_emb], dim=-1))
        return pred

class JEPAStateModel(nn.Module):
    def __init__(self, state_dim: int, embed_dim: int = 256, ema_decay: float = 0.99, use_mask_token: bool = True, tr_dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.encoder = Encoder(state_dim, embed_dim)
        self.encoder_ema = Encoder(state_dim, embed_dim)
        self.use_mask_token = use_mask_token

        if use_mask_token:
            self.predictor = MaskedTokenPredictor(embed_dim, nhead=4, layers=2, ff_mult=4, max_pos=4096, dropout=tr_dropout)
        
        else:
            self.predictor = MLP(embed_dim, embed_dim, hidden=512, layers=2)

        self._init_ema()
        self._ema_m = ema_decay

    @torch.no_grad()
    def _init_ema(self):
        for p, p_ema in zip(self.encoder.parameters(), self.encoder_ema.parameters()):
            p_ema.data.copy_(p.data); p_ema.requires_grad_(False)
            
    @torch.no_grad()
    def update_ema(self, momentum: float):
        self._ema_m = momentum
        for p, p_ema in zip(self.encoder.parameters(), self.encoder_ema.parameters()):
            p_ema.data.mul_(momentum).add_(p.data, alpha=1.0 - momentum)

    def forward(self, ctx1, ctx2, targets, ks):
        h1 = self.encoder(ctx1)  # [B,W,d]
        h2 = self.encoder(ctx2)

        if self.use_mask_token:
            pred1 = self.predictor(h1, ks)
            pred2 = self.predictor(h2, ks)

        else:
            c1 = h1.mean(dim=1); c2 = h2.mean(dim=1)
            pred1 = self.predictor(c1).unsqueeze(1).expand(-1, ks.shape[1], -1)
            pred2 = self.predictor(c2).unsqueeze(1).expand(-1, ks.shape[1], -1)

        with torch.no_grad():
            B, M, D = targets.shape
            targ = self.encoder_ema(targets.view(B*M, D)).view(B, M, -1)

        return pred1, pred2, targ

# -----------------------------
# Loss: JEPA/BYOL + VICReg-L components
# -----------------------------

def jepa_loss(pred: torch.Tensor, target: torch.Tensor,
              sim_coef=1.0, var_coef=1.0, cov_coef=0.1, norm_coef=0.05,
              eps=1e-4, var_upper=1.0, reg_on_target: bool = False, cov_reduce: str = 'mean'):
    """Supports [B,M,d] or [B,d]. Returns (total, sim, var_low, var_up, cov_pred, norm_pen, t_low, t_up, t_cov)."""
    if pred.dim() == 3:
        B, M, d = pred.shape
        pred = pred.reshape(B*M, d)
        target = target.detach().reshape(B*M, d)
    else:
        target = target.detach()

    # similarity on normalized embeddings
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
        return off_diag.pow(2).mean() if cov_reduce == 'mean' else (off_diag.pow(2).sum())/d

    var_low, var_up = variance_terms(pred)
    cov_pred = covariance_term(pred)

    # feature energy regularizer: E[||pred||^2/d] ≈ per-sample mean then average
    energy = pred.pow(2).mean(dim=1).mean()
    norm_pen = (energy - 1.0).pow(2)

    if reg_on_target:
        t_low, t_up = variance_terms(target)
        t_cov = covariance_term(target)
    else:
        t_low = torch.zeros((), device=pred.device); t_up = torch.zeros((), device=pred.device); t_cov = torch.zeros((), device=pred.device)

    total = sim_coef*sim + var_coef*(var_low + var_up) + cov_coef*cov_pred + norm_coef*norm_pen
    return total, sim.detach(), var_low.detach(), var_up.detach(), cov_pred.detach(), norm_pen.detach(), t_low.detach(), t_up.detach(), t_cov.detach()

# -----------------------------
# Training
# -----------------------------

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env_id', type=str, default='halfcheetah-medium-v2')
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--steps', type=int, default=200_000)
    ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--window', type=int, default=16)
    ap.add_argument('--k_max', type=int, default=5)
    ap.add_argument('--num_mask', type=int, default=3)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--min_lr', type=float, default=1e-6)
    ap.add_argument('--warmup_steps', type=int, default=5000)
    ap.add_argument('--ema_base', type=float, default=0.99)
    ap.add_argument('--ema_final', type=float, default=0.9999)
    ap.add_argument('--feature_mask_ratio', type=float, default=0.3)
    ap.add_argument('--time_mask_ratio', type=float, default=0.1)
    ap.add_argument('--dual_view_noise_std', type=float, default=0.0)
    ap.add_argument('--tr_dropout', type=float, default=0.0)
    ap.add_argument('--use_mask_token', action='store_true', default=True)
    ap.add_argument('--log_interval', type=int, default=200)
    ap.add_argument('--ckpt_dir', type=str, default='./amltckpt/jepa_sweep')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--wandb_project', type=str, default='jepa-sweep')
    ap.add_argument('--wandb_run', type=str, default=None)
    ap.add_argument('--wandb_group', type=str, default=None)
    ap.add_argument('--wandb_mode', type=str, default='online', choices=['online','offline','disabled'])
    ap.add_argument('--wandb_tags', type=str, nargs='*', default=None)
    ap.add_argument('--sim_coef', type=float, default=1.0)
    ap.add_argument('--var_coef', type=float, default=1.0)
    ap.add_argument('--cov_coef', type=float, default=0.1)
    ap.add_argument('--norm_coef', type=float, default=0.05)
    ap.add_argument('--var_upper', type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    set_seed(args.seed)

    print(f"[Load] D4RL dataset: {args.env_id}")
    obs, _, ep_bounds = load_d4rl_dataset(args.env_id)
    state_dim = obs.shape[1]
    print(f"[Data] obs: {obs.shape}, episodes: {len(ep_bounds)}")

    ds = StateJEPADataset(
        obs, ep_bounds,
        window=args.window,
        k_max=args.k_max,
        num_mask=args.num_mask,
        feature_mask_ratio=args.feature_mask_ratio,
        time_mask_ratio=args.time_mask_ratio,
        dual_view_noise_std=args.dual_view_noise_std,
        normalize=True,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    device = torch.device(args.device)
    model = JEPAStateModel(state_dim=state_dim, embed_dim=256, ema_decay=args.ema_base,
                           use_mask_token=args.use_mask_token, tr_dropout=args.tr_dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # --- Weights & Biases ---
    wandb.init(project=args.wandb_project,
               name=args.wandb_run,
               group=args.wandb_group,
               tags=args.wandb_tags,
               mode=args.wandb_mode,
               config=dict(env_id=args.env_id, steps=args.steps, batch_size=args.batch_size,
                           window=args.window, k_max=args.k_max, num_mask=args.num_mask,
                           lr=args.lr, min_lr=args.min_lr, warmup_steps=args.warmup_steps,
                           ema_base=args.ema_base, ema_final=args.ema_final,
                           feature_mask_ratio=args.feature_mask_ratio, time_mask_ratio=args.time_mask_ratio,
                           dual_view_noise_std=args.dual_view_noise_std,
                           tr_dropout=args.tr_dropout,
                           sim_coef=args.sim_coef, var_coef=args.var_coef, cov_coef=args.cov_coef,
                           norm_coef=args.norm_coef, var_upper=args.var_upper,
                           use_mask_token=args.use_mask_token,
                           seed=args.seed, state_dim=state_dim))
    wandb.watch(model, log='gradients', log_freq=args.log_interval)
    wandb.define_metric('step')
    for k in ['loss','fps','sim1','sim2','var1_low','var1_up','var2_low','var2_up','cov1p','cov2p','norm1','norm2','t1_low','t1_up','t1_cov','t2_low','t2_up','t2_cov','grad_norm','lr']:
        wandb.define_metric(k, step_metric='step')
    wandb.summary['episodes'] = len(ep_bounds)
    wandb.summary['dataset_obs'] = int(obs.shape[0])

    global_step = 0
    base_lr = args.lr
    min_lr = args.min_lr
    warmup = max(1, args.warmup_steps)

    def compute_lr(step: int) -> float:
        if step < warmup:
            return base_lr * float(step + 1) / float(warmup)
        else:
            progress = float(step - warmup) / float(max(1, args.steps - warmup))
            return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

    ema_update_every = 1
    t0 = time.time()
    running = []

    while global_step < args.steps:
        for ctx1, ctx2, targets, ks in dl:
            ctx1 = ctx1.to(device)
            ctx2 = ctx2.to(device)
            targets = targets.to(device)         # [B,M,D]
            ks = ks.to(device)                   # [B,M]

            pred1, pred2, targ = model(ctx1, ctx2, targets, ks)
            loss1, sim1, var1_low, var1_up, cov1p, norm1, t1_low, t1_up, t1_cov = jepa_loss(
                pred1, targ,
                sim_coef=args.sim_coef, var_coef=args.var_coef, cov_coef=args.cov_coef,
                norm_coef=args.norm_coef, var_upper=args.var_upper,
                reg_on_target=True, cov_reduce='mean')
            loss2, sim2, var2_low, var2_up, cov2p, norm2, t2_low, t2_up, t2_cov = jepa_loss(
                pred2, targ,
                sim_coef=args.sim_coef, var_coef=args.var_coef, cov_coef=args.cov_coef,
                norm_coef=args.norm_coef, var_upper=args.var_upper,
                reg_on_target=True, cov_reduce='mean')
            loss = 0.5*(loss1 + loss2)

            # lr schedule
            lr_now = compute_lr(global_step)
            for g in opt.param_groups:
                g['lr'] = lr_now

            opt.zero_grad(set_to_none=True)
            loss.backward()
            total_gn = float(nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0))
            opt.step()

            if (global_step % ema_update_every) == 0:
                with torch.no_grad():
                    # Cosine EMA momentum from base->final
                    progress = (global_step + 1) / max(1, args.steps)
                    m = args.ema_base + (args.ema_final - args.ema_base) * (1 - math.cos(math.pi * progress)) * 0.5
                    model.update_ema(m)

            running.append(float(loss.item()))
            if (global_step + 1) % args.log_interval == 0:
                avg = float(np.mean(running)); running.clear()
                elapsed = time.time() - t0; t0 = time.time()
                fps = (args.log_interval*args.batch_size)/max(1e-6, elapsed)
                print(f"step {global_step+1:7d} | loss {avg:.4f} | fps ~{fps:.1f} | lr {lr_now:.2e}")
                if wandb.run is not None:
                    wandb.log({'step': int(global_step+1), 'loss': avg, 'fps': float(fps),
                               'sim1': float(sim1), 'sim2': float(sim2),
                               'var1_low': float(var1_low), 'var1_up': float(var1_up),
                               'var2_low': float(var2_low), 'var2_up': float(var2_up),
                               'cov1p': float(cov1p), 'cov2p': float(cov2p),
                               'norm1': float(norm1), 'norm2': float(norm2),
                               't1_low': float(t1_low), 't1_up': float(t1_up), 't1_cov': float(t1_cov),
                               't2_low': float(t2_low), 't2_up': float(t2_up), 't2_cov': float(t2_cov),
                               'grad_norm': float(total_gn), 'lr': float(lr_now)})

            if (global_step + 1) % (10_000) == 0:
                ckpt_path = os.path.join(args.ckpt_dir, f"ckpt_{global_step+1}.pt")
                os.makedirs(args.ckpt_dir, exist_ok=True)
                torch.save({'encoder': model.encoder.state_dict(),
                            'encoder_ema': model.encoder_ema.state_dict(),
                            'predictor': model.predictor.state_dict()}, ckpt_path)
                if wandb.run is not None:
                    art = wandb.Artifact(f"state-jepa-ckpt-{global_step+1}", type="model")
                    art.add_file(ckpt_path); wandb.log_artifact(art)

            global_step += 1
            if global_step >= args.steps:
                break

    final_path = os.path.join(args.ckpt_dir, "encoder_ema.pt")
    torch.save(model.encoder_ema.state_dict(), final_path)
    print(f"[Done] Saved EMA encoder to {final_path}")
    if wandb.run is not None:
        art = wandb.Artifact(f"state-jepa-encoder-ema-{args.env_id.replace('-', '_')}", type="model")
        art.add_file(final_path); wandb.log_artifact(art); wandb.finish()

if __name__ == "__main__":
    main()
