#!/usr/bin/env python3
"""
S-JEPA → Action-Conditioned Predictor (Mujoco general)
-----------------------------------------------------
- Loads a pretrained State-JEPA encoder (EMA) per D4RL Mujoco env.
- Trains an action-conditioned predictor P_phi to forecast future latent z.
- Works for {halfcheetah, hopper, walker2d}-{medium, medium-replay, medium-expert}-v2.
- With W&B logging and periodic checkpoints.

Default encoder path pattern (if --encoder_ckpt is not set):
  {--encoder_root}/{env_id}/seed{--seed}/encoder_ema.pt
  e.g., checkpoints/mujoco/halfcheetah-medium-replay-v2/seed42/encoder_ema.pt

Default AC predictor save dir (if --ckpt_dir is not set):
  {--ckpt_root}/{env_id}/seed{--seed}
  e.g., checkpoints/mujoco_ac/halfcheetah-medium-replay-v2/seed42

Usage example:
  python s-jepa-pipelines/s_jepa_ac_mujoco.py \
    --env_id halfcheetah-medium-replay-v2 \
    --seed 42 \
    --latent_whiten --action_whiten --use_s_token \
    --steps 200000 --batch_size 256 --window 32 \
    --hidden 512 --layers 8 --nhead 8 \
    --rollout_horizon 8 --rollout_weight 1.0 \
    --wandb_project s-jepa-ac-mujoco --wandb_run ac-halfcheetah-med-replay-s42
"""

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

import gym
import d4rl  

# -----------------------------
# D4RL dataset loading helpers
# -----------------------------

def load_d4rl_dataset(env_id: str):

    env = gym.make(env_id)
    ds = env.get_dataset() if hasattr(env, "get_dataset") else __import__('d4rl').qlearning_dataset(env)

    obs = ds["observations"].astype(np.float32)
    actions = ds["actions"].astype(np.float32)
    terminals = ds.get("terminals")
    if terminals is None:
        terminals = ds.get("dones")
    if terminals is None:
        terminals = np.zeros((len(obs),), dtype=np.bool_)
    timeouts = ds.get("timeouts")
    if timeouts is None:
        timeouts = np.zeros_like(terminals, dtype=np.bool_)

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
    return env, obs, actions, dones, episode_bounds

# -----------------------------
# Normalization utils
# -----------------------------

class Stats:
    def __init__(self, x: np.ndarray):
        self.mean = torch.from_numpy(x.mean(axis=0)).float()
        self.std = torch.from_numpy(x.std(axis=0)).float()
        self.std[self.std < 1e-6] = 1e-6
    def to(self, device):
        self.mean = self.mean.to(device); self.std = self.std.to(device); return self
    def norm(self, x: torch.Tensor):
        return (x - self.mean) / self.std
    def denorm(self, x: torch.Tensor):
        return x * self.std + self.mean

@torch.no_grad()
def compute_latent_stats(encoder: nn.Module, obs: np.ndarray, s_stats: Stats, device: torch.device, batch: int = 4096):
    N = obs.shape[0]
    zs = []
    for i in range(0, N, batch):
        s = torch.from_numpy(obs[i:i+batch]).float().to(device)
        s = (s - s_stats.mean.to(device)) / s_stats.std.to(device)
        z = encoder(s)
        zs.append(z.cpu())
    z_all = torch.cat(zs, dim=0)
    mu = z_all.mean(dim=0)
    std = z_all.std(dim=0)
    std[std < 1e-6] = 1e-6
    return mu.to(device), std.to(device)

# -----------------------------
# Dataset for AC training: sliding windows within episodes
# -----------------------------

class ACWindowDataset(Dataset):
    def __init__(self, obs: np.ndarray, actions: np.ndarray, ep_bounds: List[Tuple[int,int]],
                 window: int = 16):
        self.obs = obs
        self.actions = actions
        self.window = window
        self.starts: List[int] = []
        for (s, e) in ep_bounds:
            T = e - s + 1
            # need obs length >= window+1 (because we predict next)
            max_start = T - (window + 1)
            if max_start < 0:
                continue
            for off in range(max_start + 1):
                self.starts.append(s + off)
    def __len__(self):
        return len(self.starts)
    def __getitem__(self, idx):
        t0 = self.starts[idx]
        # states: [W+1, Ds], actions: [W, Da]
        s = torch.from_numpy(self.obs[t0:t0+self.window+1]).float()
        a = torch.from_numpy(self.actions[t0:t0+self.window]).float()
        return s, a

# -----------------------------
# Encoder (frozen) – must match pretraining
# -----------------------------

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=512, layers=3, act=nn.GELU):
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

# -----------------------------
# Action-Conditioned Predictor
# -----------------------------

class ACTransformer(nn.Module):
    """Block-causal Transformer over time with per-timestep token bundle [Z_t, A_t, S_t]."""
    def __init__(self, z_dim: int, s_dim: int, a_dim: int, hidden: int = 512, layers: int = 8, nhead: int = 8,
                 use_s_token: bool = True, token_drop: float = 0.0):
        super().__init__()
        self.use_s_token = use_s_token
        self.bundle = 3 if use_s_token else 2  # [Z, A, (S)]
        # token projections to hidden
        self.z_in = nn.Linear(z_dim, hidden)
        self.a_in = nn.Linear(a_dim, hidden)
        if use_s_token:
            self.s_in = nn.Linear(s_dim, hidden)
        # token-type embeddings
        self.type_z = nn.Parameter(torch.zeros(1, 1, hidden))
        self.type_a = nn.Parameter(torch.zeros(1, 1, hidden))
        self.type_s = nn.Parameter(torch.zeros(1, 1, hidden)) if use_s_token else None
        nn.init.trunc_normal_(self.type_z, std=0.02)
        nn.init.trunc_normal_(self.type_a, std=0.02)
        if self.type_s is not None:
            nn.init.trunc_normal_(self.type_s, std=0.02)
        # time positional embedding (learned)
        self.time_pos = nn.Embedding(4096, hidden)
        # transformer
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=nhead, dim_feedforward=4*hidden,
                                               batch_first=True, activation='gelu', norm_first=True, dropout=token_drop)
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=layers)
        # prediction head: from Z_t slot representation -> predict z_{t+1}
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 2*hidden), nn.GELU(), nn.Linear(2*hidden, z_dim))

    def _build_seq(self, z_seq: torch.Tensor, a_seq: torch.Tensor, s_seq: torch.Tensor):
        """z_seq: [B,W+1,dz], a_seq: [B,W,Da], s_seq: [B,W+1,Ds] or None
           Returns tokens [B, (W+1)*bundle, hidden] and index map for each time step's Z-slot.
        """
        B, Wp1, dz = z_seq.shape
        W = Wp1 - 1
        device = z_seq.device
        tokens = []
        z_slots = []  # indices of Z_t positions in the flattened sequence (length W)
        for t in range(Wp1):
            time_emb = self.time_pos.weight[t].view(1, 1, -1)
            z_tok = self.z_in(z_seq[:, t, :]).unsqueeze(1) + self.type_z + time_emb
            if t < W:
                a_tok = self.a_in(a_seq[:, t, :]).unsqueeze(1) + self.type_a + time_emb
            else:
                a_tok = self.a_in(torch.zeros_like(a_seq[:, 0, :])).unsqueeze(1) + self.type_a + time_emb
            if self.use_s_token:
                s_tok = self.s_in(s_seq[:, t, :]).unsqueeze(1) + self.type_s + time_emb
                bundle = torch.cat([z_tok, a_tok, s_tok], dim=1)
                z_index = len(tokens)*self.bundle
            else:
                bundle = torch.cat([z_tok, a_tok], dim=1)
                z_index = len(tokens)*self.bundle
            tokens.append(bundle)
            z_slots.append(z_index)
        seq = torch.cat(tokens, dim=1)  # [B, (W+1)*bundle, H]
        return seq, torch.tensor(z_slots[:-1], device=device)  # only Z_0..Z_{W-1} produce preds

    def _causal_mask(self, T: int, device: torch.device):
        """Block-causal by time step (bundle tokens per time)."""
        S = T * self.bundle
        m = torch.ones(S, S, device=device, dtype=torch.bool)
        for ti in range(T):
            for tj in range(T):
                if tj > ti:
                    m[ti*self.bundle:(ti+1)*self.bundle, tj*self.bundle:(tj+1)*self.bundle] = False
        return m

    def forward_teacher(self, z_seq: torch.Tensor, a_seq: torch.Tensor, s_seq: torch.Tensor):
        B, Wp1, dz = z_seq.shape
        W = Wp1 - 1
        seq, z_slots = self._build_seq(z_seq, a_seq, s_seq)
        attn_mask_bool = self._causal_mask(Wp1, seq.device)
        attn_mask = (~attn_mask_bool).float() * -1e9
        out = self.tr(seq, mask=attn_mask)
        z_t_repr = out.gather(dim=1, index=z_slots.view(1, -1, 1).expand(B, -1, out.size(-1)))
        z_pred = self.head(z_t_repr)  # [B, W, dz]
        return z_pred

    @torch.no_grad()
    def forward_rollout(self, z_seq: torch.Tensor, a_seq: torch.Tensor, s_seq: torch.Tensor, horizon: int = 2):
        """Autoregressive rollout starting at t=0 for given horizon. Returns ẑ_{horizon}."""
        B, Wp1, dz = z_seq.shape
        z_roll = z_seq.clone()
        for step in range(horizon):
            z_pred = self.forward_teacher(z_roll[:, :step+1+1, :], a_seq[:, :step+1, :], s_seq[:, :step+1+1, :])
            z_hat_next = z_pred[:, -1, :]
            z_roll[:, step+1, :] = z_hat_next
        return z_roll[:, horizon, :]

# -----------------------------
# Training loop
# -----------------------------

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env_id', type=str, default='halfcheetah-medium-v2')
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    # encoder path options
    ap.add_argument('--encoder_ckpt', type=str, default=None, help='override path to encoder_ema.pt')
    ap.add_argument('--encoder_root', type=str, default='checkpoints/mujoco', help='root to locate encoder if --encoder_ckpt unset')

    # training steps
    ap.add_argument('--steps', type=int, default=200_000)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--window', type=int, default=16)

    ap.add_argument('--hidden', type=int, default=512)
    ap.add_argument('--layers', type=int, default=8)
    ap.add_argument('--nhead', type=int, default=8)
    ap.add_argument('--token_dropout', type=float, default=0.0)
    ap.add_argument('--use_s_token', action='store_true')

    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--min_lr', type=float, default=1e-6)
    ap.add_argument('--warmup_steps', type=int, default=5000)

    ap.add_argument('--rollout_horizon', type=int, default=2)
    ap.add_argument('--rollout_weight', type=float, default=1.0)

    ap.add_argument('--latent_whiten', action='store_true')
    ap.add_argument('--action_whiten', action='store_true')

    ap.add_argument('--log_interval', type=int, default=200)
    ap.add_argument('--ckpt_dir', type=str, default=None, help='override save dir for AC predictor')
    ap.add_argument('--ckpt_root', type=str, default='./checkpoints/mujoco_ac', help='root to save AC predictor if --ckpt_dir unset')
    ap.add_argument('--seed', type=int, default=42)

    # W&B
    ap.add_argument('--wandb_project', type=str, default='s-jepa-ac-mujoco')
    ap.add_argument('--wandb_run', type=str, default=None)
    ap.add_argument('--wandb_group', type=str, default='mujoco')
    ap.add_argument('--wandb_mode', type=str, default='online', choices=['online','offline','disabled'])

    args = ap.parse_args()
    set_seed(args.seed)

    # resolve paths (encoder + ac ckpt dir)
    if args.encoder_ckpt is None:
        args.encoder_ckpt = os.path.join(args.encoder_root, args.env_id, f'seed{args.seed}', 'encoder_ema.pt')
    if args.ckpt_dir is None:
        args.ckpt_dir = os.path.join(args.ckpt_root, args.env_id, f'seed{args.seed}')
    os.makedirs(args.ckpt_dir, exist_ok=True)

    print(f"[Load] D4RL dataset: {args.env_id}")
    env, obs, actions, dones, ep_bounds = load_d4rl_dataset(args.env_id)
    state_dim = obs.shape[1]
    act_dim = actions.shape[1]

    s_stats = Stats(obs)
    a_stats = Stats(actions) if args.action_whiten else None

    ds = ACWindowDataset(obs, actions, ep_bounds, window=args.window)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    device = torch.device(args.device)

    # --- frozen encoder ---
    encoder = Encoder(state_dim, embed_dim=256).to(device)
    sd = torch.load(args.encoder_ckpt, map_location=device)
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    encoder.load_state_dict(sd)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    print(f"[Encoder] Loaded: {args.encoder_ckpt}")

    # optional z whitening
    if args.latent_whiten:
        print('[Latent] Computing z mean/std for whitening...')
        z_mu, z_std = compute_latent_stats(encoder, obs, s_stats, device)
    else:
        z_mu = torch.zeros(256, device=device)
        z_std = torch.ones(256, device=device)

    # --- predictor ---
    predictor = ACTransformer(z_dim=256, s_dim=state_dim, a_dim=act_dim, hidden=args.hidden, layers=args.layers,
                              nhead=args.nhead, use_s_token=args.use_s_token, token_drop=args.token_dropout).to(device)

    opt = torch.optim.AdamW(predictor.parameters(), lr=args.lr, weight_decay=1e-4)

    # lr schedule
    base_lr = args.lr; min_lr = args.min_lr; warmup = max(1, args.warmup_steps)
    def compute_lr(step: int):
        if step < warmup:
            return base_lr * float(step + 1) / float(warmup)
        else:
            prog = float(step - warmup)/float(max(1, args.steps - warmup))
            return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * prog))

    # wandb
    wandb.init(project=args.wandb_project,
               name=(args.wandb_run or f"ac-{args.env_id}-s{args.seed}"),
               group=args.wandb_group,
               mode=args.wandb_mode,
               config=dict(env_id=args.env_id, steps=args.steps, batch_size=args.batch_size, window=args.window,
                           hidden=args.hidden, layers=args.layers, nhead=args.nhead,
                           use_s_token=args.use_s_token, rollout_horizon=args.rollout_horizon,
                           rollout_weight=args.rollout_weight, lr=args.lr, min_lr=args.min_lr, warmup_steps=args.warmup_steps,
                           latent_whiten=args.latent_whiten, action_whiten=args.action_whiten,
                           encoder_ckpt=args.encoder_ckpt, ckpt_dir=args.ckpt_dir, seed=args.seed))
                           
    wandb.define_metric('step')
    for k in ['loss','tf_l1','ro_l1','lr','fps']:
        wandb.define_metric(k, step_metric='step')

    global_step = 0
    t0 = time.time(); running = []

    while global_step < args.steps:
        for s, a in dl:
            s = s.to(device)   # [B, W+1, Ds]
            a = a.to(device)   # [B, W, Da]
            # normalize states/actions
            s = s_stats.to(device).norm(s)
            if args.action_whiten:
                a = a_stats.to(device).norm(a)

            with torch.no_grad():
                # encode z for all time steps (freeze encoder)
                B, Wp1, Ds = s.shape
                s_flat = s.reshape(B*Wp1, Ds)
                z_flat = encoder(s_flat)
                # optional z whiten
                z_flat = (z_flat - z_mu) / z_std
                z = z_flat.view(B, Wp1, -1)

            # teacher forcing: predict z_{t+1} for t=0..W-1
            z_pred = predictor.forward_teacher(z, a, s if args.use_s_token else None)  # [B, W, dz]
            tf_l1 = F.l1_loss(z_pred, z[:, 1:, :])

            # short rollout loss: start at t=0, horizon H
            z_hat_H = predictor.forward_rollout(z, a, s if args.use_s_token else None, horizon=args.rollout_horizon)
            ro_l1 = F.l1_loss(z_hat_H, z[:, args.rollout_horizon, :])

            loss = tf_l1 + args.rollout_weight * ro_l1

            # step
            lr_now = compute_lr(global_step)
            for g in opt.param_groups:
                g['lr'] = lr_now
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
            opt.step()

            running.append(float(loss.item()))
            if (global_step + 1) % args.log_interval == 0:
                avg = float(np.mean(running)); running.clear()
                elapsed = time.time() - t0; t0 = time.time()
                fps = (args.log_interval*args.batch_size)/max(1e-6, elapsed)
                print(f"step {global_step+1:7d} | loss {avg:.4f} | tf_l1 {tf_l1.item():.4f} | ro_l1 {ro_l1.item():.4f} | fps ~{fps:.1f} | lr {lr_now:.2e}")
                wandb.log({'step': int(global_step+1), 'loss': avg, 'tf_l1': float(tf_l1.item()), 'ro_l1': float(ro_l1.item()),
                           'lr': float(lr_now), 'fps': float(fps)})

            # save ckpt
            if (global_step + 1) % 10_000 == 0:
                ckpt_path = os.path.join(args.ckpt_dir, f"ac_predictor_{global_step+1}.pt")
                torch.save({'predictor': predictor.state_dict()}, ckpt_path)
                art = wandb.Artifact(f"sjepa-ac-{args.env_id}-s{args.seed}-{global_step+1}", type="model")
                art.add_file(ckpt_path)
                wandb.log_artifact(art)

            global_step += 1
            if global_step >= args.steps:
                break

    final_path = os.path.join(args.ckpt_dir, "ac_predictor_final.pt")
    torch.save({'predictor': predictor.state_dict()}, final_path)
    print(f"[Done] Saved predictor to {final_path}")
    art = wandb.Artifact(f"sjepa-ac-final-{args.env_id}-s{args.seed}", type="model")
    art.add_file(final_path); wandb.log_artifact(art)
    wandb.finish()

if __name__ == "__main__":
    main()
