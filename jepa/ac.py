#!/usr/bin/env python3
# ac.py
import os
import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb

from utils import (
    set_seed, worker_init_fn, enable_torch_perf,
    load_d4rl, Stats, load_stats,
    ACWindowDataset,
    Encoder, ACTinyTransformer,
    cosine_warmup_lr,
    compute_latent_stats,
)

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--env_id", type=str, default="halfcheetah-medium-v2")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # paths
    ap.add_argument("--encoder_ckpt", type=str, required=True)         # encoder_ema.pt
    ap.add_argument("--state_stats", type=str, default=None)           # state_stats.npz (optional)
    ap.add_argument("--ckpt_dir", type=str, required=True)

    # must match pretrain encoder
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--enc_hidden", type=int, default=512)
    ap.add_argument("--enc_layers", type=int, default=3)

    # AC transformer (SANE defaults)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--use_s_token", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--delta_pred", action=argparse.BooleanOptionalAction, default=True)

    # training
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--warmup_steps", type=int, default=5000)

    ap.add_argument("--rollout_horizon", type=int, default=4)
    ap.add_argument("--rollout_weight", type=float, default=1.0)

    # whitening defaults ON (recommended)
    ap.add_argument("--latent_whiten", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--action_whiten", action=argparse.BooleanOptionalAction, default=True)

    # action-usage hinge (permute actions)
    ap.add_argument("--neg_weight", type=float, default=1.0)
    ap.add_argument("--neg_margin", type=float, default=0.10)

    # logging
    ap.add_argument("--log_interval", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)

    # wandb
    ap.add_argument("--wandb_project", type=str, default="s-jepa-ac-d4rl")
    ap.add_argument("--wandb_run", type=str, default=None)
    ap.add_argument("--wandb_group", type=str, default="d4rl")
    ap.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])

    args = ap.parse_args()
    set_seed(args.seed)
    enable_torch_perf()

    device = torch.device(args.device)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    print(f"[Load] {args.env_id}")
    _, obs, actions, _, ep_bounds = load_d4rl(args.env_id, need_actions=True)
    state_dim = obs.shape[1]
    act_dim = actions.shape[1]

    # state stats: prefer loading from pretrain
    if args.state_stats is not None and os.path.isfile(args.state_stats):
        s_stats = load_stats(args.state_stats)
    else:
        s_stats = Stats.from_array(obs)

    # action stats (optional whitening)
    a_stats = Stats.from_array(actions) if args.action_whiten else None

    ds = ACWindowDataset(obs, actions, ep_bounds, window=args.window)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        worker_init_fn=worker_init_fn,
    )

    # frozen encoder
    encoder = Encoder(state_dim, embed_dim=args.embed_dim, hidden=args.enc_hidden, layers=args.enc_layers).to(device)
    sd = torch.load(args.encoder_ckpt, map_location=device)
    if isinstance(sd, dict) and "encoder_ema" in sd:
        encoder.load_state_dict(sd["encoder_ema"])
    else:
        encoder.load_state_dict(sd)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    print(f"[Encoder] Loaded: {args.encoder_ckpt}")

    # latent whitening (recommended)
    if args.latent_whiten:
        print("[Latent] computing z mean/std for whitening...")
        z_mu, z_std = compute_latent_stats(encoder, obs, s_stats, device)
    else:
        z_mu = torch.zeros(args.embed_dim, device=device)
        z_std = torch.ones(args.embed_dim, device=device)

    predictor = ACTinyTransformer(
        z_dim=args.embed_dim,
        s_dim=state_dim,
        a_dim=act_dim,
        hidden=args.hidden,
        layers=args.layers,
        nhead=args.nhead,
        use_s_token=args.use_s_token,
        delta_pred=args.delta_pred,
        dropout=args.dropout,
        max_T=max(1024, args.window + 2),
    ).to(device)

    opt = torch.optim.AdamW(predictor.parameters(), lr=args.lr, weight_decay=1e-4)

    wandb.init(
        project=args.wandb_project,
        name=(args.wandb_run or f"ac-{args.env_id}-seed{args.seed}"),
        group=args.wandb_group,
        mode=args.wandb_mode,
        config=vars(args) | {"state_dim": state_dim, "act_dim": act_dim, "episodes": len(ep_bounds), "dataset_obs": int(obs.shape[0])},
    )

    global_step = 0
    t0 = time.time()
    running = []

    m_s, s_s = s_stats.to_torch(device)

    while global_step < args.steps:
        for s, a in dl:
            s = s.to(device)   # [B,W+1,Ds]
            a = a.to(device)   # [B,W,Da]

            # normalize
            s = (s - m_s) / s_s
            if a_stats is not None:
                m_a, s_a = a_stats.to_torch(device)
                a = (a - m_a) / s_a

            with torch.no_grad():
                B, Wp1, Ds = s.shape
                z = encoder(s.reshape(B * Wp1, Ds))           # [B*(W+1),Dz]
                z = (z - z_mu) / z_std
                z = z.view(B, Wp1, -1)                        # [B,W+1,Dz]

            # teacher forcing
            z_pred = predictor.forward_teacher(z, a, s if args.use_s_token else None)  # [B,W,Dz]
            tf_l1 = F.l1_loss(z_pred, z[:, 1:, :])

            # rollout
            z_hat_H = predictor.forward_rollout(z, a, s if args.use_s_token else None, horizon=args.rollout_horizon)
            ro_l1 = F.l1_loss(z_hat_H, z[:, args.rollout_horizon, :])

            # negative-action hinge (permute actions)
            perm = torch.randperm(B, device=device)
            a_neg = a[perm]
            z_pred_neg = predictor.forward_teacher(z, a_neg, s if args.use_s_token else None)
            neg_err = (z_pred_neg - z[:, 1:, :]).abs().mean()
            neg_hinge = F.relu(args.neg_margin - neg_err)

            loss = tf_l1 + args.rollout_weight * ro_l1 + args.neg_weight * neg_hinge

            lr_now = cosine_warmup_lr(global_step, args.steps, args.lr, args.min_lr, args.warmup_steps)
            for g in opt.param_groups:
                g["lr"] = lr_now

            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0))
            opt.step()

            running.append(float(loss.item()))

            if (global_step + 1) % args.log_interval == 0:
                avg = float(np.mean(running)); running.clear()
                elapsed = time.time() - t0; t0 = time.time()
                fps = (args.log_interval * args.batch_size) / max(1e-6, elapsed)
                print(
                    f"step {global_step+1:7d} | loss {avg:.4f} | tf {tf_l1.item():.4f} | ro {ro_l1.item():.4f} | neg {neg_hinge.item():.4f} | fps {fps:.1f} | lr {lr_now:.2e}"
                )
                wandb.log({
                    "step": int(global_step + 1),
                    "loss": avg,
                    "tf_l1": float(tf_l1.item()),
                    "ro_l1": float(ro_l1.item()),
                    "neg_hinge": float(neg_hinge.item()),
                    "lr": float(lr_now),
                    "fps": float(fps),
                    "grad_norm": float(grad_norm),
                })

            if (global_step + 1) % 10_000 == 0:
                ckpt_path = os.path.join(args.ckpt_dir, f"ac_predictor_{global_step+1}.pt")
                torch.save({"predictor": predictor.state_dict(), "args": vars(args)}, ckpt_path)
                wandb.save(ckpt_path)

            global_step += 1
            if global_step >= args.steps:
                break

    final_path = os.path.join(args.ckpt_dir, "ac_predictor_final.pt")
    torch.save({"predictor": predictor.state_dict(), "args": vars(args)}, final_path)
    print(f"[Done] Saved predictor to {final_path}")
    wandb.save(final_path)
    wandb.finish()

if __name__ == "__main__":
    main()
