#!/usr/bin/env python3
# ssl.py
import os
import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

from utils import (
    set_seed, worker_init_fn, enable_torch_perf,
    load_d4rl, Stats, save_stats,
    StateJEPADataset,
    JEPAStateModel, jepa_loss,
    cosine_warmup_lr,
)

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--env_id", type=str, default="halfcheetah-medium-v2")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--k_max", type=int, default=5)
    ap.add_argument("--num_mask", type=int, default=3)

    # model sizes
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--enc_hidden", type=int, default=512)
    ap.add_argument("--enc_layers", type=int, default=3)

    ap.add_argument("--use_mask_token", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--tr_dropout", type=float, default=0.0)
    ap.add_argument("--pred_nhead", type=int, default=4)
    ap.add_argument("--pred_layers", type=int, default=2)
    ap.add_argument("--pred_ff_mult", type=int, default=4)

    # masking / augmentation
    ap.add_argument("--feature_mask_ratio", type=float, default=0.3)
    ap.add_argument("--time_mask_ratio", type=float, default=0.1)
    ap.add_argument("--dual_view_noise_std", type=float, default=0.0)

    # opt
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--warmup_steps", type=int, default=5000)
    ap.add_argument("--ema_base", type=float, default=0.99)
    ap.add_argument("--ema_final", type=float, default=0.9999)

    # loss weights
    ap.add_argument("--sim_coef", type=float, default=1.0)
    ap.add_argument("--var_coef", type=float, default=1.0)
    ap.add_argument("--cov_coef", type=float, default=0.1)
    ap.add_argument("--norm_coef", type=float, default=0.05)
    ap.add_argument("--var_upper", type=float, default=1.0)

    # logging / io
    ap.add_argument("--log_interval", type=int, default=200)
    ap.add_argument("--ckpt_dir", type=str, default="results/d4rl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=4)

    # wandb
    ap.add_argument("--wandb_project", type=str, default="s-jepa-d4rl")
    ap.add_argument("--wandb_run", type=str, default=None)
    ap.add_argument("--wandb_group", type=str, default="all")
    ap.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    ap.add_argument("--wandb_tags", type=str, nargs="*", default=None)
    ap.add_argument("--wandb_watch", action=argparse.BooleanOptionalAction, default=False)

    args = ap.parse_args()
    set_seed(args.seed)
    enable_torch_perf()

    ckpt_dir = os.path.join(args.ckpt_dir, args.env_id, f"seed{args.seed}")
    os.makedirs(ckpt_dir, exist_ok=True)

    device = torch.device(args.device)

    print(f"[Load] {args.env_id}")
    _, obs, _, ep_bounds = load_d4rl(args.env_id, need_actions=False)
    state_dim = obs.shape[1]
    print(f"[Data] obs={obs.shape} episodes={len(ep_bounds)}")

    # normalize states ONCE and save stats for AC stage
    s_stats = Stats.from_array(obs)
    save_stats(os.path.join(ckpt_dir, "state_stats.npz"), s_stats)
    obs_norm = s_stats.normalize_np(obs)

    ds = StateJEPADataset(
        obs_norm=obs_norm,
        episode_bounds=ep_bounds,
        window=args.window,
        k_max=args.k_max,
        num_mask=args.num_mask,
        feature_mask_ratio=args.feature_mask_ratio,
        time_mask_ratio=args.time_mask_ratio,
        dual_view_noise_std=args.dual_view_noise_std,
    )
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

    model = JEPAStateModel(
        state_dim=state_dim,
        embed_dim=args.embed_dim,
        enc_hidden=args.enc_hidden,
        enc_layers=args.enc_layers,
        ema_decay=args.ema_base,
        use_mask_token=args.use_mask_token,
        tr_dropout=args.tr_dropout,
        pred_nhead=args.pred_nhead,
        pred_layers=args.pred_layers,
        pred_ff_mult=args.pred_ff_mult,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    wandb.init(
        project=args.wandb_project,
        name=(args.wandb_run or f"{args.env_id}-seed{args.seed}"),
        group=args.wandb_group,
        tags=args.wandb_tags,
        mode=args.wandb_mode,
        config=vars(args) | {"state_dim": state_dim, "episodes": len(ep_bounds), "dataset_obs": int(obs.shape[0])},
    )
    if args.wandb_watch:
        wandb.watch(model, log="gradients", log_freq=args.log_interval)

    global_step = 0
    t0 = time.time()
    running = []

    while global_step < args.steps:
        for ctx1, ctx2, targets, ks in dl:
            ctx1 = ctx1.to(device)
            ctx2 = ctx2.to(device)
            targets = targets.to(device)
            ks = ks.to(device)

            pred1, pred2, targ = model(ctx1, ctx2, targets, ks)

            loss1, sim1, var1_low, var1_up, cov1p, norm1, *_ = jepa_loss(
                pred1, targ,
                sim_coef=args.sim_coef, var_coef=args.var_coef, cov_coef=args.cov_coef,
                norm_coef=args.norm_coef, var_upper=args.var_upper,
                reg_on_target=False,
            )
            loss2, sim2, var2_low, var2_up, cov2p, norm2, *_ = jepa_loss(
                pred2, targ,
                sim_coef=args.sim_coef, var_coef=args.var_coef, cov_coef=args.cov_coef,
                norm_coef=args.norm_coef, var_upper=args.var_upper,
                reg_on_target=False,
            )
            loss = 0.5 * (loss1 + loss2)

            lr_now = cosine_warmup_lr(global_step, args.steps, args.lr, args.min_lr, args.warmup_steps)
            for g in opt.param_groups:
                g["lr"] = lr_now

            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0))
            opt.step()

            # EMA cosine schedule (base -> final)
            with torch.no_grad():
                prog = (global_step + 1) / max(1, args.steps)
                m = args.ema_base + (args.ema_final - args.ema_base) * (1 - math.cos(math.pi * prog)) * 0.5
                model.update_ema(m)

            running.append(float(loss.item()))

            if (global_step + 1) % args.log_interval == 0:
                avg = float(np.mean(running)); running.clear()
                elapsed = time.time() - t0; t0 = time.time()
                fps = (args.log_interval * args.batch_size) / max(1e-6, elapsed)
                print(f"step {global_step+1:7d} | loss {avg:.4f} | fps {fps:.1f} | lr {lr_now:.2e}")

                wandb.log({
                    "step": int(global_step + 1),
                    "loss": avg,
                    "fps": float(fps),
                    "lr": float(lr_now),
                    "grad_norm": float(grad_norm),
                    "sim1": float(sim1), "sim2": float(sim2),
                    "var1_low": float(var1_low), "var1_up": float(var1_up),
                    "var2_low": float(var2_low), "var2_up": float(var2_up),
                    "cov1p": float(cov1p), "cov2p": float(cov2p),
                    "norm1": float(norm1), "norm2": float(norm2),
                })

            if (global_step + 1) % 10_000 == 0:
                ckpt_path = os.path.join(ckpt_dir, f"ckpt_{global_step+1}.pt")
                torch.save({
                    "encoder": model.encoder.state_dict(),
                    "encoder_ema": model.encoder_ema.state_dict(),
                    "predictor": model.predictor.state_dict(),
                    "args": vars(args),
                }, ckpt_path)
                wandb.save(ckpt_path)

            global_step += 1
            if global_step >= args.steps:
                break

    final_path = os.path.join(ckpt_dir, "encoder_ema.pt")
    torch.save(model.encoder_ema.state_dict(), final_path)
    print(f"[Done] Saved EMA encoder to {final_path}")
    wandb.save(final_path)
    wandb.finish()

if __name__ == "__main__":
    main()
