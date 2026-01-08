import os
# os.environ['MUJOCO_GL'] = 'egl'

import d4rl
import gym
import hydra, wandb, uuid
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from utils import set_seed
from tqdm import tqdm
from omegaconf import OmegaConf


from typing import Optional
from dataclasses import dataclass
import random


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=512, layers=3, act=nn.GELU):
        super().__init__()
        dims = [in_dim] + [hidden]*(layers-1) + [out_dim]
        mods = []

        for i in range(len(dims)-2):
            mods += [nn.Linear(dims[i], dims[i+1]), act(), nn.LayerNorm(dims[i+1])]

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

class ACTransformer(nn.Module):
    def __init__(self, z_dim: int, s_dim: int, a_dim: int, hidden: int = 512, layers: int = 8, nhead: int = 8,
                 use_s_token: bool = True):
        super().__init__()
        self.use_s_token = use_s_token
        self.bundle = 3 if use_s_token else 2
        self.z_in = nn.Linear(z_dim, hidden)
        self.a_in = nn.Linear(a_dim, hidden)

        if use_s_token:
            self.s_in = nn.Linear(s_dim, hidden)

        self.type_z = nn.Parameter(torch.zeros(1, 1, hidden))
        self.type_a = nn.Parameter(torch.zeros(1, 1, hidden))
        self.type_s = nn.Parameter(torch.zeros(1, 1, hidden)) if use_s_token else None
        nn.init.trunc_normal_(self.type_z, std=0.02)
        nn.init.trunc_normal_(self.type_a, std=0.02)

        if self.type_s is not None:
            nn.init.trunc_normal_(self.type_s, std=0.02)

        self.time_pos = nn.Embedding(4096, hidden)
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=nhead, dim_feedforward=4*hidden,
                                               batch_first=True, activation='gelu', norm_first=True, dropout=0.0)
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 2*hidden), nn.GELU(), nn.Linear(2*hidden, z_dim))

    def _build_tokens(self, z0, a0, s0):
        # z0:[B, dz], a0:[B, Da], s0:[B, Ds] or None -> tokens [B, bundle*2, H]
        B = z0.size(0); H = self.z_in.out_features
        t0 = self.time_pos.weight[0].view(1,1,-1); t1 = self.time_pos.weight[1].view(1,1,-1)
        z0_tok = self.z_in(z0).unsqueeze(1) + self.type_z + t0
        a0_tok = self.a_in(a0).unsqueeze(1) + self.type_a + t0

        if self.use_s_token:
            s0_tok = self.s_in(s0).unsqueeze(1) + self.type_s + t0
            bundle0 = torch.cat([z0_tok, a0_tok, s0_tok], dim=1)

        else:
            bundle0 = torch.cat([z0_tok, a0_tok], dim=1)
        # 为了预测 z1, 构造一个占位的第二个时间步的 [Z1, A1, S1]，其中 A1/S1 仅用于对齐时间位置编码
        z1_tok_dummy = self.z_in(torch.zeros_like(z0)).unsqueeze(1) + self.type_z + t1
        a1_tok_dummy = self.a_in(torch.zeros_like(a0)).unsqueeze(1) + self.type_a + t1

        if self.use_s_token:
            s1_tok_dummy = self.s_in(torch.zeros_like(s0)).unsqueeze(1) + self.type_s + t1
            bundle1 = torch.cat([z1_tok_dummy, a1_tok_dummy, s1_tok_dummy], dim=1)
        else:
            bundle1 = torch.cat([z1_tok_dummy, a1_tok_dummy], dim=1)
        seq = torch.cat([bundle0, bundle1], dim=1)  # [B, 2*bundle, H]
        return seq

    def forward_step(self, z0, a0, s0):
        # 仅用 t=0 的块去预测 z1
        seq = self._build_tokens(z0, a0, s0)
        S = seq.size(1); b = self.bundle
        # 构造块因果 mask: 仅允许 attend 到 t<=当前 的块
        m = torch.ones(S, S, device=seq.device, dtype=torch.bool)
        # t=0 可看自身; t=1 不可看 t=1 以后的(本例无)
        for ti in range(2):
            for tj in range(2):
                if tj > ti:
                    m[ti*b:(ti+1)*b, tj*b:(tj+1)*b] = False
        attn_mask = (~m).float() * -1e9
        out = self.tr(seq, mask=attn_mask)
        # 取 t=0 的 Z 槽位表征, 经过 head 预测 z1
        z0_slot = out[:, 0, :]  # 第一个 token 是 Z0
        z1_pred = self.head(z0_slot)
        return z1_pred  # [B, dz]

class Stats:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean.clone(); self.std = std.clone(); self.std[self.std<1e-6]=1e-6
    @classmethod
    def from_numpy(cls, x: np.ndarray, device):
        mean = torch.from_numpy(x.mean(axis=0)).float().to(device)
        std  = torch.from_numpy(x.std(axis=0)).float().to(device)
        std[std<1e-6]=1e-6
        return cls(mean, std)
    def norm(self, x: torch.Tensor):
        return (x - self.mean) / self.std

@torch.no_grad()
def compute_latent_stats(encoder: nn.Module, obs_np: np.ndarray, s_stats: Stats, device: torch.device, batch: int = 4096):
    N = obs_np.shape[0]
    mu = None; m2 = None; n = 0
    for i in range(0, N, batch):
        s = torch.from_numpy(obs_np[i:i+batch]).float().to(device)
        s = s_stats.norm(s)
        z = encoder(s)
        if mu is None:
            mu = z.mean(dim=0)
            m2 = ((z - mu)**2).sum(dim=0)
            n = z.size(0)
        else:
            n_new = n + z.size(0)
            delta = z.mean(dim=0) - mu
            mu = mu + delta * (z.size(0)/n_new)
            m2 = m2 + ((z - mu)**2).sum(dim=0)
            n = n_new
    var = m2 / max(1, n-1)
    std = torch.sqrt(var); std[std<1e-6]=1e-6
    return mu, std

@dataclass
class GatingConfig:
    K: int = 1                 # 用前 K 步做平均能量(默认 1)
    top_p: Optional[float] = None  # 保留每个 env 中能量最小的 p 分位候选
    tau: Optional[float] = None    # 绝对阈值, e<=tau 保留
    lambda_pen: float = 0.0    # 软惩罚权重, 外部自行 value -= λ*e
    policy_steps: int = 10     # diffusion policy 在门控时的采样步数(加速)

# ----------------- 门控主体 -----------------
class SJEPAGate:
    def __init__(self, env_id: str, encoder_ckpt: str, ac_ckpt: str,
                 use_s_token: bool = True, latent_whiten: bool = True,
                 device: str = 'cuda'):
        self.device = torch.device(device)

        env = gym.make(env_id)
        ds = env.get_dataset() if hasattr(env, 'get_dataset') else __import__('d4rl').qlearning_dataset(env)
        obs_np = ds['observations'].astype(np.float32)
        acts_np = ds['actions'].astype(np.float32)
        self.state_dim = obs_np.shape[1]
        self.act_dim = acts_np.shape[1]
        self.s_stats = Stats.from_numpy(obs_np, self.device)

        # 2) 编码器 + AC 预测器
        self.encoder = Encoder(self.state_dim, embed_dim=256).to(self.device)
        sd_e = torch.load(encoder_ckpt, map_location=self.device)
        if isinstance(sd_e, dict) and 'state_dict' in sd_e: sd_e = sd_e['state_dict']
        self.encoder.load_state_dict(sd_e); self.encoder.eval()

        self.predictor = ACTransformer(z_dim=256, s_dim=self.state_dim, a_dim=self.act_dim,
                                       hidden=512, layers=8, nhead=8, use_s_token=use_s_token).to(self.device)
        sd_p = torch.load(ac_ckpt, map_location=self.device)
        sd_p = sd_p.get('predictor', sd_p)  # 兼容 {'predictor': state_dict}
        self.predictor.load_state_dict(sd_p); self.predictor.eval()

        # 3) z 白化
        self.latent_whiten = latent_whiten
        if latent_whiten:
            mu, std = compute_latent_stats(self.encoder, obs_np, self.s_stats, self.device)
            self.z_mu = mu; self.z_std = std
        else:
            self.z_mu = torch.zeros(256, device=self.device)
            self.z_std = torch.ones(256, device=self.device)

    @torch.no_grad()
    def _encode_state(self, s_raw: torch.Tensor) -> torch.Tensor:
        # s_raw: [N, Ds] (未归一化的原始状态)
        s_n = self.s_stats.norm(s_raw)
        z = self.encoder(s_n)
        if self.latent_whiten:
            z = (z - self.z_mu) / self.z_std
        return z


    @torch.no_grad()
    def score_k_steps(self,
                      obs_raw: torch.Tensor,                
                      traj_norm: torch.Tensor,         
                      normalizer,       
                      policy=None, invdyn=None,
                      pipeline_type: str = 'separate',
                      obs_dim: int = None, act_dim: int = None,
                      solver: str = None, use_ema: bool = None, temperature: float = None,
                      gate_cfg: Optional[GatingConfig] = None,
                      rebase_policy: bool = False) -> torch.Tensor:
        
        assert obs_dim is not None and act_dim is not None
        device = self.device
        gate_cfg = gate_cfg or GatingConfig()

        B, C, H, P = traj_norm.shape
        K = int(gate_cfg.K) if gate_cfg.K is not None else 1
        K = max(1, min(K, H - 1)) 

      
        s_norm_all = traj_norm[:, :, :, :obs_dim] 
        
        s_raw_all = torch.as_tensor(
            normalizer.unnormalize(s_norm_all.reshape(B * C * H, obs_dim).detach().cpu().numpy()),
            device=device, dtype=torch.float32
        ).view(B, C, H, obs_dim)

        
       
        s0_raw = obs_raw.to(device)                                  
        if getattr(normalizer, "center_mapping", True):
            s_raw_all[:, :, 1:, 0:2] += s0_raw[:, None, None, 0:2]

        s0_norm = torch.as_tensor(
            normalizer.normalize(s0_raw.detach().cpu().numpy()),
            device=device, dtype=torch.float32
        )  

        E_accum = torch.zeros((B, C), device=device, dtype=torch.float32)

        for t in range(K):
           
            if t == 0:
                s_t_raw  = s0_raw.unsqueeze(1).repeat(1, C, 1)       # [B, C, Ds]
                s_t_norm = s0_norm.unsqueeze(1).repeat(1, C, 1)      # [B, C, Ds]
            else:
                s_t_raw  = s_raw_all[:, :, t, :]                     # [B, C, Ds]
                s_t_norm = s_norm_all[:, :, t, :]                    # [B, C, Ds]

            s_tp1_norm = s_norm_all[:, :, t+1, :]                    # [B, C, Ds]
            s_tp1_raw  = s_raw_all[:, :, t+1, :]                     # [B, C, Ds]

           
            if pipeline_type == 'separate':
              
                obs_policy      = s_t_norm.reshape(B * C, obs_dim)       # [B*C, Ds]
                next_obs_policy = s_tp1_norm.reshape(B * C, obs_dim)     # [B*C, Ds]
                if rebase_policy:
                    next_obs_policy[:, :2] -= obs_policy[:, :2]
                    obs_policy[:, :2] = 0

                if policy is not None:
                    policy_prior = torch.zeros((B * C, act_dim), device=device)
                    act, _ = policy.sample(
                        policy_prior,
                        solver=solver,
                        n_samples=B * C,
                        sample_steps=gate_cfg.policy_steps,
                        condition_cfg=torch.cat([obs_policy, next_obs_policy], dim=-1),
                        w_cfg=1.0, use_ema=(True if use_ema is None else use_ema),
                        temperature=(1.0 if temperature is None else temperature)
                    )  # [B*C, Da]
                    a_t = act.view(B, C, act_dim)
                elif invdyn is not None:
                    a_flat = invdyn.predict(obs_policy, next_obs_policy)  # [B*C, Da]
                    a_t = a_flat.view(B, C, act_dim)
                else:
                    raise ValueError('separate 模式需要 policy 或 invdyn 之一')
            else:

                a_t = traj_norm[:, :, t, obs_dim:obs_dim + act_dim]      # [B, C, Da]

            
            z_t = self._encode_state(s_t_raw.reshape(B * C, -1))                 # [B*C, dz]
            z_plan_tp1 = self._encode_state(s_tp1_raw.reshape(B * C, -1))        # [B*C, dz]

            
            if self.predictor.use_s_token:
                s_tn = self.s_stats.norm(s_t_raw.reshape(B * C, -1))             # [B*C, Ds]
            else:
                s_tn = None

           
            z_hat_tp1 = self.predictor.forward_step(
                z_t, a_t.reshape(B * C, -1), s_tn
            )  

           
            e_t = (z_hat_tp1 - z_plan_tp1).abs().mean(dim=-1).view(B, C)
            E_accum += e_t

        E_mean = E_accum / float(K)  
        return E_mean


    @staticmethod
    def make_mask(E: torch.Tensor, top_p: Optional[float] = None, tau: Optional[float] = None) -> torch.Tensor:
        
        B, C = E.shape
        keep = torch.ones_like(E, dtype=torch.bool)
        if top_p is not None:
            k = max(1, int(round(C * float(top_p))))
          
            idx = torch.topk(-E, k=k, dim=1).indices 
            m = torch.zeros_like(keep)
            ar = torch.arange(B).unsqueeze(1)
            m[ar, idx] = True
            keep = keep & m
        if tau is not None:
            keep = keep & (E <= tau)
        return keep
    

