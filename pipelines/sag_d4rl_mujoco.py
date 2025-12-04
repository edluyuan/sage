import os
import sys
import d4rl
import gym
import hydra, wandb, uuid
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from cleandiffuser.classifier import CumRewClassifier
from cleandiffuser.dataset.d4rl_mujoco_dataset import DV_D4RLMuJoCoSeqDataset, D4RLMuJoCoTDDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader, loop_two_dataloaders
from cleandiffuser.diffusion import ContinuousDiffusionSDE, DiscreteDiffusionSDE
from cleandiffuser.invdynamic import MlpInvDynamic
from cleandiffuser.nn_condition import MLPCondition, IdentityCondition
from cleandiffuser.nn_diffusion import DiT1d, DVInvMlp
from cleandiffuser.nn_classifier import HalfJannerUNet1d
from cleandiffuser.nn_diffusion import JannerUNet1d
from cleandiffuser.utils import report_parameters, DD_RETURN_SCALE, DVHorizonCritic, IDQLVNet
from utils import set_seed
from tqdm import tqdm
from omegaconf import OmegaConf


#from gates.sjepa_gate import SJEPAGate, GatingConfig

from typing import Optional
from dataclasses import dataclass

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
        import gym
        import d4rl  # noqa: F401
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
    def score_first_step(self,
                         obs_raw: torch.Tensor,                     # [B, Ds] 原始观测(未normalize)
                         traj_norm: torch.Tensor,                   # [B, C, H, planner_dim] 规划器归一化输出
                         normalizer,                                # 提供 denormalize()
                         policy=None, invdyn=None,                  # 两者二选一(separate 模式),或都 None(联合模式从 traj 拿动作)
                         pipeline_type: str = 'separate',
                         obs_dim: int = None, act_dim: int = None,
                         solver: str = None, use_ema: bool = None, temperature: float = None,
                         gate_cfg: Optional[GatingConfig] = None,
                         rebase_policy: bool = False) -> torch.Tensor:
        """返回可行性能量 E: [B, C]. 默认只看第一步(K=1)。
        - separate: 由 policy/invdyn 产生 a0
        - 联合: 从 traj 直接取 a0
        """
        assert obs_dim is not None and act_dim is not None
        B, C, H, P = traj_norm.shape
        device = self.device
        gate_cfg = gate_cfg or GatingConfig()

        # 0) 拿到计划的下一帧 s^plan_1 (原始尺度)
        s1_plan_norm = traj_norm[:, :, 1, :obs_dim]                 # [B, C, Ds_norm]



        def _obs_denorm(norm, x_t: torch.Tensor) -> torch.Tensor:
            x_np = x_t.detach().cpu().numpy()
            if hasattr(norm, 'denormalize'):
                y = norm.denormalize(x_np)
                return torch.as_tensor(y, device=x_t.device, dtype=torch.float32)
            if hasattr(norm, 'unnormalize'):
                y = norm.unnormalize(x_np)
                return torch.as_tensor(y, device=x_t.device, dtype=torch.float32)
            if hasattr(norm, 'denorm'):
                y = norm.denorm(x_np)
                return torch.as_tensor(y, device=x_t.device, dtype=torch.float32)
            mean = getattr(norm, 'obs_mean', getattr(norm, 'mean', None))
            std  = getattr(norm, 'obs_std',  getattr(norm, 'std',  None))
            if mean is None or std is None:
                raise AttributeError("Normalizer needs denormalize/unnormalize or (obs_)mean/(obs_)std.")
            mean_t = torch.as_tensor(mean, device=x_t.device, dtype=torch.float32)
            std_t  = torch.as_tensor(std,  device=x_t.device, dtype=torch.float32).clamp_min(1e-6)
            return x_t * std_t + mean_t


        #s1_plan_raw = _obs_denorm(normalizer, s1_plan_norm)
        s1_plan_raw = torch.tensor(normalizer.unnormalize(s1_plan_norm.detach().cpu().numpy()), device=device, dtype=torch.float32)  # [B, C, Ds]
        s0_raw = obs_raw.to(device)                                  # [B, Ds]

        # 1) 产出 a0 (在原始动作尺度)
        if pipeline_type == 'separate':
            # 为每个 candidate 生成动作
            # 构造 policy 条件: 使用 normalize 之后的 obs & next_obs
            obs_norm = torch.tensor(normalizer.normalize(s0_raw.detach().cpu().numpy()), device=device, dtype=torch.float32)  # [B, Ds]
            s1_plan_norm_flat = s1_plan_norm.reshape(B*C, obs_dim)
            obs_norm_rep = obs_norm.unsqueeze(1).repeat(1, C, 1).reshape(B*C, obs_dim)
            if policy is not None:  # diffusion policy
                policy_prior = torch.zeros((B*C, act_dim), device=device)
                obs_policy = obs_norm_rep.clone()
                next_obs_policy = s1_plan_norm_flat.clone()
                if rebase_policy:
                    next_obs_policy[:, :2] -= obs_policy[:, :2]
                    obs_policy[:, :2] = 0
                act, _ = policy.sample(
                    policy_prior, solver=solver, n_samples=B*C,
                    sample_steps=gate_cfg.policy_steps, condition_cfg=torch.cat([obs_policy, next_obs_policy], dim=-1),
                    w_cfg=1.0, use_ema=(use_ema if use_ema is not None else True), temperature=(temperature if temperature is not None else 1.0))
                a0 = act  # [B*C, Da] (已在原本策略尺度, 通常为原始动作范围)
            elif invdyn is not None:  # MLP 逆动力学
                a0 = invdyn.predict(obs_norm_rep, s1_plan_norm_flat)  # [B*C, Da]
            else:
                raise ValueError('separate 模式需要 policy 或 invdyn 之一')
        else:
            # 联合模式: 直接从规划轨迹取动作 a0 (注意: 可能是 normalize 过的, 需要 denormalize)
            a0_norm = traj_norm[:, :, 0, obs_dim:obs_dim+act_dim].reshape(B*C, act_dim)
            # 若 normalizer 提供动作反归一化, 可改成 normalizer.denormalize_action
            a0 = a0_norm  # 假设轨迹中的动作已在原始[-1,1] 范围
        a0 = a0.reshape(B, C, act_dim)

        # 2) 编码 z0, z1_plan
        z0 = self._encode_state(s0_raw)                       # [B, dz]
        z0 = z0.unsqueeze(1).repeat(1, C, 1).reshape(B*C, -1) # [B*C, dz]
        z1_plan = self._encode_state(s1_plan_raw.reshape(B*C, -1))  # [B*C, dz]

        # 3) 通过 AC 预测器得到 z_hat1
        if self.predictor.use_s_token:
            s0n = self.s_stats.norm(s0_raw).unsqueeze(1).repeat(1, C, 1).reshape(B*C, -1)
        else:
            s0n = None
        z_hat1 = self.predictor.forward_step(z0, a0.reshape(B*C, -1), s0n)  # [B*C, dz]

        # 4) 能量: mean L1 per-dim -> [B, C]
        e = (z_hat1 - z1_plan).abs().mean(dim=-1).reshape(B, C)
        return e

    @torch.no_grad()
    def score_k_steps(self,
                      obs_raw: torch.Tensor,                 # [B, Ds] 原始观测(未normalize)
                      traj_norm: torch.Tensor,               # [B, C, H, planner_dim] 规划器的归一化轨迹
                      normalizer,
                      policy=None, invdyn=None,
                      pipeline_type: str = 'separate',
                      obs_dim: int = None, act_dim: int = None,
                      solver: str = None, use_ema: bool = None, temperature: float = None,
                      gate_cfg: Optional[GatingConfig] = None,
                      rebase_policy: bool = False) -> torch.Tensor:
        """返回 K 步平均能量: [B, C]
        第 t 步的能量定义为 mean(| z_hat_{t+1} - z^{plan}_{t+1} |)，
        其中:
          z_hat_{t+1} = AC(z_t, a_t, s_t) 的一步预测，
          z^{plan}_{t+1} 来自计划的下一帧状态编码。
        t=0 使用真实 s_0=obs_raw；t>0 使用计划中的 s^{plan}_t。
        """
        assert obs_dim is not None and act_dim is not None
        device = self.device
        gate_cfg = gate_cfg or GatingConfig()

        B, C, H, P = traj_norm.shape
        K = int(gate_cfg.K) if gate_cfg.K is not None else 1
        K = max(1, min(K, H - 1))  # 最多只能到 H-1 步

        # 取出所有归一化的状态序列 [B, C, H, obs_dim]
        s_norm_all = traj_norm[:, :, :, :obs_dim]  # 归一化的 s_t

        # 反归一化得到原始尺度状态 [B, C, H, obs_dim]
        s_raw_all = torch.as_tensor(
            normalizer.unnormalize(s_norm_all.reshape(B * C * H, obs_dim).detach().cpu().numpy()),
            device=device, dtype=torch.float32
        ).view(B, C, H, obs_dim)

        # t=0 的真实观测：raw & norm（注意：norm 需用 normalizer 对 obs_raw 的 numpy 版本）
        s0_raw = obs_raw.to(device)                                   # [B, Ds]
        s0_norm = torch.as_tensor(
            normalizer.normalize(s0_raw.detach().cpu().numpy()),
            device=device, dtype=torch.float32
        )  # [B, Ds]

        E_accum = torch.zeros((B, C), device=device, dtype=torch.float32)

        for t in range(K):
            # s_t 与 s_{t+1}^{plan}
            if t == 0:
                s_t_raw  = s0_raw.unsqueeze(1).repeat(1, C, 1)       # [B, C, Ds]
                s_t_norm = s0_norm.unsqueeze(1).repeat(1, C, 1)      # [B, C, Ds]
            else:
                s_t_raw  = s_raw_all[:, :, t, :]                     # [B, C, Ds]
                s_t_norm = s_norm_all[:, :, t, :]                    # [B, C, Ds]

            s_tp1_norm = s_norm_all[:, :, t+1, :]                    # [B, C, Ds]
            s_tp1_raw  = s_raw_all[:, :, t+1, :]                     # [B, C, Ds]

            # 生成 a_t（原始动作尺度，通常为 [-1,1]）
            if pipeline_type == 'separate':
                # 给策略/逆动力学准备条件（全部在“归一化状态空间”里）
                obs_policy      = s_t_norm.reshape(B * C, obs_dim)       # [B*C, Ds]
                next_obs_policy = s_tp1_norm.reshape(B * C, obs_dim)     # [B*C, Ds]
                if rebase_policy:
                    # 可选的坐标重基
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
                # 联合模式: 轨迹里自带动作（已在原动作范围）
                a_t = traj_norm[:, :, t, obs_dim:obs_dim + act_dim]      # [B, C, Da]

            # 编码 z_t 与 z_{t+1}^{plan}
            z_t = self._encode_state(s_t_raw.reshape(B * C, -1))                 # [B*C, dz]
            z_plan_tp1 = self._encode_state(s_tp1_raw.reshape(B * C, -1))        # [B*C, dz]

            # 若预测器使用 s token，传入标准化后的 s_t
            if self.predictor.use_s_token:
                s_tn = self.s_stats.norm(s_t_raw.reshape(B * C, -1))             # [B*C, Ds]
            else:
                s_tn = None

            # 一步预测 z_hat_{t+1}
            z_hat_tp1 = self.predictor.forward_step(
                z_t, a_t.reshape(B * C, -1), s_tn
            )  # [B*C, dz]

            # 本步能量 e_t -> [B, C]
            e_t = (z_hat_tp1 - z_plan_tp1).abs().mean(dim=-1).view(B, C)
            E_accum += e_t

        E_mean = E_accum / float(K)   # K 步平均
        return E_mean


    @staticmethod
    def make_mask(E: torch.Tensor, top_p: Optional[float] = None, tau: Optional[float] = None) -> torch.Tensor:
        """E: [B, C] -> bool mask(True=保留)。top_p 与 tau 可同时使用(取交集)。"""
        B, C = E.shape
        keep = torch.ones_like(E, dtype=torch.bool)
        if top_p is not None:
            k = max(1, int(round(C * float(top_p))))
            # 对每个 batch 取能量最小的前 k
            idx = torch.topk(-E, k=k, dim=1).indices  # 负号=按小到大
            m = torch.zeros_like(keep)
            ar = torch.arange(B).unsqueeze(1)
            m[ar, idx] = True
            keep = keep & m
        if tau is not None:
            keep = keep & (E <= tau)
        return keep













@hydra.main(config_path="../configs/veteran/mujoco", config_name="mujoco", version_base=None)
def pipeline(args):
    args.device = args.device if torch.cuda.is_available() else "cpu"
    if args.enable_wandb and args.mode in ["inference", "train"]:
        wandb.require("core")
        print(args)
        wandb.init(
            reinit=True,
            id=str(uuid.uuid4()),
            project=str(args.project),
            group=str(args.group),
            name=str(args.name),
            config=OmegaConf.to_container(args, resolve=True)
        )

    set_seed(args.seed)
    
    # base config
    base_path = f"{args.pipeline_name}_H{args.task.planner_horizon}_Jump{args.task.stride}"
    base_path += f"_next{args.planner_next_obs_loss_weight}"
    # guidance type
    base_path += f"_{args.guidance_type}"
    # For Planner
    base_path += f"_{args.planner_net}"
    if args.planner_net == "transformer":
        base_path += f"_d{args.planner_depth}"
        base_path += f"_width{args.planner_d_model}"
    elif args.planner_net == "unet":
        base_path += f"_width{args.unet_dim}"
    
    if not args.planner_predict_noise:
        base_path += f"_pred_x0"
    
    # pipeline_type
    base_path += f"_{args.pipeline_type}"
    base_path += f"_dp{args.use_diffusion_invdyn}"
    base_path += f"_penalty{args.terminal_penalty}"
    base_path += f"_bonus{args.full_traj_bonus}"
    base_path += f"_gamma{args.discount}"
    base_path += f"_adv{args.use_weighted_regression}"
    base_path += f"_weight{args.weight_factor}"
    # task name
    base_path += f"/{args.task.env_name}/"
    
    save_path = f"{args.save_dir}/" + base_path
    video_path = "video_outputs/" + base_path
    
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    
    if os.path.exists(video_path) is False:
        os.makedirs(video_path)

    # ---------------------- Create Dataset ----------------------
    env = gym.make(args.task.env_name)
    planner_dataset = DV_D4RLMuJoCoSeqDataset(
        env.get_dataset(), horizon=args.task.planner_horizon, discount=args.discount, 
        stride=args.task.stride, center_mapping=(args.guidance_type!="cfg"),
        terminal_penalty=args.terminal_penalty,
        full_traj_bonus=args.full_traj_bonus
    )
    policy_dataset = DV_D4RLMuJoCoSeqDataset(
        env.get_dataset(), horizon=args.task.planner_horizon, discount=args.discount, 
        stride=args.task.stride, center_mapping=(args.guidance_type!="cfg"),
        terminal_penalty=args.terminal_penalty,
        full_traj_bonus=args.full_traj_bonus
    )
    planner_dataloader = DataLoader(
        planner_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    obs_dim, act_dim = planner_dataset.o_dim, planner_dataset.a_dim
    
    policy_dataloader = DataLoader(
        policy_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    obs_dim, act_dim = planner_dataset.o_dim, planner_dataset.a_dim

    planner_dim = obs_dim if args.pipeline_type=="separate" else obs_dim + act_dim

    # --------------- Network Architecture -----------------
    if args.planner_net == "transformer":
        nn_diffusion_planner = DiT1d(
            planner_dim, emb_dim=args.planner_emb_dim,
            d_model=args.planner_d_model, n_heads=args.planner_d_model//32, depth=args.planner_depth, timestep_emb_type="fourier")
    elif args.planner_net == "unet":
        nn_diffusion_planner = JannerUNet1d(
            planner_dim, model_dim=args.unet_dim, emb_dim=args.unet_dim,
            timestep_emb_type="positional", attention=False, kernel_size=5)
    
    nn_condition_planner = None
    classifier = None
        
    if args.guidance_type == "MCSS":
        # --------------- Horizon Critic -----------------
        critic = DVHorizonCritic(
            planner_dim, emb_dim=args.planner_emb_dim,
            d_model=args.planner_d_model, n_heads=args.planner_d_model//32, depth=2, norm_type="pre").to(args.device)
        critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_learning_rate)
        print(f"=============== Parameter Report of Value ====================================")
        report_parameters(critic)
        print(f"==============================================================================")
        
    elif args.guidance_type=="cfg":
        if args.planner_net == "transformer":
            nn_condition_planner = MLPCondition(
                in_dim=1, out_dim=args.planner_emb_dim, hidden_dims=[args.planner_emb_dim, ], act=nn.SiLU(), dropout=0.25)
        elif args.planner_net == "unet":
            nn_condition_planner = MLPCondition(
                in_dim=1, out_dim=args.unet_dim, hidden_dims=[args.unet_dim, ], act=nn.SiLU(), dropout=0.25)
    
    elif args.guidance_type=="cg":
        nn_classifier = HalfJannerUNet1d(
            args.task.planner_horizon, planner_dim, out_dim=1,
            model_dim=args.unet_dim, emb_dim=args.unet_dim,
            timestep_emb_type="positional", kernel_size=3)
        classifier = CumRewClassifier(nn_classifier, device=args.device)
        print(f"=============== Parameter Report of Classifier ===============================")
        report_parameters(nn_classifier)
        print(f"==============================================================================")

    print(f"=============== Parameter Report of Planner ==================================")
    report_parameters(nn_diffusion_planner)
    print(f"==============================================================================")

    # ----------------- Masking -------------------
    fix_mask = torch.zeros((args.task.planner_horizon, planner_dim))
    fix_mask[0, :obs_dim] = 1.
    loss_weight = torch.ones((args.task.planner_horizon, planner_dim))
    loss_weight[1] = args.planner_next_obs_loss_weight

    # --------------- Diffusion Model with Classifier-Free Guidance --------------------
    planner = ContinuousDiffusionSDE(
        nn_diffusion_planner, nn_condition=nn_condition_planner,
        fix_mask=fix_mask, loss_weight=loss_weight, classifier=classifier, ema_rate=args.planner_ema_rate,
        device=args.device, predict_noise=args.planner_predict_noise, noise_schedule="linear")

    # --------------- Inverse Dynamic (Policy) -------------------
    if args.pipeline_type=="separate":
        if args.use_diffusion_invdyn:
            nn_diffusion_invdyn = DVInvMlp(obs_dim, act_dim, emb_dim=64, hidden_dim=args.policy_hidden_dim, timestep_emb_type="positional").to(args.device)
            nn_condition_invdyn = IdentityCondition(dropout=0.0).to(args.device)
            print(f"=============== Parameter Report of Policy ===================================")
            report_parameters(nn_diffusion_invdyn)
            print(f"==============================================================================")
            # --------------- Diffusion Model Actor --------------------
            policy = DiscreteDiffusionSDE(
                nn_diffusion_invdyn, nn_condition_invdyn, predict_noise=args.policy_predict_noise, optim_params={"lr": args.policy_learning_rate},
                x_max=+1. * torch.ones((1, act_dim), device=args.device),
                x_min=-1. * torch.ones((1, act_dim), device=args.device),
                diffusion_steps=args.policy_diffusion_steps, ema_rate=args.policy_ema_rate, device=args.device)
        else:
            invdyn = MlpInvDynamic(obs_dim, act_dim, 512, nn.Tanh(), {"lr": 2e-4}, device=args.device)

    
    
    # ---------------------- Inference ----------------------
    if args.mode == "inference":
        
        if args.guidance_type=="MCSS":
            # load planner
            planner.load(save_path + f"planner_ckpt_{args.planner_ckpt}.pt")
            planner.eval()
            # load critic
            critic_ckpt = torch.load(save_path + f"critic_ckpt_{args.critic_ckpt}.pt")
            critic.load_state_dict(critic_ckpt["critic"])
            critic.eval()
            # load policy
            if args.pipeline_type == "separate":
                if args.use_diffusion_invdyn:
                    policy.load(save_path + f"policy_ckpt_{args.policy_ckpt}.pt")
                    policy.eval()
                else:
                    invdyn.load(save_path + f"invdyn_ckpt_{args.invdyn_ckpt}.pt")
                    invdyn.eval()
        
        elif args.guidance_type=="cfg":
            # load planner
            planner.load(save_path + f"planner_ckpt_{args.planner_ckpt}.pt")
            planner.eval()
            # load policy
            if args.pipeline_type == "separate":
                if args.use_diffusion_invdyn:
                    policy.load(save_path + f"policy_ckpt_{args.policy_ckpt}.pt")
                    policy.eval()
                else:
                    invdyn.load(save_path + f"invdyn_ckpt_{args.invdyn_ckpt}.pt")
                    invdyn.eval()
            
        elif args.guidance_type=="cg":
            # load planner
            planner.load(save_path + f"planner_ckpt_{args.planner_ckpt}.pt")
            # load classifier
            planner.classifier.load(save_path + f"classifier_ckpt_{args.planner_ckpt}.pt")
            planner.eval()
            # load policy
            if args.pipeline_type == "separate":
                if args.use_diffusion_invdyn:
                    policy.load(save_path + f"policy_ckpt_{args.policy_ckpt}.pt")
                    policy.eval()
                else:
                    invdyn.load(save_path + f"invdyn_ckpt_{args.invdyn_ckpt}.pt")
                    invdyn.eval()
                    
        
        MAX_VALUE_STEPS = 1_000_000
        
        EV = IDQLVNet(obs_dim, hidden_dim=256).to(args.device)
        ev_ckpt = torch.load(save_path + f"EV_ckpt_{MAX_VALUE_STEPS}.pt")
        EV.load_state_dict(ev_ckpt["ev"])
        EV.eval()

        env_eval = gym.vector.make(args.task.env_name, args.num_envs)
        normalizer = planner_dataset.get_normalizer()
        
        gate = SJEPAGate(
            env_id=args.task.env_name,
            encoder_ckpt=getattr(args, 'sj_encoder_ckpt', 'jepa/results/d4rl/walker2d-medium-expert-v2/seed42/encoder_ema.pt'),
            ac_ckpt=getattr(args, 'sj_ac_ckpt', 'jepa/results-ac/d4rl/walker2d-medium-expert-v2/seed42/ac_predictor_final.pt'),
            use_s_token=bool(getattr(args, 'sj_use_s_token', True)),
            latent_whiten=bool(getattr(args, 'sj_latent_whiten', True)),
            device=args.device,
        )
        gate_cfg = GatingConfig(
            K=int(getattr(args, 'gate_K', 10)),
            top_p=float(getattr(args, 'gate_top_p', 0.8)),
            tau=getattr(args, 'gate_tau', None),
            lambda_pen=float(getattr(args, 'gate_lambda_pen', 0.0)),
            policy_steps=int(getattr(args, 'gate_policy_steps', 20)),
        )


        episode_rewards = []
        
        for i in range(args.num_episodes):
            obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0
            while not np.all(cum_done) and t < args.task.max_path_length + 1:
                
                # 1) generate plan
                if args.guidance_type == "MCSS":
                    B, C, H = args.num_envs, args.planner_num_candidates, args.task.planner_horizon
                    planner_prior = torch.zeros((B * C, H, planner_dim), device=args.device)

                    # keep RAW numpy obs, and create a normalized Torch tensor separately
                    obs_np_raw = obs                                              # np.ndarray from env
                    obs_norm_np = normalizer.normalize(obs_np_raw)                # np.ndarray
                    obs_norm_t  = torch.as_tensor(obs_norm_np, device=args.device, dtype=torch.float32)

                    # tile normalized obs across candidates
                    obs_repeat = obs_norm_t.unsqueeze(1).repeat(1, C, 1).view(-1, obs_dim)
                    planner_prior[:, 0, :obs_dim] = obs_repeat

                    # sample normalized trajectories: traj shape [B*C, H, planner_dim]
                    traj, log = planner.sample(
                        planner_prior, solver=args.planner_solver,
                        n_samples=B * C, sample_steps=args.planner_sampling_steps, use_ema=args.planner_use_ema,
                        condition_cfg=None, w_cfg=1.0, temperature=args.task.planner_temperature
                    )

                    # reshape for value/gating
                    traj_bchd = traj.view(B, C, H, planner_dim)

                    # value scores per candidate
                    with torch.no_grad():
                        v_td  = EV(traj)[:, 1:]            # [B*C, H-1]
                        v_sum = v_td.sum(dim=1).view(B, C) # [B, C]

                    # S-JEPA first-step feasibility gating
                    feas_E = gate.score_k_steps(
                        obs_raw=torch.as_tensor(obs_np_raw, device=args.device, dtype=torch.float32),
                        traj_norm=traj_bchd,
                        normalizer=normalizer,
                        policy=policy if (args.pipeline_type == "separate" and args.use_diffusion_invdyn) else None,
                        invdyn=invdyn if (args.pipeline_type == "separate" and not args.use_diffusion_invdyn) else None,
                        pipeline_type=args.pipeline_type,
                        obs_dim=obs_dim, act_dim=act_dim,
                        solver=args.policy_solver if (args.pipeline_type == "separate" and args.use_diffusion_invdyn) else None,
                        use_ema=bool(args.policy_use_ema) if (args.pipeline_type == "separate" and args.use_diffusion_invdyn) else None,
                        temperature=args.policy_temperature if (args.pipeline_type == "separate" and args.use_diffusion_invdyn) else None,
                        gate_cfg=gate_cfg,
                        rebase_policy=bool(args.rebase_policy),
                    )


                    mask = SJEPAGate.make_mask(feas_E, top_p=gate_cfg.top_p, tau=gate_cfg.tau)  # [B, C]
                    v_masked = torch.where(mask, v_sum, torch.full_like(v_sum, -1e9))
                    if gate_cfg.lambda_pen > 0.0:
                        v_masked = v_masked - gate_cfg.lambda_pen * feas_E

                    # best candidate per env -> traj: [B, H, planner_dim]
                    best_idx = torch.argmax(v_masked, dim=-1)
                    traj = traj_bchd[torch.arange(B, device=args.device), best_idx]

                elif args.guidance_type == "cfg":
                    B, H = args.num_envs, args.task.planner_horizon
                    planner_prior = torch.zeros((B, H, planner_dim), device=args.device)
                    condition = torch.ones((B, 1), device=args.device) * args.task.planner_target_return

                    obs_np_raw = obs
                    obs_norm_np = normalizer.normalize(obs_np_raw)
                    obs_norm_t  = torch.as_tensor(obs_norm_np, device=args.device, dtype=torch.float32)

                    planner_prior[:, 0, :obs_dim] = obs_norm_t
                    traj, log = planner.sample(
                        planner_prior, solver=args.planner_solver,
                        n_samples=B, sample_steps=args.planner_sampling_steps, use_ema=args.planner_use_ema,
                        condition_cfg=condition, w_cfg=args.task.planner_w_cfg, temperature=args.task.planner_temperature
                    )

                elif args.guidance_type == "cg":
                    B, C, H = args.num_envs, args.planner_num_candidates, args.task.planner_horizon
                    planner_prior = torch.zeros((B * C, H, planner_dim), device=args.device)

                    obs_np_raw = obs
                    obs_norm_np = normalizer.normalize(obs_np_raw)
                    obs_norm_t  = torch.as_tensor(obs_norm_np, device=args.device, dtype=torch.float32)

                    obs_repeat = obs_norm_t.unsqueeze(1).repeat(1, C, 1).view(-1, obs_dim)
                    planner_prior[:, 0, :obs_dim] = obs_repeat

                    traj, log = planner.sample(
                        planner_prior, solver=args.planner_solver,
                        n_samples=B * C, sample_steps=args.planner_sampling_steps, use_ema=args.planner_use_ema,
                        w_cg=args.task.planner_w_cfg, temperature=args.task.planner_temperature
                    )

                    with torch.no_grad():
                        logp = log["log_p"].view(B, C)
                        idx = torch.argmax(logp, -1)
                        traj = traj.view(B, C, H, planner_dim)[torch.arange(B), idx]


                # 2) generate action
                if args.pipeline_type == "separate":
                    if args.use_diffusion_invdyn:
                        policy_prior = torch.zeros((args.num_envs, act_dim), device=args.device)
                        with torch.no_grad():
                            next_obs_plan = traj[:, 1, :obs_dim]   # 修正切片
                            obs_policy      = obs_norm_t   # 用上面保存的归一化 obs
                            next_obs_policy = next_obs_plan.clone()
                            if args.rebase_policy:
                                next_obs_policy[:, :2] -= obs_policy[:, :2]
                                obs_policy[:, :2] = 0
                            act, log = policy.sample(
                                policy_prior,
                                solver=args.policy_solver,
                                n_samples=args.num_envs,
                                sample_steps=args.policy_sampling_steps,
                                condition_cfg=torch.cat([obs_policy, next_obs_policy], dim=-1),
                                w_cfg=1.0, use_ema=args.policy_use_ema, temperature=args.policy_temperature)
                            act = act.cpu().numpy()
                    else:
                        with torch.no_grad():
                            act = invdyn.predict(
                                obs_norm_t,
                                traj[:, 1, :obs_dim]
                            ).cpu().numpy()
                else:
                    act = traj[:, 0, obs_dim:].cpu().numpy()
                    
                # step
                obs, rew, done, info = env_eval.step(act)

                t += 1
                cum_done = done if cum_done is None else np.logical_or(cum_done, done)
                ep_reward += (rew * (1 - cum_done)) if t < args.task.max_path_length else rew
                # print(f'[t={t}] xy: {np.around(obs[:, :2], 2)}')
                print(f'[t={t}] rew: {np.around((rew * (1 - cum_done)), 2)}')

            episode_rewards.append(ep_reward)

        episode_rewards = [list(map(lambda x: env.get_normalized_score(x), r)) for r in episode_rewards]
        episode_rewards = np.array(episode_rewards).reshape(-1) * 100
        mean = np.mean(episode_rewards)
        err = np.std(episode_rewards) / np.sqrt(len(episode_rewards))
        print(mean, err)

        if args.enable_wandb:
            wandb.log({'Mean Reward': mean, 'Error': err})
            wandb.finish()

        
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    pipeline()
