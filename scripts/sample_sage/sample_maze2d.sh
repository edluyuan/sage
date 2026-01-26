#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-python}
MAIN=${MAIN:-pipelines/sage_d4rl_maze2d.py}

# ------------------ wandb / device ------------------
PROJECT=${PROJECT:-dv-sage-test-d4rl}
GROUP=${GROUP:-d4rl}
WANDB_MODE=${WANDB_MODE:-online}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export WANDB_MODE CUDA_VISIBLE_DEVICES

# ------------------ run spec ------------------
SEED=${SEED:-42}
ENV_ID=${ENV_ID:-maze2d-umaze-v1}   # or maze2d-umaze-v1 / maze2d-large-v1

# IMPORTANT: this matches your artifact layout
ENCODER_ROOT=${ENCODER_ROOT:-results/sage/pretrain_enc}
AC_ROOT=${AC_ROOT:-results/sage/posttrain_ac}

# DV ckpt selectors (planner/policy/critic live under save_path for this env)
PLANNER_CKPT=${PLANNER_CKPT:-latest}
POLICY_CKPT=${POLICY_CKPT:-latest}
CRITIC_CKPT=${CRITIC_CKPT:-latest}

# ------------------ SAGE params ------------------
K=${K:-10}
KEEP_P=${KEEP_P:-0.8}
LAM=${LAM:-0.1}

# Maze2D: actions are already in [-1,1] (diffusion policy), so typically False
SAGE_ACTIONS_TANH=${SAGE_ACTIONS_TANH:-false}

TAG=${TAG:-dv_sage_test}
RUN_NAME="${TAG}-${ENV_ID}-seed${SEED}-k${K}-p${KEEP_P}-lam${LAM}"

# ------------------ resolve SAGE artifacts ------------------
ENCODER_CKPT="${ENCODER_ROOT}/${ENV_ID}/seed${SEED}/encoder_ema.pt"
STATE_STATS="${ENCODER_ROOT}/${ENV_ID}/seed${SEED}/state_stats.npz"
AC_CKPT="${AC_ROOT}/${ENV_ID}/seed${SEED}/ac_predictor_final.pt"

[[ -f "${ENCODER_CKPT}" ]] || { echo "[ERR] missing ${ENCODER_CKPT}" >&2; exit 1; }
[[ -f "${STATE_STATS}"  ]] || { echo "[ERR] missing ${STATE_STATS}"  >&2; exit 1; }
[[ -f "${AC_CKPT}"      ]] || { echo "[ERR] missing ${AC_CKPT}"      >&2; exit 1; }

echo "============================================================"
echo "[ENV] ${ENV_ID} | seed ${SEED} | K=${K} keep_p=${KEEP_P} lambda=${LAM}"
echo "[ART] encoder=${ENCODER_CKPT}"
echo "[ART] stats  =${STATE_STATS}"
echo "[ART] ac     =${AC_CKPT}"
echo "------------------------------------------------------------"

${PY} ${MAIN} \
  mode="inference" \
  task="${ENV_ID}" \
  seed="${SEED}" \
  enable_wandb=1 \
  project="${PROJECT}" \
  group="${GROUP}" \
  name="${RUN_NAME}" \
  planner_ckpt="${PLANNER_CKPT}" \
  policy_ckpt="${POLICY_CKPT}" \
  critic_ckpt="${CRITIC_CKPT}" \
  +sage_encoder_ckpt="${ENCODER_CKPT}" \
  +sage_state_stats="${STATE_STATS}" \
  +sage_ac_ckpt="${AC_CKPT}" \
  +sage_prefix="${K}" \
  +sage_keep_p="${KEEP_P}" \
  +sage_lambda="${LAM}" \
  +sage_actions_tanh="${SAGE_ACTIONS_TANH}"
