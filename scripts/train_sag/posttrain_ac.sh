#!/usr/bin/env bash
set -euo pipefail

# ---------- Basic switches ----------
PY=${PY:-python}
MAIN=${MAIN:-gates/ac.py}

PROJECT=${PROJECT:-sag}
GROUP=${GROUP:-d4rl}
WANDB_MODE=${WANDB_MODE:-online}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export WANDB_MODE CUDA_VISIBLE_DEVICES

# ---------- Training hyperparams (AC stage) ----------
AC_STEPS=${AC_STEPS:-200000}
AC_BATCH=${AC_BATCH:-256}
AC_WINDOW=${AC_WINDOW:-16}

AC_HIDDEN=${AC_HIDDEN:-512}
AC_LAYERS=${AC_LAYERS:-8}
AC_NHEAD=${AC_NHEAD:-8}
AC_TOKEN_DROPOUT=${AC_TOKEN_DROPOUT:-0.0}

LR=${LR:-1e-4}
MIN_LR=${MIN_LR:-1e-6}
WARMUP=${WARMUP:-10000}

ROLLOUT_H=${ROLLOUT_H:-8}
ROLLOUT_W=${ROLLOUT_W:-1.0}

LATENT_WHITEN=${LATENT_WHITEN:-1}
ACTION_WHITEN=${ACTION_WHITEN:-1}
USE_S_TOKEN=${USE_S_TOKEN:-1}

LOG_INTERVAL=${LOG_INTERVAL:-200}

# ---------- Paths ----------
ENCODER_ROOT=${ENCODER_ROOT:-results/sag/jepa/d4rl}
AC_CKPT_ROOT=${AC_CKPT_ROOT:-results/sag/ac/d4rl}

# ---------- Seeds / Envs ----------
SEEDS=(${SEEDS:-42})

ENVS=(
 
  halfcheetah-medium-v2
  halfcheetah-medium-replay-v2
  halfcheetah-medium-expert-v2
  
  hopper-medium-v2
  hopper-medium-replay-v2
  hopper-medium-expert-v2

  walker2d-medium-v2
  walker2d-medium-replay-v2
  walker2d-medium-expert-v2



  maze2d-umaze-v1 
  maze2d-medium-v1 
  maze2d-large-v1

  kitchen-partial-v0 
  kitchen-mixed-v0

  antmaze-medium-play-v2 
  antmaze-medium-diverse-v2 
  antmaze-large-play-v2 
  antmaze-large-diverse-v2
  
)
# ---------- Loop ----------
for ENV_ID in "${ENVS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    # encoder path (pretrained State-JEPA encoder EMA)
    ENCODER_CKPT="${ENCODER_ROOT}/${ENV_ID}/seed${SEED}/encoder_ema.pt"

    # AC predictor output dir
    CKPT_DIR="${AC_CKPT_ROOT}/${ENV_ID}/seed${SEED}"

    RUN_NAME="ac-${ENV_ID}-seed${SEED}"

    echo "[Run] ENV=${ENV_ID} SEED=${SEED}"
    echo "      ENCODER=${ENCODER_CKPT}"
    echo "      OUT_DIR=${CKPT_DIR}"

    ${PY} ${MAIN} \
      --env_id "${ENV_ID}" \
      --seed ${SEED} \
      --encoder_ckpt "${ENCODER_CKPT}" \
      --ckpt_dir "${CKPT_DIR}" \
      --steps ${AC_STEPS} \
      --batch_size ${AC_BATCH} \
      --window ${AC_WINDOW} \
      --hidden ${AC_HIDDEN} \
      --layers ${AC_LAYERS} \
      --nhead ${AC_NHEAD} \
      --token_dropout ${AC_TOKEN_DROPOUT} \
      --lr ${LR} --min_lr ${MIN_LR} --warmup_steps ${WARMUP} \
      --rollout_horizon ${ROLLOUT_H} --rollout_weight ${ROLLOUT_W} \
      $( (( LATENT_WHITEN )) && echo --latent_whiten ) \
      $( (( ACTION_WHITEN )) && echo --action_whiten ) \
      $( (( USE_S_TOKEN )) && echo --use_s_token ) \
      --log_interval ${LOG_INTERVAL} \
      --wandb_project "${PROJECT}" \
      --wandb_run "${RUN_NAME}" \
      --wandb_group "${GROUP}" \
      --wandb_mode "${WANDB_MODE}"
  done
done
