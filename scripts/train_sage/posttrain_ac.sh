#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-python}
MAIN=${MAIN:-jepa/posttrain_ac.py}

PROJECT=${PROJECT:-sag-posttrain_ac-d4rl}
GROUP=${GROUP:-d4rl}
WANDB_MODE=${WANDB_MODE:-online}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export WANDB_MODE CUDA_VISIBLE_DEVICES

AC_STEPS=${AC_STEPS:-200000}
AC_BATCH=${AC_BATCH:-256}
AC_WINDOW=${AC_WINDOW:-16}

# scaled-down defaults
HIDDEN=${HIDDEN:-256}
LAYERS=${LAYERS:-2}
NHEAD=${NHEAD:-4}
DROPOUT=${DROPOUT:-0.0}

LR=${LR:-1e-4}
MIN_LR=${MIN_LR:-1e-6}
WARMUP=${WARMUP:-5000}

ROLLOUT_H=${ROLLOUT_H:-4}
ROLLOUT_W=${ROLLOUT_W:-1.0}

NEG_W=${NEG_W:-1.0}
NEG_MARGIN=${NEG_MARGIN:-0.10}

LATENT_WHITEN=${LATENT_WHITEN:-1}
ACTION_WHITEN=${ACTION_WHITEN:-1}
USE_S_TOKEN=${USE_S_TOKEN:-0}
DELTA_PRED=${DELTA_PRED:-1}

SEEDS=(${SEEDS:-42})

ENCODER_ROOT=${ENCODER_ROOT:-results/sag/pretrain_enc}
AC_ROOT=${AC_ROOT:-results/sag/posttrain_ac}

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

for ENV_ID in "${ENVS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    ENCODER_CKPT="${ENCODER_ROOT}/${ENV_ID}/seed${SEED}/encoder_ema.pt"
    STATE_STATS="${ENCODER_ROOT}/${ENV_ID}/seed${SEED}/state_stats.npz"
    OUT_DIR="${AC_ROOT}/${ENV_ID}/seed${SEED}"
    mkdir -p "${OUT_DIR}"

    RUN_NAME="ac-${ENV_ID}-seed${SEED}"

    ${PY} ${MAIN} \
      --env_id "${ENV_ID}" \
      --encoder_ckpt "${ENCODER_CKPT}" \
      --state_stats "${STATE_STATS}" \
      --ckpt_dir "${OUT_DIR}" \
      --steps ${AC_STEPS} \
      --batch_size ${AC_BATCH} \
      --window ${AC_WINDOW} \
      --hidden ${HIDDEN} \
      --layers ${LAYERS} \
      --nhead ${NHEAD} \
      --dropout ${DROPOUT} \
      --lr ${LR} --min_lr ${MIN_LR} --warmup_steps ${WARMUP} \
      --rollout_horizon ${ROLLOUT_H} --rollout_weight ${ROLLOUT_W} \
      --neg_weight ${NEG_W} --neg_margin ${NEG_MARGIN} \
      --seed ${SEED} \
      $( (( LATENT_WHITEN )) && echo --latent_whiten || echo --no-latent_whiten ) \
      $( (( ACTION_WHITEN )) && echo --action_whiten || echo --no-action_whiten ) \
      $( (( USE_S_TOKEN )) && echo --use_s_token || echo --no-use_s_token ) \
      $( (( DELTA_PRED )) && echo --delta_pred || echo --no-delta_pred ) \
      --wandb_project "${PROJECT}" \
      --wandb_run "${RUN_NAME}" \
      --wandb_group "${GROUP}" \
      --wandb_mode "${WANDB_MODE}"
  done
done
