#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-python}
MAIN=${MAIN:-gates/jepa.py}

PROJECT=${PROJECT:sag}
GROUP=${GROUP:-all}
WANDB_MODE=${WANDB_MODE:-online}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}


STEPS=${STEPS:-200000}
BATCH=${BATCH:-128}
WINDOW=${WINDOW:-64}
KMAX=${KMAX:-5}
NUM_MASK=${NUM_MASK:-3}
LR=${LR:-1e-4}
MIN_LR=${MIN_LR:-1e-6}
WARMUP=${WARMUP:-10000}
EMA_BASE=${EMA_BASE:-0.99}
EMA_FINAL=${EMA_FINAL:-0.9999}
FEA_MASK=${FEA_MASK:-0.20}
TIME_MASK=${TIME_MASK:-0.05}
NOISE_STD=(${NOISE_STD:-0.0 0.1})
TR_DROPOUT=${TR_DROPOUT:-0.0}
SIM_C=${SIM_C:-1.0}
VAR_C=${VAR_C:-1.0}
COV_C=${COV_C:-0.20}
NORM_C=${NORM_C:-0.05}
VAR_UP=${VAR_UP:-1.0}
SEEDS=${SEEDS:-42}

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
export WANDB_MODE

for ENV_ID in "${ENVS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    CKPT_DIR="results/sag/jepa/d4rl/${ENV_ID}/seed${SEED}"
    RUN_NAME="${ENV_ID}-seed${SEED}"

    ${PY} ${MAIN} \
      --env_id "${ENV_ID}" \
      --steps ${STEPS} \
      --batch_size ${BATCH} \
      --window ${WINDOW} \
      --k_max ${KMAX} \
      --num_mask ${NUM_MASK} \
      --lr ${LR} --min_lr ${MIN_LR} --warmup_steps ${WARMUP} \
      --ema_base ${EMA_BASE} --ema_final ${EMA_FINAL} \
      --feature_mask_ratio ${FEA_MASK} --time_mask_ratio ${TIME_MASK} \
      --dual_view_noise_std ${NOISE_STD} \
      --tr_dropout ${TR_DROPOUT} \
      --sim_coef ${SIM_C} --var_coef ${VAR_C} --cov_coef ${COV_C} \
      --norm_coef ${NORM_C} --var_upper ${VAR_UP} \
      --use_mask_token \
      --ckpt_dir "${CKPT_DIR}" \
      --seed ${SEED} \
      --wandb_project "${PROJECT}" \
      --wandb_run "${RUN_NAME}" \
      --wandb_group "${GROUP}" \
      --wandb_mode "${WANDB_MODE}" \
      --wandb_tags d4rl s-jepa
  done
done
