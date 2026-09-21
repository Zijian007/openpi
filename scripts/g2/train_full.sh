#!/usr/bin/env bash
# G2 OpenPI full fine-tune (default: pi05_g2_vr).
#
#   bash scripts/g2/train_full.sh
#   CUDA_VISIBLE_DEVICES=0 bash scripts/g2/train_full.sh
#   tmux new -s g2-openpi 'cd ~/work/openpi && bash scripts/g2/train_full.sh'
#
# Parallel to ~/work/G2_pi/scripts/train/train_full.sh — separate venv, data, GPUs.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/g2/_env.sh"

########################################
# CONFIG — 改这里
########################################

# Full FT: pi05_g2_vr | LoRA / 24GB-friendlier: pi05_g2_vr_low_mem
CONFIG_NAME="pi05_g2_vr"
EXP_NAME="g2_vr_pi05_$(date +%Y%m%d_%H%M%S)"
REPO_ID="g2_vr_lerobot_v21"

# Prefer a free GPU when G2_pi occupies others (e.g. 0).
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

# Optional tyro overrides (empty = use TrainConfig defaults)
BATCH_SIZE=""          # e.g. 4
NUM_TRAIN_STEPS=""     # e.g. 30000
OVERWRITE=true
WANDB_ENABLED=true
USE_PROXY=true

########################################
# RUN
########################################

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES
export XLA_PYTHON_CLIENT_MEM_FRACTION
[[ "${USE_PROXY}" == true ]] && g2_openpi_setup_proxy || g2_openpi_clear_proxy

g2_openpi_require_dataset "${REPO_ID}" || exit 1

if [[ ! -d "${PROJECT_ROOT}/.venv" ]]; then
  echo "ERROR: missing .venv — run bash scripts/g2/bootstrap.sh" >&2
  exit 1
fi

echo "========================================"
echo "mode:     OpenPI fine-tune"
echo "project:  ${PROJECT_ROOT}"
echo "config:   ${CONFIG_NAME}"
echo "exp:      ${EXP_NAME}"
echo "data:     ${HF_LEROBOT_HOME}/${REPO_ID}"
echo "ckpt →    ${PROJECT_ROOT}/checkpoints/${CONFIG_NAME}/${EXP_NAME}/"
echo "GPU:      CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "========================================"

ARGS=(scripts/train.py "${CONFIG_NAME}" --exp-name="${EXP_NAME}")
[[ "${OVERWRITE}" == true ]] && ARGS+=(--overwrite)
[[ -n "${BATCH_SIZE}" ]] && ARGS+=(--batch-size="${BATCH_SIZE}")
[[ -n "${NUM_TRAIN_STEPS}" ]] && ARGS+=(--num-train-steps="${NUM_TRAIN_STEPS}")
if [[ "${WANDB_ENABLED}" == true || "${WANDB_ENABLED}" == "1" ]]; then
  ARGS+=(--wandb-enabled)
else
  ARGS+=(--no-wandb-enabled)
fi

uv run "${ARGS[@]}"
