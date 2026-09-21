#!/usr/bin/env bash
# G2 OpenPI smoke: few steps on pi05_g2_vr_low_mem (LoRA).
#
#   bash scripts/g2/train_smoke.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/g2/_env.sh"

########################################
# CONFIG
########################################

CONFIG_NAME="pi05_g2_vr_low_mem"
EXP_NAME="g2_vr_smoke_$(date +%Y%m%d_%H%M%S)"
REPO_ID="g2_vr_lerobot_v21"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"
USE_PROXY=true
OVERWRITE=true

########################################
# RUN
########################################

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES
export XLA_PYTHON_CLIENT_MEM_FRACTION
[[ "${USE_PROXY}" == true ]] && g2_openpi_setup_proxy || g2_openpi_clear_proxy

g2_openpi_require_dataset "${REPO_ID}" || exit 1

echo "========================================"
echo "mode:     OpenPI smoke (LoRA low_mem)"
echo "project:  ${PROJECT_ROOT}"
echo "config:   ${CONFIG_NAME}"
echo "exp:      ${EXP_NAME}"
echo "data:     ${HF_LEROBOT_HOME}/${REPO_ID}"
echo "GPU:      CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "========================================"

ARGS=(scripts/train.py "${CONFIG_NAME}" --exp-name="${EXP_NAME}")
if [[ "${OVERWRITE}" == true ]]; then
  ARGS+=(--overwrite)
fi

# Short run via tyro overrides if supported; otherwise user stops early.
# OpenPI TrainConfig exposes num_train_steps as overridable CLI field.
ARGS+=(--num-train-steps=20 --batch-size=1 --no-wandb-enabled)

uv run "${ARGS[@]}"
