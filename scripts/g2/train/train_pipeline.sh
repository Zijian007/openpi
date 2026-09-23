#!/usr/bin/env bash
# One-shot: compute_norm_stats → train.
# Daily knobs: CONFIG block below. Model / lr defaults: src/openpi/training/config.py.
#
#   CUDA_VISIBLE_DEVICES=0 bash scripts/g2/train/train_pipeline.sh
#   CUDA_VISIBLE_DEVICES=0 bash scripts/g2/train/train_pipeline.sh --smoke
#   CUDA_VISIBLE_DEVICES=0 bash scripts/g2/train/train_pipeline.sh --dry-run
#   CUDA_VISIBLE_DEVICES=0 bash scripts/g2/train/train_pipeline.sh -- --optimizer.peak-lr=1e-5
#
# Wrapper around official scripts/compute_norm_stats.py + scripts/train.py.

set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"

########################################
# CONFIG — 改这里（日常只动本块即可）
########################################
#
# 路径（_env.sh / 相对仓库根）:
#   data:  $HF_LEROBOT_HOME/$REPO_ID/
#   ckpt:  $CHECKPOINT_BASE_DIR/$CONFIG_NAME/$EXP_NAME/<step>/
#   norm:  $ASSETS_BASE_DIR/$CONFIG_NAME/$REPO_ID/norm_stats.json
# 模型结构、动作空间、action_horizon、discrete_state_input、学习率
# 都在 src/openpi/training/config.py 的 pi05_g2_vr*。
# 下面空字符串 = 不传 CLI，用那份 TrainConfig 的默认值。
#

# *********** 实验 ***********

# TrainConfig 名（src/openpi/training/config.py）。
#   pi05_g2_vr_low_mem  LoRA，单张 24GB
#   pi05_g2_vr          全参；单卡通常 OOM
# 短跑：--smoke（步数 20、batch 1、关 W&B）。全参也可以 --full。
CONFIG_NAME="pi05_g2_vr_low_mem"

# 实验名 = checkpoint 子目录 = W&B run 名。
# 空 = 自动 g2_vr_pi05_YYYYMMDD_HHMMSS（smoke 为 g2_vr_smoke_...）。
EXP_NAME=""

# 数据集目录名：$HF_LEROBOT_HOME/$REPO_ID/。会传给 --data.repo-id。
REPO_ID="g2_vr_lerobot_v21"

# *********** norm stats ***********

# state/action 归一化。推理从 ckpt 里的 assets 读，不读后来重算的这份。
#   auto  → 已有 norm_stats.json 就跳过
#   skip  → 强制跳过
#   force → 强制重算
NORM="auto"

# 只扫这么多帧再算 norm。空 = 全量（三路视频，可能几十分钟）。
MAX_FRAMES=""

# *********** GPU / 代理 ***********

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
USE_PROXY=true

# *********** 路径 ***********

CHECKPOINT_BASE_DIR="./checkpoints"
ASSETS_BASE_DIR="./assets"

# *********** 训练超参（空 = 用 config.py）***********

BATCH_SIZE="8"
NUM_WORKERS="4"
NUM_TRAIN_STEPS="30000"
LOG_INTERVAL="50"
SAVE_INTERVAL="5000"
KEEP_PERIOD=""
SEED=""
FSDP_DEVICES=""

# *********** checkpoint ***********

OVERWRITE=true
RESUME=false

# *********** W&B ***********

WANDB="auto"
WANDB_PROJECT="G2_openpi"

########################################
# 启动（CLI 参数会覆盖上面 CONFIG）
########################################

cd "${G2_OPENPI_PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES
export XLA_PYTHON_CLIENT_MEM_FRACTION

export G2_TRAIN_CONFIG_NAME="${CONFIG_NAME}"
export G2_TRAIN_EXP_NAME="${EXP_NAME}"
export G2_TRAIN_REPO_ID="${REPO_ID}"
export G2_TRAIN_NORM="${NORM}"
export G2_TRAIN_MAX_FRAMES="${MAX_FRAMES}"
export G2_TRAIN_CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR}"
export G2_TRAIN_ASSETS_BASE_DIR="${ASSETS_BASE_DIR}"
export G2_TRAIN_BATCH_SIZE="${BATCH_SIZE}"
export G2_TRAIN_NUM_WORKERS="${NUM_WORKERS}"
export G2_TRAIN_NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS}"
export G2_TRAIN_LOG_INTERVAL="${LOG_INTERVAL}"
export G2_TRAIN_SAVE_INTERVAL="${SAVE_INTERVAL}"
export G2_TRAIN_KEEP_PERIOD="${KEEP_PERIOD}"
export G2_TRAIN_SEED="${SEED}"
export G2_TRAIN_FSDP_DEVICES="${FSDP_DEVICES}"
export G2_TRAIN_OVERWRITE="${OVERWRITE}"
export G2_TRAIN_RESUME="${RESUME}"
export G2_TRAIN_WANDB="${WANDB}"
export G2_TRAIN_WANDB_PROJECT="${WANDB_PROJECT}"
export G2_TRAIN_USE_PROXY="${USE_PROXY}"

for _arg in "$@"; do
  case "${_arg}" in
    --no-proxy|--no-use-proxy) USE_PROXY=false ;;
    --use-proxy) USE_PROXY=true ;;
  esac
done
[[ "${USE_PROXY}" == true ]] && g2_openpi_setup_proxy || g2_openpi_clear_proxy
g2_openpi_require_dataset "${REPO_ID}" || exit 1

uv run scripts/g2/train/train_pipeline.py "$@"
