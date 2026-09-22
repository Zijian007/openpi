#!/usr/bin/env bash
# One-shot: compute_norm_stats → train.
# Daily knobs: CONFIG block below. Model / lr defaults: src/openpi/training/config.py.
#
#   CUDA_VISIBLE_DEVICES=0 bash scripts/g2/train/train_pipeline.sh
#   CUDA_VISIBLE_DEVICES=0 bash scripts/g2/train/train_pipeline.sh --smoke
#   CUDA_VISIBLE_DEVICES=0 bash scripts/g2/train/train_pipeline.sh --dry-run
#   CUDA_VISIBLE_DEVICES=0 bash scripts/g2/train/train_pipeline.sh -- --optimizer.peak-lr=1e-5

set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/g2/_env.sh"
# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/_train_common.sh"

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

g2_train_set_defaults

# *********** 实验 ***********

# TrainConfig 名（src/openpi/training/config.py）。
#   pi05_g2_vr_low_mem  LoRA，单张 24GB
#   pi05_g2_vr          全参；单卡通常 OOM
# 短跑：--smoke（步数 20、batch 1、关 W&B）。全参也可以 --full。
CONFIG_NAME="pi05_g2_vr_low_mem"

# 实验名 = checkpoint 子目录 = W&B run 名。
# 空 = 自动 g2_vr_pi05_YYYYMMDD_HHMMSS（smoke 为 g2_vr_smoke_...）。
# 想和上一次分开就留空，或写成 g2_vr_pi05_h50_state 这类固定名。
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

# 可见 GPU。多卡写成 "0,1"。与 G2_pi 错开卡号。
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# JAX 预留显存比例。OOM 可降到 0.8。
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

# true → socks5 代理（拉 pi05_base / HF）。纯离线设 false，或 CLI --no-proxy。
USE_PROXY=true

# *********** 路径 ***********

# checkpoint 根目录。相对仓库根，或绝对路径。
CHECKPOINT_BASE_DIR="./checkpoints"

# norm stats 根目录。相对仓库根，或绝对路径。
ASSETS_BASE_DIR="./assets"

# *********** 训练超参（空 = 用 config.py）***********
# G2 两份现在都是 horizon 50、state 打进 prompt。

# low_mem 默认 2，full 默认 8（全局 batch）。
BATCH_SIZE="8"

# DataLoader 进程数。config 默认 2；视频解码重可试 4～8；0 = 主进程加载。
NUM_WORKERS="4"

# 优化步数（一步 = 一个 batch）。两份 G2 默认 30000。smoke 预设会改成 20。
NUM_TRAIN_STEPS="30000"

# 每隔多少 step 打一次 loss / wandb。默认 100。
LOG_INTERVAL="50"

# 每隔多少 step 存一次 ckpt；最后一步也会存。默认 1000。
SAVE_INTERVAL="5000"

# 保留 step % N == 0 的旧 ckpt，其余中间步可能被清掉。默认 5000。
# "none" → 关掉这条保留规则。
KEEP_PERIOD=""

# 随机种子。默认 42。
SEED=""

# >1 时把模型切到多卡（FSDP）。须 ≤ 可见 GPU 数。默认 1。
FSDP_DEVICES=""

# *********** checkpoint ***********

# true：同名 EXP 目录已存在则清空重开。不会动别的实验名。
# 不能与 RESUME 同时为 true。
OVERWRITE=true

# true：从该 EXP 目录最新 ckpt 接着训，并续 W&B。
RESUME=false

# *********** W&B ***********

# auto = low_mem/full_ft 开，smoke 关。on|off 强制。
# run 名就是 EXP_NAME，没有单独的 job 名字段。
WANDB="auto"

# W&B 项目名。
WANDB_PROJECT="G2_openpi"

g2_train_parse_args "$@"
g2_train_run_pipeline
