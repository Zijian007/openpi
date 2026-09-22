#!/usr/bin/env bash
# One-shot: compute_norm_stats → train.
# 所有常用训练参数在下面 CONFIG 改（对齐 G2_pi/train_full.sh 风格）。
# CLI 可选覆盖；更细项也可直接改 src/openpi/training/config.py。
#
#   CUDA_VISIBLE_DEVICES=0 bash scripts/g2/train/train_pipeline.sh
#   CUDA_VISIBLE_DEVICES=0 bash scripts/g2/train/train_pipeline.sh --smoke
#   tmux new -s g2-openpi 'cd ~/work/openpi && CUDA_VISIBLE_DEVICES=0 bash scripts/g2/train/train_pipeline.sh'

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/g2/_env.sh"

########################################
# CONFIG — 改这里（日常只动本块即可）
########################################
#
# 数据：  $HF_LEROBOT_HOME/$REPO_ID/   （_env.sh 默认 HF_LEROBOT_HOME=仓库/data）
# ckpt：  $CHECKPOINT_BASE_DIR/$CONFIG_NAME/$EXP_NAME/<step>/
# norm：  $ASSETS_BASE_DIR/$CONFIG_NAME/$REPO_ID/norm_stats.json
# 权重缓存：OPENPI_DATA_HOME=~/.cache/openpi（pi05_base；勿与 data 混）
# 结构细节（delta / LoRA variant）：src/openpi/training/config.py
#

# ---- 模式 / 实验 ----
# MODE:
#   full  → CONFIG_NAME 默认 pi05_g2_vr（全参微调）
#   smoke → CONFIG_NAME 默认 pi05_g2_vr_low_mem（JAX LoRA，省显存；短跑验证）
MODE="full"
# 非空则强制指定 TrainConfig 名，覆盖 MODE 默认（一般保持空）
CONFIG_NAME="pi05_g2_vr_low_mem"
# 实验名：同时用作
#   1) checkpoint 子目录名
#   2) W&B run 名（wandb.init(name=exp_name)）
# 空 = 自动：g2_vr_pi05_YYYYMMDD_HHMMSS 或 g2_vr_smoke_...
EXP_NAME=""
# LeRobot 本地数据集目录名（不是必须上 HF）。实际路径：
#   ~/work/openpi/data/$REPO_ID/{meta,data,videos}/
# 改数据文件夹名时这里一起改；须与 config.py 里 repo_id 一致（或靠 CLI 覆盖）。
REPO_ID="g2_vr_lerobot_v21"

# ---- norm stats（state/action 归一化统计）----
# SKIP_NORM:
#   ""     → 若已有 norm_stats.json 则自动跳过（推荐）
#   true   → 强制跳过
#   false  → 强制重算
# FORCE_NORM=true → 无视已有文件，强制重算（会设 SKIP_NORM=false）
SKIP_NORM=""
FORCE_NORM=false
# MAX_FRAMES：传给 compute_norm_stats.py --max-frames
#   ""     → 全量扫数据集（准，但 3 路视频 IO 很慢，数十分钟～1h+ 常见）
#   8000   → 约随机采样 8000 帧再算（几分钟级；日常够用）
# 仅影响 norm 这一步，不影响后续训练步数。
MAX_FRAMES=""

# ---- GPU / 代理 ----
# 多卡：如 "0,1"；与 G2_pi 错开卡号。JAX 用可见设备，FSDP 见 FSDP_DEVICES。
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# JAX 预留显存比例（OOM 可降到 0.8）
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
# true → socks5h://127.0.0.1:1080（拉权重/HF）；纯离线可 false
USE_PROXY=true

# ---- 路径（相对仓库根，或绝对路径；→ TrainConfig）----
# checkpoints/<CONFIG_NAME>/<EXP_NAME>/
CHECKPOINT_BASE_DIR="./checkpoints"
# assets/<CONFIG_NAME>/<REPO_ID>/norm_stats.json
ASSETS_BASE_DIR="./assets"

# ---- 训练超参 ----
# 空字符串 = 不覆盖 CLI，用 config.py 里该 TrainConfig 的默认值。
# 下列「默认」指 G2 两个配置；改完需重新开跑才生效（当前已在训的进程读不到）。
#
# BATCH_SIZE：全局 batch（每步采样条数；单卡时 = 每卡 batch）
#   pi05_g2_vr（全参）默认 8 —— 单 4090(24GB) 通常 OOM（XLA 估 ~50GiB），需多卡 FSDP
#   pi05_g2_vr_low_mem（LoRA）默认 2 —— 单 4090 可跑（峰值约 ~17GiB）；OOM 再降到 1
#   MODE=smoke 且此处为空时，脚本强制 1（短跑验证）
#   加大吞吐就加这个；显存不够优先降它，而不是先动 FSDP
BATCH_SIZE="8"
#
# NUM_WORKERS：Torch DataLoader 预取进程数（默认 2）
#   增大可减轻 GPU 等数据，但吃 CPU/主机内存；视频解码重时可试 4～8
#   设 0 = 主进程加载（调试方便，通常更慢）
NUM_WORKERS=""
#
# NUM_TRAIN_STEPS：总优化步数（一个 step = 一个 batch 的一次更新）
#   两个 G2 配置默认都是 30000；MODE=smoke 且此处为空时脚本强制 20
#   与 epoch 无关：数据会循环；想「多看几遍数据」就加大步数或加大 BATCH_SIZE
NUM_TRAIN_STEPS=""
#
# LOG_INTERVAL：每隔多少 step 打一次训练指标
#   默认 100 → 终端打印 Step N: loss=...，并 wandb.log（区间内 mean）
#   另：启动时 step=0 会额外 log 一张 camera_views
#   想曲线更密可设 25/50；过密会刷屏、略增开销
LOG_INTERVAL=""
#
# SAVE_INTERVAL：每隔多少 step 写一次 checkpoint
#   默认 1000；最后一步（num_train_steps-1）也会存
#   路径：checkpoints/<CONFIG_NAME>/<EXP_NAME>/<step>/
#   全参/LoRA ckpt 都很大（数 GB～十余 GB），太密会占满磁盘
SAVE_INTERVAL=""
#
# KEEP_PERIOD：ckpt 保留策略（配合 Orbax CheckpointManager）
#   默认 5000：step % 5000 == 0 的旧 ckpt 会被保留，其它中间步可能被清理
#   "none" → 传 --keep-period=None，关闭该「按周期保留」规则
#   空 = 用配置默认 3000
KEEP_PERIOD=""
#
# SEED：随机种子（默认 42）；影响数据 shuffle / 初始化等可复现性
SEED=""
#
# FSDP_DEVICES：模型分片设备数（Fully Sharded Data Parallel）
#   默认 1 = 不分片；>1 时把模型切到 N 张可见卡上，省单卡显存、可能变慢
#   例：CUDA_VISIBLE_DEVICES=0,1 且 FSDP_DEVICES=2 → 两卡切模型
#   须 ≤ 可见 GPU 数；单卡全参 OOM 时优先改 LoRA，多卡再开 FSDP
#   与 BATCH_SIZE：FSDP 后仍是「全局 batch」，按卡数切分数据并行组
FSDP_DEVICES=""

# ---- checkpoint 行为 ----
# OVERWRITE：同名 EXP 目录已存在则清空重开（不能与 RESUME 同时 true）
OVERWRITE=true
# RESUME：从该 EXP 目录最新 ckpt 继续（会读 wandb_id.txt 续 W&B）
RESUME=false

# ---- W&B ----
# WANDB_ENABLED:
#   ""     → smoke 默认关 / full 默认开
#   true|false → 显式开关
# 无独立「run 名」字段：run 名 = EXP_NAME
# 本地一般不写 train.log；需要时：... 2>&1 | tee logs/xxx.log
# 离线落盘：export WANDB_MODE=offline → ./wandb/offline-run-*
WANDB_ENABLED=""
WANDB_PROJECT="G2_openpi"  # → --project-name

########################################
# CLI（可选覆盖 CONFIG）
########################################

usage() {
  cat <<EOF
Usage: bash scripts/g2/train/train_pipeline.sh [options]

  Prefer editing the CONFIG block at the top of this script.
  CLI overrides:

  --smoke | --full
  --skip-norm | --force-norm
  --max-frames N
  --config-name NAME | --exp-name NAME | --repo-id ID
  --batch-size N | --num-train-steps N | --save-interval N
  --checkpoint-base-dir DIR | --project-name NAME
  --resume | --no-overwrite
  --no-proxy | --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) MODE="smoke"; shift ;;
    --full) MODE="full"; shift ;;
    --skip-norm) SKIP_NORM=true; shift ;;
    --force-norm) FORCE_NORM=true; SKIP_NORM=false; shift ;;
    --max-frames) MAX_FRAMES="$2"; shift 2 ;;
    --config-name) CONFIG_NAME="$2"; shift 2 ;;
    --exp-name) EXP_NAME="$2"; shift 2 ;;
    --repo-id) REPO_ID="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --num-train-steps) NUM_TRAIN_STEPS="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --log-interval) LOG_INTERVAL="$2"; shift 2 ;;
    --save-interval) SAVE_INTERVAL="$2"; shift 2 ;;
    --keep-period) KEEP_PERIOD="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --fsdp-devices) FSDP_DEVICES="$2"; shift 2 ;;
    --checkpoint-base-dir) CHECKPOINT_BASE_DIR="$2"; shift 2 ;;
    --assets-base-dir) ASSETS_BASE_DIR="$2"; shift 2 ;;
    --project-name) WANDB_PROJECT="$2"; shift 2 ;;
    --resume) RESUME=true; OVERWRITE=false; shift ;;
    --no-overwrite) OVERWRITE=false; shift ;;
    --wandb) WANDB_ENABLED=true; shift ;;
    --no-wandb) WANDB_ENABLED=false; shift ;;
    --no-proxy) USE_PROXY=false; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "${CONFIG_NAME}" ]]; then
  if [[ "${MODE}" == "smoke" ]]; then
    CONFIG_NAME="pi05_g2_vr_low_mem"
  else
    CONFIG_NAME="pi05_g2_vr"
  fi
fi

if [[ -z "${EXP_NAME}" ]]; then
  if [[ "${MODE}" == "smoke" ]]; then
    EXP_NAME="g2_vr_smoke_$(date +%Y%m%d_%H%M%S)"
  else
    EXP_NAME="g2_vr_pi05_$(date +%Y%m%d_%H%M%S)"
  fi
fi

if [[ -z "${WANDB_ENABLED}" ]]; then
  if [[ "${MODE}" == "smoke" ]]; then
    WANDB_ENABLED=false
  else
    WANDB_ENABLED=true
  fi
fi

# smoke 未显式设步数/batch 时给短跑默认
if [[ "${MODE}" == "smoke" ]]; then
  [[ -z "${NUM_TRAIN_STEPS}" ]] && NUM_TRAIN_STEPS=20
  [[ -z "${BATCH_SIZE}" ]] && BATCH_SIZE=1
fi

if [[ "${RESUME}" == true && "${OVERWRITE}" == true ]]; then
  echo "ERROR: RESUME and OVERWRITE both true" >&2
  exit 1
fi

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

_EP_STATS="${HF_LEROBOT_HOME}/${REPO_ID}/meta/episodes_stats.jsonl"
if [[ ! -f "${_EP_STATS}" ]]; then
  echo "ERROR: missing ${_EP_STATS}" >&2
  echo "  OpenPI LeRobot 0.1 requires episodes_stats.jsonl for codebase_version v2.1." >&2
  exit 1
fi

_NORM_STATS="${PROJECT_ROOT}/${ASSETS_BASE_DIR#./}/${CONFIG_NAME}/${REPO_ID}/norm_stats.json"
# Resolve relative assets path for existence check
if [[ "${ASSETS_BASE_DIR}" = /* ]]; then
  _NORM_STATS="${ASSETS_BASE_DIR}/${CONFIG_NAME}/${REPO_ID}/norm_stats.json"
else
  _NORM_STATS="${PROJECT_ROOT}/${ASSETS_BASE_DIR}/${CONFIG_NAME}/${REPO_ID}/norm_stats.json"
fi

if [[ "${FORCE_NORM}" == true ]]; then
  SKIP_NORM=false
elif [[ -z "${SKIP_NORM}" && -f "${_NORM_STATS}" ]]; then
  SKIP_NORM=true
elif [[ -z "${SKIP_NORM}" ]]; then
  SKIP_NORM=false
fi

if [[ "${CHECKPOINT_BASE_DIR}" = /* ]]; then
  _CKPT_DIR="${CHECKPOINT_BASE_DIR}/${CONFIG_NAME}/${EXP_NAME}"
else
  _CKPT_DIR="${PROJECT_ROOT}/${CHECKPOINT_BASE_DIR}/${CONFIG_NAME}/${EXP_NAME}"
fi

echo "========================================"
echo "pipeline:   norm_stats -> train (${MODE})"
echo "project:    ${PROJECT_ROOT}"
echo "config:     ${CONFIG_NAME}"
echo "exp:        ${EXP_NAME}"
echo "data:       ${HF_LEROBOT_HOME}/${REPO_ID}"
echo "hf_cache:   ${HF_DATASETS_CACHE}"
echo "openpi:     ${OPENPI_DATA_HOME}"
echo "skip_norm:  ${SKIP_NORM}"
echo "max_frames: ${MAX_FRAMES:-<full>}"
echo "ckpt_dir:   ${_CKPT_DIR}"
echo "batch/steps:${BATCH_SIZE:-<cfg>} / ${NUM_TRAIN_STEPS:-<cfg>}"
echo "save/log:   ${SAVE_INTERVAL:-<cfg>} / ${LOG_INTERVAL:-<cfg>}"
echo "wandb:      ${WANDB_ENABLED}  project=${WANDB_PROJECT}"
echo "overwrite:  ${OVERWRITE}  resume=${RESUME}"
echo "GPU:        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "========================================"

if [[ "${SKIP_NORM}" != true ]]; then
  echo ""
  echo "[1/2] compute_norm_stats ..."
  g2_openpi_setup_hf_cache
  NORM_ARGS=(scripts/compute_norm_stats.py --config-name "${CONFIG_NAME}")
  [[ -n "${MAX_FRAMES}" ]] && NORM_ARGS+=(--max-frames="${MAX_FRAMES}")
  uv run "${NORM_ARGS[@]}"
else
  echo ""
  echo "[1/2] compute_norm_stats skipped (reuse ${_NORM_STATS})"
fi

echo ""
echo "[2/2] train ..."
g2_openpi_setup_hf_cache

ARGS=(
  scripts/train.py "${CONFIG_NAME}"
  --exp-name="${EXP_NAME}"
  --checkpoint-base-dir="${CHECKPOINT_BASE_DIR}"
  --assets-base-dir="${ASSETS_BASE_DIR}"
  --project-name="${WANDB_PROJECT}"
)

[[ -n "${BATCH_SIZE}" ]] && ARGS+=(--batch-size="${BATCH_SIZE}")
[[ -n "${NUM_WORKERS}" ]] && ARGS+=(--num-workers="${NUM_WORKERS}")
[[ -n "${NUM_TRAIN_STEPS}" ]] && ARGS+=(--num-train-steps="${NUM_TRAIN_STEPS}")
[[ -n "${LOG_INTERVAL}" ]] && ARGS+=(--log-interval="${LOG_INTERVAL}")
[[ -n "${SAVE_INTERVAL}" ]] && ARGS+=(--save-interval="${SAVE_INTERVAL}")
if [[ -n "${KEEP_PERIOD}" ]]; then
  if [[ "${KEEP_PERIOD}" == "none" || "${KEEP_PERIOD}" == "None" ]]; then
    ARGS+=(--keep-period=None)
  else
    ARGS+=(--keep-period="${KEEP_PERIOD}")
  fi
fi
[[ -n "${SEED}" ]] && ARGS+=(--seed="${SEED}")
[[ -n "${FSDP_DEVICES}" ]] && ARGS+=(--fsdp-devices="${FSDP_DEVICES}")

if [[ "${OVERWRITE}" == true ]]; then
  ARGS+=(--overwrite)
else
  ARGS+=(--no-overwrite)
fi
if [[ "${RESUME}" == true ]]; then
  ARGS+=(--resume)
else
  ARGS+=(--no-resume)
fi
if [[ "${WANDB_ENABLED}" == true || "${WANDB_ENABLED}" == "1" ]]; then
  ARGS+=(--wandb-enabled)
else
  ARGS+=(--no-wandb-enabled)
fi

uv run "${ARGS[@]}"

echo ""
echo "done. checkpoints: ${_CKPT_DIR}"
