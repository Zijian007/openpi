#!/usr/bin/env bash
# Serve an OpenPI G2 checkpoint over WebSocket (default port 8000).
#
#   bash scripts/g2/deploy/serve_policy.sh
#   bash scripts/g2/deploy/serve_policy.sh --policy-dir checkpoints/pi05_g2_vr/<exp>/<step>
#   bash scripts/g2/deploy/serve_policy.sh --port 8001 ...   # avoid clash with G2_pi :8000
#   bash scripts/g2/deploy/serve_policy.sh --no-warmup   # skip JIT dummy infer
#
# Smoke (another terminal; wait until warmup logs + healthz):
#   bash scripts/g2/deploy/smoke_infer_client.sh --port 8000
#
# Starts official scripts/serve_policy.py. Prompt is sent by the domain controller.
# Observation keys must match g2_policy.G2Inputs / make_g2_example().

set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"

########################################
# CONFIG — 改这里（日常只动本块即可）
########################################
#
# 加载的是 TrainConfig，不是只读权重目录。
# action_horizon、discrete_state_input、norm stats 都跟着 CONFIG_NAME 走。
# 必须和这份 checkpoint 训练时的配置一致。现在 G2 两份都是 horizon 50、state 打进 prompt。
#

# *********** 权重 ***********

# 哪份 TrainConfig。须与 checkpoint 目录上一级的配置名一致。
#   pi05_g2_vr_low_mem  LoRA
#   pi05_g2_vr          全参
CONFIG_NAME="pi05_g2_vr_low_mem"

# checkpoint 的 step 目录（…/<exp>/<step>），相对仓库根或绝对路径。
# 空 = 在 checkpoints/$CONFIG_NAME/ 里选最近写入的实验，再取该实验数值最大的 step。
POLICY_DIR="/home/zijianwang/work/openpi/checkpoints/pi05_g2_vr_low_mem/g2_vr_pi05_20260922_181334/29999"

CHECKPOINT_BASE_DIR="./checkpoints"

# *********** 服务 ***********

# WebSocket 端口。本机 smoke 默认 8000；域控真机用 8001（和 LeRobot :8000 错开）。
PORT=8000

# 监听前用 224 假图 infer 两次：step0 付 JAX JIT（约 10–15s），step1 应落到稳态 ~100ms。
# healthz 在 warmup 结束前不会 OK，所以首包不会再让真机付编译。排障可 --no-warmup。
WARMUP=true

# prompt 只由域控随 observation 传入，服务端不设默认任务句。

# *********** GPU / 代理 ***********

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
USE_PROXY=false

usage() {
  cat <<EOF
Usage: bash scripts/g2/deploy/serve_policy.sh [options]

  --policy-dir DIR | -d DIR   checkpoint step dir (…/<exp>/<step>)
  --config-name NAME          TrainConfig name (default ${CONFIG_NAME})
  --port PORT                 WebSocket port (default ${PORT}; use 8001 if LeRobot holds 8000)
  --warmup | --no-warmup      dummy infer before listen (default ${WARMUP})
  --proxy | --no-proxy
  -h | --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-dir|-d)
      POLICY_DIR="$2"
      shift 2
      ;;
    --config-name)
      CONFIG_NAME="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --warmup)
      WARMUP=true
      shift
      ;;
    --no-warmup)
      WARMUP=false
      shift
      ;;
    --checkpoint-base-dir)
      CHECKPOINT_BASE_DIR="$2"
      shift 2
      ;;
    --proxy|--use-proxy)
      USE_PROXY=true
      shift
      ;;
    --no-proxy|--no-use-proxy)
      USE_PROXY=false
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 1
      ;;
  esac
done

cd "${G2_OPENPI_PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES
[[ "${USE_PROXY}" == true ]] && g2_openpi_setup_proxy || g2_openpi_clear_proxy

if [[ ! -d "${G2_OPENPI_PROJECT_ROOT}/.venv" ]]; then
  echo "ERROR: missing ${G2_OPENPI_PROJECT_ROOT}/.venv — run bash scripts/g2/bootstrap.sh" >&2
  exit 1
fi

PICK_ARGS=(
  scripts/g2/deploy/pick_checkpoint.py
  --config-name="${CONFIG_NAME}"
  --checkpoint-base-dir="${CHECKPOINT_BASE_DIR}"
)
[[ -n "${POLICY_DIR}" ]] && PICK_ARGS+=(--policy-dir="${POLICY_DIR}")
POLICY_DIR="$(uv run "${PICK_ARGS[@]}")"

WARMUP_FLAG=--warmup
[[ "${WARMUP}" == true ]] || WARMUP_FLAG=--no-warmup

echo "========================================"
echo "serve_policy  (OpenPI line ≠ G2_pi)"
echo "config: ${CONFIG_NAME}"
echo "dir:    ${POLICY_DIR}"
echo "port:   ${PORT}"
echo "warmup: ${WARMUP}"
echo "jax_cache: ${JAX_COMPILATION_CACHE_DIR:-unset}"
echo "GPU:    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "smoke:  bash scripts/g2/deploy/smoke_infer_client.sh --port ${PORT}"
echo "========================================"

uv run scripts/serve_policy.py \
  --port="${PORT}" \
  "${WARMUP_FLAG}" \
  policy:checkpoint \
  --policy.config="${CONFIG_NAME}" \
  --policy.dir="${POLICY_DIR}"
