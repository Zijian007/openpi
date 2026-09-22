#!/usr/bin/env bash
# Serve an OpenPI G2 checkpoint over WebSocket (default port 8000).
#
#   bash scripts/g2/deploy/serve_policy.sh
#   bash scripts/g2/deploy/serve_policy.sh --policy-dir checkpoints/pi05_g2_vr/<exp>/<step>
#   bash scripts/g2/deploy/serve_policy.sh --port 8001 ...   # avoid clash with G2_pi :8000
#
# Smoke (another terminal):
#   bash scripts/g2/deploy/smoke_infer_client.sh --port 8000
#
# Observation keys must match g2_policy.G2Inputs / make_g2_example().
# Not compatible with G2_pi policy_server / wholebody lerobot_pi05_client without an adapter.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/g2/_env.sh"

########################################
# CONFIG
########################################

CONFIG_NAME="pi05_g2_vr_low_mem"
# Leave empty → newest experiment under checkpoints/<CONFIG_NAME>/, then its highest numeric step.
POLICY_DIR=""
PORT=8000
DEFAULT_PROMPT=""
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
USE_PROXY=false

########################################
# CLI
########################################

usage() {
  cat <<EOF
Usage: bash scripts/g2/deploy/serve_policy.sh [options]

  --policy-dir DIR | -d DIR   checkpoint step dir (…/<exp>/<step>)
  --config-name NAME          TrainConfig name (default ${CONFIG_NAME})
  --port PORT                 WebSocket port (default ${PORT}; use 8001 if LeRobot holds 8000)
  --default-prompt TEXT       injected when client omits prompt
  --proxy | --no-proxy
  -h | --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-dir|-d)
      POLICY_DIR="$2"; shift 2 ;;
    --config-name)
      CONFIG_NAME="$2"; shift 2 ;;
    --port)
      PORT="$2"; shift 2 ;;
    --default-prompt)
      DEFAULT_PROMPT="$2"; shift 2 ;;
    --no-proxy) USE_PROXY=false; shift ;;
    --proxy) USE_PROXY=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES
[[ "${USE_PROXY}" == true ]] && g2_openpi_setup_proxy || g2_openpi_clear_proxy

if [[ -z "${POLICY_DIR}" ]]; then
  # Newest experiment (by its latest step dir mtime), then highest numeric step in that exp.
  # Lexicographic sort is wrong: "5000" sorts after "29999".
  # Max step across all exps is also wrong: a new run at step 5000 would lose to an old 30000.
  _CKPT_ROOT="${PROJECT_ROOT}/checkpoints/${CONFIG_NAME}"
  if [[ -d "${_CKPT_ROOT}" ]]; then
    _NEWEST_STEP="$(
      find "${_CKPT_ROOT}" -mindepth 2 -maxdepth 2 -type d -printf '%T@\t%f\t%p\n' \
        | awk -F '\t' '$2 ~ /^[0-9]+$/ { print }' \
        | sort -n \
        | tail -1 \
        | cut -f3-
    )"
    if [[ -n "${_NEWEST_STEP}" ]]; then
      _EXP_DIR="$(dirname "${_NEWEST_STEP}")"
      POLICY_DIR="$(
        find "${_EXP_DIR}" -mindepth 1 -maxdepth 1 -type d -printf '%f\t%p\n' \
          | awk -F '\t' '$1 ~ /^[0-9]+$/ { print $1+0 "\t" $2 }' \
          | sort -n \
          | tail -1 \
          | cut -f2-
      )"
      echo "auto policy: newest exp $(basename "${_EXP_DIR}")"
    fi
  fi
fi

if [[ -z "${POLICY_DIR}" || ! -d "${POLICY_DIR}" ]]; then
  echo "ERROR: set --policy-dir to a checkpoint step directory" >&2
  exit 1
fi

# Resolve relative paths
[[ "${POLICY_DIR}" != /* ]] && POLICY_DIR="${PROJECT_ROOT}/${POLICY_DIR}"

if [[ ! -d "${POLICY_DIR}/params" && ! -f "${POLICY_DIR}/model.safetensors" ]]; then
  echo "ERROR: ${POLICY_DIR} looks incomplete (need params/ or model.safetensors)" >&2
  exit 1
fi

echo "========================================"
echo "serve_policy  (OpenPI line ≠ G2_pi)"
echo "config: ${CONFIG_NAME}"
echo "dir:    ${POLICY_DIR}"
echo "port:   ${PORT}"
echo "GPU:    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "smoke:  bash scripts/g2/deploy/smoke_infer_client.sh --port ${PORT}"
echo "========================================"

ARGS=(
  scripts/serve_policy.py
  --port="${PORT}"
)
[[ -n "${DEFAULT_PROMPT}" ]] && ARGS+=(--default-prompt="${DEFAULT_PROMPT}")
ARGS+=(
  policy:checkpoint
  --policy.config="${CONFIG_NAME}"
  --policy.dir="${POLICY_DIR}"
)

uv run "${ARGS[@]}"
