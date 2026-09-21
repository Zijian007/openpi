#!/usr/bin/env bash
# Serve an OpenPI G2 checkpoint over WebSocket (default port 8000).
#
#   bash scripts/g2/serve_policy.sh
#   bash scripts/g2/serve_policy.sh --policy-dir checkpoints/pi05_g2_vr/<exp>/<step>
#
# Observation keys must match g2_policy.G2Inputs / make_g2_example().
# Not compatible with G2_pi policy_server / wholebody pi05_client without an adapter.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/g2/_env.sh"

########################################
# CONFIG
########################################

CONFIG_NAME="pi05_g2_vr"
# Leave empty to require --policy-dir
POLICY_DIR=""
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
USE_PROXY=false

########################################
# CLI
########################################

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-dir|-d)
      POLICY_DIR="$2"; shift 2 ;;
    --config-name)
      CONFIG_NAME="$2"; shift 2 ;;
    --no-proxy) USE_PROXY=false; shift ;;
    --proxy) USE_PROXY=true; shift ;;
    -h|--help)
      echo "Usage: bash scripts/g2/serve_policy.sh --policy-dir checkpoints/.../<step> [--config-name pi05_g2_vr]"
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES
[[ "${USE_PROXY}" == true ]] && g2_openpi_setup_proxy || g2_openpi_clear_proxy

if [[ -z "${POLICY_DIR}" ]]; then
  # Pick latest step under checkpoints/<config>/*/*/
  if [[ -d "${PROJECT_ROOT}/checkpoints/${CONFIG_NAME}" ]]; then
    POLICY_DIR="$(find "${PROJECT_ROOT}/checkpoints/${CONFIG_NAME}" -mindepth 2 -maxdepth 2 -type d | sort | tail -1 || true)"
  fi
fi

if [[ -z "${POLICY_DIR}" || ! -d "${POLICY_DIR}" ]]; then
  echo "ERROR: set --policy-dir to a checkpoint step directory" >&2
  exit 1
fi

# Resolve relative paths
[[ "${POLICY_DIR}" != /* ]] && POLICY_DIR="${PROJECT_ROOT}/${POLICY_DIR}"

echo "========================================"
echo "serve_policy"
echo "config: ${CONFIG_NAME}"
echo "dir:    ${POLICY_DIR}"
echo "GPU:    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "========================================"

uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config="${CONFIG_NAME}" \
  --policy.dir="${POLICY_DIR}"
