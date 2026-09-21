#!/usr/bin/env bash
# Compute norm stats for pi05_g2_vr (writes under ./assets/).
#
#   bash scripts/g2/compute_norm_stats.sh
#   bash scripts/g2/compute_norm_stats.sh --config-name pi05_g2_vr_low_mem

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/g2/_env.sh"

CONFIG_NAME="pi05_g2_vr"
REPO_ID="g2_vr_lerobot_v21"
USE_PROXY=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config-name) CONFIG_NAME="$2"; shift 2 ;;
    --repo-id) REPO_ID="$2"; shift 2 ;;
    --no-proxy) USE_PROXY=false; shift ;;
    -h|--help)
      echo "Usage: bash scripts/g2/compute_norm_stats.sh [--config-name NAME] [--repo-id ID] [--no-proxy]"
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

cd "${PROJECT_ROOT}"
[[ "${USE_PROXY}" == true ]] && g2_openpi_setup_proxy || g2_openpi_clear_proxy

g2_openpi_require_dataset "${REPO_ID}" || exit 1

echo "========================================"
echo "compute_norm_stats"
echo "config:  ${CONFIG_NAME}"
echo "data:    ${HF_LEROBOT_HOME}/${REPO_ID}"
echo "========================================"

uv run scripts/compute_norm_stats.py --config-name "${CONFIG_NAME}"
