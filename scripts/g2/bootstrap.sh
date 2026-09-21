#!/usr/bin/env bash
# One-shot: uv sync + (optional) download pi05_base into OpenPI cache.
#
#   bash scripts/g2/bootstrap.sh
#   bash scripts/g2/bootstrap.sh --no-proxy
#   bash scripts/g2/bootstrap.sh --download-base
#
# Does NOT touch ~/work/G2_pi or /home/agi/.venv.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/g2/_env.sh"

USE_PROXY=true
DOWNLOAD_BASE=false

usage() {
  cat <<EOF
Usage: bash scripts/g2/bootstrap.sh [options]

  --no-proxy         Do not set HTTP(S)_PROXY (default: ${G2_OPENPI_DEFAULT_PROXY})
  --download-base    Also download gs://openpi-assets/checkpoints/pi05_base
  --help             Show this help

Project: ${PROJECT_ROOT}
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-proxy) USE_PROXY=false; shift ;;
    --download-base) DOWNLOAD_BASE=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

cd "${PROJECT_ROOT}"

if [[ "${USE_PROXY}" == true ]]; then
  g2_openpi_setup_proxy
  echo "[bootstrap] proxy: HTTPS_PROXY=${HTTPS_PROXY}"
else
  g2_openpi_clear_proxy
  echo "[bootstrap] proxy: disabled (--no-proxy)"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv not found. Install: https://docs.astral.sh/uv/" >&2
  exit 1
fi

if [[ ! -f "${PROJECT_ROOT}/.gitmodules" ]]; then
  echo "WARN: no .gitmodules — submodules may be incomplete" >&2
fi

echo "[bootstrap] git submodule update --init --recursive"
git submodule update --init --recursive

echo "[bootstrap] GIT_LFS_SKIP_SMUDGE=1 uv sync"
GIT_LFS_SKIP_SMUDGE=1 uv sync

mkdir -p "${HF_LEROBOT_HOME}"
echo "[bootstrap] HF_LEROBOT_HOME=${HF_LEROBOT_HOME}"
echo "[bootstrap] place v2.1 data at: ${HF_LEROBOT_HOME}/g2_vr_lerobot_v21/"

if [[ "${DOWNLOAD_BASE}" == true ]]; then
  echo "[bootstrap] downloading pi05_base ..."
  uv run python -c "
from openpi.shared import download
p = download.maybe_download('gs://openpi-assets/checkpoints/pi05_base')
print('pi05_base ->', p)
"
fi

echo "[bootstrap] done. Next:"
echo "  1) rsync lerobot_v21 → ${HF_LEROBOT_HOME}/g2_vr_lerobot_v21/"
echo "  2) bash scripts/g2/compute_norm_stats.sh"
echo "  3) bash scripts/g2/train_smoke.sh"
