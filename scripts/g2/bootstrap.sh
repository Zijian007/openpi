#!/usr/bin/env bash
# uv sync --extra g2, optionally download pi05_base into the OpenPI cache.
#
#   bash scripts/g2/bootstrap.sh
#   bash scripts/g2/bootstrap.sh --download-base
#
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"
cd "${G2_OPENPI_PROJECT_ROOT}"

DOWNLOAD_BASE=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --download-base) DOWNLOAD_BASE=true; shift ;;
    -h|--help)
      echo "Usage: bash scripts/g2/bootstrap.sh [--download-base]"
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv not found. Install: https://docs.astral.sh/uv/" >&2
  exit 1
fi

echo "[bootstrap] GIT_LFS_SKIP_SMUDGE=1 uv sync --extra g2"
sync_args=(--extra g2)
if [[ -n "${G2_PROTOCOL_WHEELS:-}" ]]; then
  sync_args+=(--no-sources-package g2-policy-protocol --find-links "${G2_PROTOCOL_WHEELS}")
fi
GIT_LFS_SKIP_SMUDGE=1 uv sync "${sync_args[@]}"
mkdir -p "${HF_LEROBOT_HOME}"
echo "[bootstrap] HF_LEROBOT_HOME=${HF_LEROBOT_HOME}"

if [[ "${DOWNLOAD_BASE}" == true ]]; then
  echo "[bootstrap] downloading pi05_base ..."
  uv run --extra g2 python -c "
from openpi.shared import download
print('pi05_base ->', download.maybe_download('gs://openpi-assets/checkpoints/pi05_base'))
"
fi

echo "[bootstrap] done. Next:"
echo "  1) rsync lerobot v2.1 → \${HF_LEROBOT_HOME}/<data.repo_id>/"
echo "  2) bash scripts/g2/train/compute_norm_stats.sh"
echo "  3) bash scripts/g2/train/train_smoke.sh"
