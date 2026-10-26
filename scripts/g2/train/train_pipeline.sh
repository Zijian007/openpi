#!/usr/bin/env bash
# Train G2 using config.yaml beside this script.
# Edit training parameters in config.yaml, then run this script without arguments.
set -euo pipefail

if [[ $# -ne 0 ]]; then
  echo "ERROR: this launcher takes no arguments; edit scripts/g2/train/config.yaml." >&2
  echo "For previews or CLI overrides, run .venv/bin/python scripts/g2/train/train_pipeline.py directly." >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIG_FILE="${PROJECT_ROOT}/scripts/g2/train/config.yaml"

# Shared HF / model caches; Python reads dataset and proxy settings from YAML.
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/g2/_env.sh"
cd "${PROJECT_ROOT}"

if [[ ! -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  echo "ERROR: missing ${PROJECT_ROOT}/.venv — run bash scripts/g2/bootstrap.sh --no-proxy" >&2
  exit 1
fi

exec "${PROJECT_ROOT}/.venv/bin/python" scripts/g2/train/train_pipeline.py \
  --config "${CONFIG_FILE}"
