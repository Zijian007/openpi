#!/usr/bin/env bash
# Short LoRA smoke run. Skips norm if stats already exist (--norm auto).
#
#   bash scripts/g2/train/train_smoke.sh

set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/train_pipeline.sh" --smoke "$@"
