#!/usr/bin/env bash
# Full fine-tune preset (pi05_g2_vr). Prefer train_pipeline.sh for daily use.
#
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/g2/train/train_full.sh

set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/train_pipeline.sh" --preset full_ft "$@"
