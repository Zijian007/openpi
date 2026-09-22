#!/usr/bin/env bash
# Compute norm stats only (step 1 of train_pipeline.sh).
#
#   bash scripts/g2/train/compute_norm_stats.sh
#   bash scripts/g2/train/compute_norm_stats.sh --config-name pi05_g2_vr_low_mem

set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/train_pipeline.sh" --norm-only "$@"
