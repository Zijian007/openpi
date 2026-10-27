#!/usr/bin/env bash
# Train using scripts/g2/train/config.yaml.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"
g2_openpi_no_launcher_args "$@"
g2_openpi_run_train
