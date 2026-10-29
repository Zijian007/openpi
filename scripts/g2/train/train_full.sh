#!/usr/bin/env bash
# Full fine-tune using the default YAML, or select another with --config PATH.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"
g2_openpi_run_train --vision-mode full --language-mode full --action-expert-mode full "$@"
