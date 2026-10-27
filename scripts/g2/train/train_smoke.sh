#!/usr/bin/env bash
# Short run preserving the selected model and module modes; --config PATH selects YAML.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"
g2_openpi_run_train --smoke "$@"
