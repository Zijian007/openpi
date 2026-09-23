#!/usr/bin/env bash
# Fake-obs smoke against a running OpenPI serve_policy (GPU-local).
#
#   bash scripts/g2/deploy/smoke_infer_client.sh
#   bash scripts/g2/deploy/smoke_infer_client.sh --port 8001 --zeros --num-infer 1
#
# Logic lives in scripts/g2/deploy/smoke_infer_client.py.

set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"

cd "${G2_OPENPI_PROJECT_ROOT}"
uv run scripts/g2/deploy/smoke_infer_client.py "$@"
