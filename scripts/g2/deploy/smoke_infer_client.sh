#!/usr/bin/env bash
# Fake-obs smoke; --config PATH uses the same deployment YAML as serve_policy.sh.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"
cd "${G2_OPENPI_PROJECT_ROOT}"
exec uv run --extra g2 scripts/g2/deploy/smoke_infer_client.py "$@"
