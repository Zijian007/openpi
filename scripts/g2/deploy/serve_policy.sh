#!/usr/bin/env bash
# YAML owns defaults; CLI flags override. CUDA is set in serve_policy.py before JAX loads.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"
cd "${G2_OPENPI_PROJECT_ROOT}"
exec uv run --extra g2 scripts/g2/deploy/serve_policy.py "$@"
