#!/usr/bin/env bash
# YAML owns defaults; CLI flags override. CUDA is set in serve_policy.py before JAX loads.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"
cd "${G2_OPENPI_PROJECT_ROOT}"
# This server always uses OpenPI's project environment, even from the robot venv.
exec env -u VIRTUAL_ENV uv run --extra g2 scripts/g2/deploy/serve_policy.py "$@"
