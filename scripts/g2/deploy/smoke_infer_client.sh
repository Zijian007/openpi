#!/usr/bin/env bash
# Fake-obs smoke against a running OpenPI serve_policy (GPU-local).
#
# Terminal A:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/g2/deploy/serve_policy.sh \
#     --config-name pi05_g2_vr_low_mem \
#     --policy-dir checkpoints/pi05_g2_vr_low_mem/<exp>/<step>
#
# Terminal B:
#   bash scripts/g2/deploy/smoke_infer_client.sh
#   bash scripts/g2/deploy/smoke_infer_client.sh --port 8001 --zeros --num-infer 1
#
# Not for wholebody / G2_pi LeRobot protocol.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/g2/_env.sh"

########################################
# CONFIG
########################################

HOST="127.0.0.1"
PORT=8000
PROMPT="Pick up the drink and put it in the box."
NUM_INFER=3
ZEROS=false
CONNECT_TIMEOUT_S=60

########################################
# CLI
########################################

usage() {
  cat <<EOF
Usage: bash scripts/g2/deploy/smoke_infer_client.sh [options]

  --host HOST              default ${HOST}
  --port PORT              default ${PORT} (match serve_policy --port)
  --prompt TEXT            default from training task string
  --num-infer N            default ${NUM_INFER}
  --zeros                  send zero obs (deterministic)
  --connect-timeout-s N    default ${CONNECT_TIMEOUT_S}
  -h | --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --prompt) PROMPT="$2"; shift 2 ;;
    --num-infer) NUM_INFER="$2"; shift 2 ;;
    --zeros) ZEROS=true; shift ;;
    --connect-timeout-s) CONNECT_TIMEOUT_S="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

cd "${PROJECT_ROOT}"

if [[ ! -d "${PROJECT_ROOT}/.venv" ]]; then
  echo "ERROR: missing .venv — run bash scripts/g2/bootstrap.sh" >&2
  exit 1
fi

echo "========================================"
echo "smoke_infer_client"
echo "uri:    ws://${HOST}:${PORT}"
echo "prompt: ${PROMPT}"
echo "infer:  ${NUM_INFER}  zeros=${ZEROS}"
echo "========================================"

ARGS=(
  scripts/g2/deploy/smoke_infer_client.py
  --host="${HOST}"
  --port="${PORT}"
  --prompt="${PROMPT}"
  --num-infer="${NUM_INFER}"
  --connect-timeout-s="${CONNECT_TIMEOUT_S}"
)
[[ "${ZEROS}" == true ]] && ARGS+=(--zeros)

uv run "${ARGS[@]}"
