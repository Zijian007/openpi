# Shared path / HF caches for G2 OpenPI scripts.
# Parallel to lerobot-finetune/scripts/_env.sh — do NOT source lerobot-finetune's env.
# shellcheck shell=bash

G2_OPENPI_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Dataset root for OpenPI's pinned LeRobot 0.1: looks up $HF_LEROBOT_HOME/<repo_id>
# Keep separate from lerobot-finetune's v3 conversion cache (.cache/datasets/v3).
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${G2_OPENPI_PROJECT_ROOT}/data}"

# OpenPI checkpoint / tokenizer cache (pi05_base, paligemma tokenizer, …).
# MUST NOT be the LeRobot dataset dir — otherwise train re-downloads weights into data/.
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${HOME}/.cache/openpi}"
mkdir -p "${OPENPI_DATA_HOME}"

# Persist JAX XLA compiles across serve restarts (first-request JIT ~12s otherwise).
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${OPENPI_DATA_HOME}/jax_compile_cache}"
mkdir -p "${JAX_COMPILATION_CACHE_DIR}"

# Lab /etc/profile.d/huggingface.sh sets HF_* → /opt/huggingface_cache (not writable).
export HF_HOME="${HOME}/.cache/openpi-hf"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}"
unset HF_ENDPOINT || true

g2_openpi_run_train() {
  cd "${G2_OPENPI_PROJECT_ROOT}"
  if [[ ! -x .venv/bin/python ]]; then
    echo "ERROR: missing ${G2_OPENPI_PROJECT_ROOT}/.venv — run bash scripts/g2/bootstrap.sh" >&2
    exit 1
  fi
  # train_pipeline.sh sets its editable YAML path; explicit CLI arguments take precedence.
  exec .venv/bin/python scripts/g2/train/train_pipeline.py \
    --config "${G2_TRAIN_CONFIG:-scripts/g2/configs/train.yaml}" "$@"
}
