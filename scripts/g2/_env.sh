# Shared path / proxy / HF caches for G2 OpenPI scripts.
# Parallel to ~/work/G2_pi/scripts/_env.sh — do NOT source G2_pi's env.
# shellcheck shell=bash

_G2_OPENPI_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Public project root for all scripts/g2/* entrypoints.
G2_OPENPI_PROJECT_ROOT="${_G2_OPENPI_ENV_DIR}"

# Dataset root for OpenPI's pinned LeRobot 0.1: looks up $HF_LEROBOT_HOME/<repo_id>
# Keep separate from G2_pi's ~/.cache/huggingface/lerobot (v3).
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${_G2_OPENPI_ENV_DIR}/data}"

# OpenPI checkpoint / tokenizer cache (pi05_base, paligemma tokenizer, …).
# MUST NOT be the LeRobot dataset dir — otherwise train re-downloads weights into data/.
# Default matches openpi.shared.download.DEFAULT_CACHE_DIR (~/.cache/openpi).
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${HOME}/.cache/openpi}"
mkdir -p "${OPENPI_DATA_HOME}"

# Persist JAX XLA compiles across serve restarts (first-request JIT ~12s otherwise).
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${OPENPI_DATA_HOME}/jax_compile_cache}"
mkdir -p "${JAX_COMPILATION_CACHE_DIR}"

# Lab /etc/profile.d/huggingface.sh sets HF_* → /opt/huggingface_cache (not writable).
# Always force override — do not inherit those paths.
g2_openpi_setup_hf_cache() {
  export HF_HOME="${HOME}/.cache/openpi-hf"
  export HF_DATASETS_CACHE="${HF_HOME}/datasets"
  export HF_HUB_CACHE="${HF_HOME}/hub"
  export TRANSFORMERS_CACHE="${HF_HOME}/hub"
  mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}"
  # Optional lab HF mirror can break / hang; clear unless user re-exports.
  unset HF_ENDPOINT || true
}

g2_openpi_setup_hf_cache

# Default proxy for lab (bootstrap / weight download). Train scripts may re-export.
G2_OPENPI_DEFAULT_PROXY="${G2_OPENPI_DEFAULT_PROXY:-socks5h://127.0.0.1:1080}"

g2_openpi_setup_proxy() {
  local proxy="${1:-${G2_OPENPI_DEFAULT_PROXY}}"
  export HTTPS_PROXY="${proxy}"
  export HTTP_PROXY="${proxy}"
  export ALL_PROXY="${proxy}"
}

g2_openpi_clear_proxy() {
  unset HTTPS_PROXY HTTP_PROXY ALL_PROXY || true
}

g2_openpi_require_dataset() {
  local repo_id="${1:-g2_vr_lerobot_v21}"
  local root="${HF_LEROBOT_HOME}/${repo_id}"
  if [[ ! -f "${root}/meta/info.json" ]]; then
    echo "ERROR: LeRobot v2.1 dataset not found: ${root}" >&2
    echo "  Expected: ${root}/{meta,data,videos}/" >&2
    echo "  Rsync from G2 recorded/.../lerobot_v21/ then set repo_id=${repo_id}" >&2
    return 1
  fi
  if ! grep -q '"codebase_version"[[:space:]]*:[[:space:]]*"v2.1"' "${root}/meta/info.json" \
    && ! grep -q '"codebase_version": "v2.1"' "${root}/meta/info.json"; then
    echo "WARN: ${root}/meta/info.json does not look like v2.1 — OpenPI may fail to load" >&2
  fi
  if [[ ! -f "${root}/meta/episodes_stats.jsonl" ]]; then
    echo "ERROR: missing ${root}/meta/episodes_stats.jsonl" >&2
    echo "  OpenPI LeRobot 0.1 requires episodes_stats.jsonl for codebase_version v2.1." >&2
    return 1
  fi
}
