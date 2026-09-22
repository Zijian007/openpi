# Shared helpers for G2 OpenPI train scripts.
# shellcheck shell=bash

_G2_TRAIN_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
G2_TRAIN_PROJECT_ROOT="$(cd "${_G2_TRAIN_COMMON_DIR}/../../.." && pwd)"

# Populated by g2_train_parse_args / g2_train_build_train_args.
G2_TRAIN_PASSTHRU=()
G2_NORM_ARGS=()
G2_TRAIN_ARGS=()

g2_train_set_defaults() {
  # Preset bundles defaults when CONFIG_NAME is empty:
  #   low_mem  → pi05_g2_vr_low_mem (single-GPU friendly)
  #   full_ft  → pi05_g2_vr (multi-GPU / FSDP)
  #   smoke    → pi05_g2_vr_low_mem + short run
  PRESET="low_mem"
  CONFIG_NAME=""
  # Set when CLI passes --preset / --smoke / --full or --config-name.
  G2_PRESET_FROM_CLI=false
  G2_CONFIG_FROM_CLI=false
  EXP_NAME=""
  REPO_ID="g2_vr_lerobot_v21"

  # norm: auto = reuse existing norm_stats.json | skip | force
  NORM="auto"
  MAX_FRAMES=""

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
  USE_PROXY=true

  CHECKPOINT_BASE_DIR="./checkpoints"
  ASSETS_BASE_DIR="./assets"

  # Empty = do not override TrainConfig (see src/openpi/training/config.py).
  BATCH_SIZE=""
  NUM_WORKERS=""
  NUM_TRAIN_STEPS=""
  LOG_INTERVAL=""
  SAVE_INTERVAL=""
  KEEP_PERIOD=""
  SEED=""
  FSDP_DEVICES=""

  OVERWRITE=true
  RESUME=false

  # wandb: auto = on for low_mem/full_ft, off for smoke
  WANDB="auto"
  WANDB_PROJECT="G2_openpi"

  DRY_RUN=false
  NORM_ONLY=false
}

g2_train_usage() {
  cat <<EOF
Usage: bash scripts/g2/train/train_pipeline.sh [options] [-- tyro-overrides...]

  Edit the CONFIG block in train_pipeline.sh for daily use.
  Defaults and model structure live in src/openpi/training/config.py.

  Presets (when --config-name is not set):
    --preset low_mem   LoRA / 24GB (default)
    --preset full_ft   full fine-tune (multi-GPU)
    --smoke            alias for --preset smoke

  Legacy aliases: --full → --preset full_ft

  Pipeline:
    --norm-only        only compute norm stats
    --skip-norm        reuse or skip norm stats step
    --force-norm       recompute norm stats
    --dry-run          print uv run commands, do not execute

  Experiment:
    --config-name NAME | --exp-name NAME | --repo-id ID
    --checkpoint-base-dir DIR | --assets-base-dir DIR | --project-name NAME

  Training overrides (empty in CONFIG = use TrainConfig default):
    --batch-size N | --num-workers N | --num-train-steps N
    --log-interval N | --save-interval N | --keep-period N|none
    --seed N | --fsdp-devices N

  Checkpoint / W&B:
    --resume | --no-overwrite | --wandb | --no-wandb

  Environment:
    --no-proxy | --help

  Advanced tyro flags (lr, model, etc.) after --:
    ... train_pipeline.sh --preset low_mem -- --optimizer.peak-lr=1e-5
EOF
}

g2_train_append_if_set() {
  local -n _out=$1
  local _var_name=$2
  local _flag=$3
  local _val="${!_var_name}"
  if [[ -n "${_val}" ]]; then
    _out+=("${_flag}=${_val}")
  fi
}

g2_train_parse_args() {
  G2_TRAIN_PASSTHRU=()
  while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--" ]]; then
      shift
      G2_TRAIN_PASSTHRU=("$@")
      return 0
    fi
    case "$1" in
      --preset)
        PRESET="$2"
        G2_PRESET_FROM_CLI=true
        shift 2
        ;;
      --smoke)
        PRESET="smoke"
        G2_PRESET_FROM_CLI=true
        shift
        ;;
      --full)
        PRESET="full_ft"
        G2_PRESET_FROM_CLI=true
        shift
        ;;
      --norm-only)
        NORM_ONLY=true
        shift
        ;;
      --skip-norm)
        NORM="skip"
        shift
        ;;
      --force-norm)
        NORM="force"
        shift
        ;;
      --dry-run)
        DRY_RUN=true
        shift
        ;;
      --max-frames)
        MAX_FRAMES="$2"
        shift 2
        ;;
      --config-name)
        CONFIG_NAME="$2"
        G2_CONFIG_FROM_CLI=true
        shift 2
        ;;
      --exp-name)
        EXP_NAME="$2"
        shift 2
        ;;
      --repo-id)
        REPO_ID="$2"
        shift 2
        ;;
      --batch-size)
        BATCH_SIZE="$2"
        shift 2
        ;;
      --num-train-steps)
        NUM_TRAIN_STEPS="$2"
        shift 2
        ;;
      --num-workers)
        NUM_WORKERS="$2"
        shift 2
        ;;
      --log-interval)
        LOG_INTERVAL="$2"
        shift 2
        ;;
      --save-interval)
        SAVE_INTERVAL="$2"
        shift 2
        ;;
      --keep-period)
        KEEP_PERIOD="$2"
        shift 2
        ;;
      --seed)
        SEED="$2"
        shift 2
        ;;
      --fsdp-devices)
        FSDP_DEVICES="$2"
        shift 2
        ;;
      --checkpoint-base-dir)
        CHECKPOINT_BASE_DIR="$2"
        shift 2
        ;;
      --assets-base-dir)
        ASSETS_BASE_DIR="$2"
        shift 2
        ;;
      --project-name)
        WANDB_PROJECT="$2"
        shift 2
        ;;
      --resume)
        RESUME=true
        OVERWRITE=false
        shift
        ;;
      --no-overwrite)
        OVERWRITE=false
        shift
        ;;
      --wandb)
        WANDB="on"
        shift
        ;;
      --no-wandb)
        WANDB="off"
        shift
        ;;
      --no-proxy)
        USE_PROXY=false
        shift
        ;;
      -h|--help)
        g2_train_usage
        exit 0
        ;;
      *)
        echo "Unknown arg: $1 (use -- before tyro overrides)" >&2
        g2_train_usage
        exit 1
        ;;
    esac
  done
}

g2_train_apply_preset() {
  # Script CONFIG_NAME wins unless CLI asked for a preset or --config-name.
  local _set_config=false
  if [[ "${G2_CONFIG_FROM_CLI}" == true ]]; then
    _set_config=false
  elif [[ "${G2_PRESET_FROM_CLI}" == true || -z "${CONFIG_NAME}" ]]; then
    _set_config=true
  fi

  case "${PRESET}" in
    smoke)
      if [[ "${_set_config}" == true ]]; then CONFIG_NAME="pi05_g2_vr_low_mem"; fi
      if [[ -z "${NUM_TRAIN_STEPS}" ]]; then NUM_TRAIN_STEPS=20; fi
      if [[ -z "${BATCH_SIZE}" ]]; then BATCH_SIZE=1; fi
      if [[ "${WANDB}" == "auto" ]]; then WANDB="off"; fi
      ;;
    low_mem)
      if [[ "${_set_config}" == true ]]; then CONFIG_NAME="pi05_g2_vr_low_mem"; fi
      if [[ "${WANDB}" == "auto" ]]; then WANDB="on"; fi
      ;;
    full_ft)
      if [[ "${_set_config}" == true ]]; then CONFIG_NAME="pi05_g2_vr"; fi
      if [[ "${WANDB}" == "auto" ]]; then WANDB="on"; fi
      ;;
    *)
      echo "ERROR: unknown PRESET=${PRESET} (expected low_mem | full_ft | smoke)" >&2
      exit 1
      ;;
  esac

  if [[ -z "${EXP_NAME}" ]]; then
    if [[ "${PRESET}" == "smoke" ]]; then
      EXP_NAME="g2_vr_smoke_$(date +%Y%m%d_%H%M%S)"
    else
      EXP_NAME="g2_vr_pi05_$(date +%Y%m%d_%H%M%S)"
    fi
  fi
}

g2_train_validate() {
  if [[ "${CONFIG_NAME}" == "pi05_g2_vr" ]]; then
    local _gpu_count
    _gpu_count="$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")"
    if [[ "${_gpu_count}" -le 1 ]]; then
      echo "WARN: ${CONFIG_NAME} full fine-tune usually OOMs on a single 24GB GPU." >&2
      echo "      Use --preset low_mem, or add GPUs and --fsdp-devices." >&2
    fi
  fi

  if [[ "${RESUME}" == true && "${OVERWRITE}" == true ]]; then
    echo "ERROR: --resume and --overwrite conflict" >&2
    exit 1
  fi
}

g2_train_norm_stats_path() {
  local _root="${G2_TRAIN_PROJECT_ROOT}"
  if [[ "${ASSETS_BASE_DIR}" = /* ]]; then
    echo "${ASSETS_BASE_DIR}/${CONFIG_NAME}/${REPO_ID}/norm_stats.json"
  else
    echo "${_root}/${ASSETS_BASE_DIR#./}/${CONFIG_NAME}/${REPO_ID}/norm_stats.json"
  fi
}

g2_train_ckpt_dir() {
  local _root="${G2_TRAIN_PROJECT_ROOT}"
  if [[ "${CHECKPOINT_BASE_DIR}" = /* ]]; then
    echo "${CHECKPOINT_BASE_DIR}/${CONFIG_NAME}/${EXP_NAME}"
  else
    echo "${_root}/${CHECKPOINT_BASE_DIR}/${CONFIG_NAME}/${EXP_NAME}"
  fi
}

g2_train_resolve_norm_action() {
  local _norm_stats
  _norm_stats="$(g2_train_norm_stats_path)"

  case "${NORM}" in
    force)
      G2_TRAIN_SKIP_NORM=false
      ;;
    skip)
      G2_TRAIN_SKIP_NORM=true
      ;;
    auto)
      if [[ -f "${_norm_stats}" ]]; then
        G2_TRAIN_SKIP_NORM=true
      else
        G2_TRAIN_SKIP_NORM=false
      fi
      ;;
    *)
      echo "ERROR: unknown NORM=${NORM} (expected auto | skip | force)" >&2
      exit 1
      ;;
  esac

  G2_TRAIN_NORM_STATS="${_norm_stats}"
}

g2_train_wandb_enabled() {
  case "${WANDB}" in
    on|true|1) echo true ;;
    off|false|0) echo false ;;
    auto)
      if [[ "${PRESET}" == "smoke" ]]; then
        echo false
      else
        echo true
      fi
      ;;
    *)
      echo "ERROR: unknown WANDB=${WANDB} (expected auto | on | off)" >&2
      exit 1
      ;;
  esac
}

g2_train_build_norm_args() {
  G2_NORM_ARGS=(
    scripts/compute_norm_stats.py
    --config-name "${CONFIG_NAME}"
    --repo-id="${REPO_ID}"
  )
  if [[ -n "${MAX_FRAMES}" ]]; then
    G2_NORM_ARGS+=(--max-frames="${MAX_FRAMES}")
  fi
}

g2_train_build_train_args() {
  G2_TRAIN_ARGS=(
    scripts/train.py "${CONFIG_NAME}"
    --exp-name="${EXP_NAME}"
    --checkpoint-base-dir="${CHECKPOINT_BASE_DIR}"
    --assets-base-dir="${ASSETS_BASE_DIR}"
    --project-name="${WANDB_PROJECT}"
    --data.repo-id="${REPO_ID}"
  )

  g2_train_append_if_set G2_TRAIN_ARGS BATCH_SIZE --batch-size
  g2_train_append_if_set G2_TRAIN_ARGS NUM_WORKERS --num-workers
  g2_train_append_if_set G2_TRAIN_ARGS NUM_TRAIN_STEPS --num-train-steps
  g2_train_append_if_set G2_TRAIN_ARGS LOG_INTERVAL --log-interval
  g2_train_append_if_set G2_TRAIN_ARGS SAVE_INTERVAL --save-interval
  if [[ -n "${KEEP_PERIOD}" ]]; then
    if [[ "${KEEP_PERIOD}" == "none" || "${KEEP_PERIOD}" == "None" ]]; then
      G2_TRAIN_ARGS+=(--keep-period=None)
    else
      G2_TRAIN_ARGS+=(--keep-period="${KEEP_PERIOD}")
    fi
  fi
  g2_train_append_if_set G2_TRAIN_ARGS SEED --seed
  g2_train_append_if_set G2_TRAIN_ARGS FSDP_DEVICES --fsdp-devices

  if [[ "${OVERWRITE}" == true ]]; then
    G2_TRAIN_ARGS+=(--overwrite)
  else
    G2_TRAIN_ARGS+=(--no-overwrite)
  fi
  if [[ "${RESUME}" == true ]]; then
    G2_TRAIN_ARGS+=(--resume)
  else
    G2_TRAIN_ARGS+=(--no-resume)
  fi
  if [[ "$(g2_train_wandb_enabled)" == true ]]; then
    G2_TRAIN_ARGS+=(--wandb-enabled)
  else
    G2_TRAIN_ARGS+=(--no-wandb-enabled)
  fi

  if [[ ${#G2_TRAIN_PASSTHRU[@]} -gt 0 ]]; then
    G2_TRAIN_ARGS+=("${G2_TRAIN_PASSTHRU[@]}")
  fi
}

g2_train_print_summary() {
  local _wandb_enabled
  _wandb_enabled="$(g2_train_wandb_enabled)"
  echo "========================================"
  echo "pipeline:   norm_stats -> train (preset=${PRESET})"
  echo "project:    ${G2_TRAIN_PROJECT_ROOT}"
  echo "config:     ${CONFIG_NAME}"
  echo "exp:        ${EXP_NAME}"
  echo "data:       ${HF_LEROBOT_HOME}/${REPO_ID}"
  echo "hf_cache:   ${HF_DATASETS_CACHE}"
  echo "openpi:     ${OPENPI_DATA_HOME}"
  echo "norm:       ${NORM} (skip=${G2_TRAIN_SKIP_NORM})"
  echo "max_frames: ${MAX_FRAMES:-<full>}"
  echo "ckpt_dir:   $(g2_train_ckpt_dir)"
  echo "batch/steps:${BATCH_SIZE:-<cfg>} / ${NUM_TRAIN_STEPS:-<cfg>}"
  echo "save/log:   ${SAVE_INTERVAL:-<cfg>} / ${LOG_INTERVAL:-<cfg>}"
  echo "wandb:      ${_wandb_enabled}  project=${WANDB_PROJECT}"
  echo "overwrite:  ${OVERWRITE}  resume=${RESUME}"
  echo "GPU:        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  if [[ "${DRY_RUN}" == true ]]; then
    echo "dry_run:    true"
  fi
  if [[ ${#G2_TRAIN_PASSTHRU[@]} -gt 0 ]]; then
    echo "passthru:   ${G2_TRAIN_PASSTHRU[*]}"
  fi
  echo "========================================"
}

g2_train_preflight() {
  cd "${G2_TRAIN_PROJECT_ROOT}"
  export CUDA_VISIBLE_DEVICES
  export XLA_PYTHON_CLIENT_MEM_FRACTION
  [[ "${USE_PROXY}" == true ]] && g2_openpi_setup_proxy || g2_openpi_clear_proxy

  g2_openpi_require_dataset "${REPO_ID}" || exit 1

  if [[ ! -d "${G2_TRAIN_PROJECT_ROOT}/.venv" ]]; then
    echo "ERROR: missing .venv — run bash scripts/g2/bootstrap.sh" >&2
    exit 1
  fi

  local _ep_stats="${HF_LEROBOT_HOME}/${REPO_ID}/meta/episodes_stats.jsonl"
  if [[ ! -f "${_ep_stats}" ]]; then
    echo "ERROR: missing ${_ep_stats}" >&2
    echo "  OpenPI LeRobot 0.1 requires episodes_stats.jsonl for codebase_version v2.1." >&2
    exit 1
  fi
}

g2_train_run_pipeline() {
  g2_train_preflight
  g2_train_apply_preset
  g2_train_validate
  g2_train_resolve_norm_action
  g2_train_build_norm_args
  g2_train_build_train_args
  g2_train_print_summary

  if [[ "${DRY_RUN}" == true ]]; then
    if [[ "${G2_TRAIN_SKIP_NORM}" != true ]]; then
      printf 'uv run'
      printf ' %q' "${G2_NORM_ARGS[@]}"
      echo
    else
      echo "[dry-run] skip norm stats (reuse ${G2_TRAIN_NORM_STATS})"
    fi
    if [[ "${NORM_ONLY}" != true ]]; then
      printf 'uv run'
      printf ' %q' "${G2_TRAIN_ARGS[@]}"
      echo
    fi
    return 0
  fi

  if [[ "${G2_TRAIN_SKIP_NORM}" != true ]]; then
    echo ""
    echo "[1/2] compute_norm_stats ..."
    g2_openpi_setup_hf_cache
    uv run "${G2_NORM_ARGS[@]}"
  else
    echo ""
    echo "[1/2] compute_norm_stats skipped (reuse ${G2_TRAIN_NORM_STATS})"
  fi

  if [[ "${NORM_ONLY}" == true ]]; then
    echo ""
    echo "done. norm stats: ${G2_TRAIN_NORM_STATS}"
    return 0
  fi

  echo ""
  echo "[2/2] train ..."
  g2_openpi_setup_hf_cache
  uv run "${G2_TRAIN_ARGS[@]}"

  echo ""
  echo "done. checkpoints: $(g2_train_ckpt_dir)"
}
