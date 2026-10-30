#!/usr/bin/env bash
set -euo pipefail

# 在这里填写训练 YAML 路径：绝对路径，或相对于 openpi 仓库根目录的路径。
G2_TRAIN_CONFIG="/home/zijianwang/openpi/scripts/g2/configs/G2_ChaPai.yaml"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"
if [[ $# -gt 0 && "$1" != -* ]]; then
  set -- --config "$1" "${@:2}"
fi
g2_openpi_run_train "$@"
