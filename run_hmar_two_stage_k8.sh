#!/usr/bin/env bash
set -euo pipefail

DATA_PATH="${1:-/data1/lkh/HMAR/dataset_v3_patches}"
EXPERIMENT="${2:-hmar-finetune-mask-d16-k8}"
GPU_ID="${3:-1}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

echo "[stage 1] ns"
HMAR_STAGE=ns python finetune_hmar_two_stage.py --experiment "${EXPERIMENT}" --data_path "${DATA_PATH}"

echo "[stage 2] mask"
HMAR_STAGE=mask HMAR_RESUME="${ROOT_DIR}/experiments/${EXPERIMENT}/ar-ckpt-best-ns.pth" \
python finetune_hmar_two_stage.py --experiment "${EXPERIMENT}" --data_path "${DATA_PATH}"
