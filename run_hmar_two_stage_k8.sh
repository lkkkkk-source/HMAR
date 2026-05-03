#!/usr/bin/env bash
set -euo pipefail

DATA_PATH="${1:-/data1/lkh/HMAR/dataset_v3_patches}"
EXPERIMENT="${2:-hmar-finetune-mask-d16-k8}"
GPU_ID="${3:-1}"
PUBLIC_HMAR_CKPT="${4:-/data1/lkh/HMAR/hmar-d16.pth}"
VAE_CKPT="${5:-/data1/lkh/HMAR/vae_ch160v4096z32.pth}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

echo "[stage 1] ns"
HMAR_STAGE=ns python finetune_hmar_two_stage.py --experiment "${EXPERIMENT}" --data_path "${DATA_PATH}"

echo "[stage 2] mask"
HMAR_STAGE=mask HMAR_RESUME="${ROOT_DIR}/experiments/${EXPERIMENT}/ar-ckpt-best-ns.pth" \
python finetune_hmar_two_stage.py --experiment "${EXPERIMENT}" --data_path "${DATA_PATH}"

MASK_CKPT="${ROOT_DIR}/experiments/${EXPERIMENT}/ar-ckpt-best-mask.pth"
if [[ ! -f "${MASK_CKPT}" ]]; then
  MASK_CKPT="${ROOT_DIR}/experiments/${EXPERIMENT}/ar-ckpt-last-mask.pth"
fi

echo "[eval] build full reference dir"
python -m evaluate.build_reference_dir \
  --split_dirs "${DATA_PATH}/train" "${DATA_PATH}/val" "${DATA_PATH}/test" \
  --out_dir "ref_all_dir_${EXPERIMENT}"

echo "[eval] generate samples from ${MASK_CKPT}"
python -m evaluate.generate_finetune_samples \
  --checkpoint "${MASK_CKPT}" \
  --public_hmar_ckpt "${PUBLIC_HMAR_CKPT}" \
  --sample_config hmar-d16 \
  --vae_ckpt "${VAE_CKPT}" \
  --out_dir "samples_all_8700_${EXPERIMENT}" \
  --total_samples 8700 \
  --batch_size 8 \
  --class_counts "0:954,1:1848,2:1602,3:1560,4:1482,5:1254"

SAMPLE_COUNT=$(find "samples_all_8700_${EXPERIMENT}" -maxdepth 1 -type f | wc -l)
if [[ "${SAMPLE_COUNT}" -ne 8700 ]]; then
  echo "[eval] expected 8700 generated samples, got ${SAMPLE_COUNT}" >&2
  exit 1
fi

echo "[eval] compute metrics"
python -m evaluate.compute_custom_metrics \
  --sample_dir "samples_all_8700_${EXPERIMENT}" \
  --ref_dirs "ref_all_dir_${EXPERIMENT}" \
  > "full_eval_${EXPERIMENT}.log"

echo "[done] metrics written to full_eval_${EXPERIMENT}.log"
