#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="${1:-/data1/lkh/HMAR/experiments/hmar-finetune-mask-d16}"
PUBLIC_HMAR_CKPT="${2:-/data1/lkh/HMAR/hmar-d16.pth}"
VAE_CKPT="${3:-/data1/lkh/HMAR/vae_ch160v4096z32.pth}"
DATASET_ROOT="${4:-/data1/lkh/HMAR/dataset_v3_patches}"
REF_DIR="${5:-ref_all_dir}"
RESULTS_JSON="${6:-full_eval_results.json}"

python -m evaluate.run_full_eval \
  --experiment_dir "${EXPERIMENT_DIR}" \
  --public_hmar_ckpt "${PUBLIC_HMAR_CKPT}" \
  --vae_ckpt "${VAE_CKPT}" \
  --dataset_root "${DATASET_ROOT}" \
  --sample_config hmar-d16 \
  --batch_size 8 \
  --total_samples 8700 \
  --class_counts "0:954,1:1848,2:1602,3:1560,4:1482,5:1254" \
  --ref_dir "${REF_DIR}" \
  --results_json "${RESULTS_JSON}"
