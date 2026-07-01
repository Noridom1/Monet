#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/sft_config.sh"
cd "$REPO_DIR"

CE_EMPHASIZE_FACTOR="${CE_EMPHASIZE_FACTOR:-2.0}"
SAVE_CKPT="${SAVE_CKPT:-sft_stage1_ce${CE_EMPHASIZE_FACTOR}}"

torchrun --nproc-per-node="$NPROC_PER_NODE" --master-port="${MASTER_PORT:-29501}" -m src.main \
  --epochs 4 \
  --bsz 1 \
  --grad_accum_steps 16 \
  --stage "sft_stage1" \
  --data_path "${SFT_DATA_PATHS[@]}" \
  --load_model_path "$BASE_MODEL_PATH" \
  --save_model_path "$CHECKPOINT_DIR/sft_stage1/$SAVE_CKPT" \
  --dataset_root "$DATASET_DIR" \
  --deepspeed ./deepspeed/ds_zero2_gpu.json \
  --wandb_name "$SAVE_CKPT" \
  --ce_emphasize_factor "$CE_EMPHASIZE_FACTOR"
