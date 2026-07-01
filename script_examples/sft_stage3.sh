#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/sft_config.sh"
cd "$REPO_DIR"

STAGE1_MODEL="${STAGE1_MODEL:-sft_stage1_ce2.0}"
TEACHER_LATENT_SIZE="${TEACHER_LATENT_SIZE:-8}"
TEACHER_CE_EMPHASIZE_FACTOR="${TEACHER_CE_EMPHASIZE_FACTOR:-4.0}"
TEACHER_ALIGN_WEIGHT="${TEACHER_ALIGN_WEIGHT:-2.0}"
TEACHER_EMPHASIZE_LATENT_WEIGHT="${TEACHER_EMPHASIZE_LATENT_WEIGHT:-2.0}"
TEACHER="${TEACHER:-sft_stage2_latent${TEACHER_LATENT_SIZE}_ce${TEACHER_CE_EMPHASIZE_FACTOR}_al${TEACHER_ALIGN_WEIGHT}_emph${TEACHER_EMPHASIZE_LATENT_WEIGHT}}"
TEACHER_LATENT_DIR="$CHECKPOINT_DIR/monet_precomputed_target_latent/$TEACHER"

# Step 1: cache target latent embeddings from the stage-2 teacher.
torchrun --nproc-per-node="$NPROC_PER_NODE" --master-port="${PRECOMPUTE_PORT:-29505}" \
  -m src.precompute_teacher_latents \
  --bsz 1 \
  --data_path "${SFT_DATA_PATHS[@]}" \
  --load_model_path "$CHECKPOINT_DIR/sft_stage2/$TEACHER" \
  --save_model_path "$TEACHER_LATENT_DIR" \
  --dataset_root "$DATASET_DIR" \
  --deepspeed ./deepspeed/ds_zero2_gpu.json \
  --latent_size "$TEACHER_LATENT_SIZE" \
  --output_hidden_states \
  --resume

# Step 2: distill the stage-2 teacher into the stage-3 student.
LATENT_SIZE="${LATENT_SIZE:-8}"
CE_EMPHASIZE_FACTOR="${CE_EMPHASIZE_FACTOR:-4.0}"
ALIGNMENT_WEIGHT="${ALIGNMENT_WEIGHT:-2.0}"
EMPHASIZE_LATENT_WEIGHT="${EMPHASIZE_LATENT_WEIGHT:-2.0}"
SAVE_CKPT="${SAVE_CKPT:-sft_stage3_target-latent${TEACHER_LATENT_SIZE}-al${TEACHER_ALIGN_WEIGHT}-emph${TEACHER_EMPHASIZE_LATENT_WEIGHT}_student-latent${LATENT_SIZE}-ce${CE_EMPHASIZE_FACTOR}-al${ALIGNMENT_WEIGHT}-emph${EMPHASIZE_LATENT_WEIGHT}}"

torchrun --nproc-per-node="$NPROC_PER_NODE" --master-port="${MASTER_PORT:-29501}" -m src.main \
  --epochs 2 \
  --bsz 1 \
  --grad_accum_steps 16 \
  --stage "sft_stage3" \
  --data_path "${SFT_DATA_PATHS[@]}" \
  --log_file "$REPO_DIR/log.txt" \
  --load_model_path "$CHECKPOINT_DIR/sft_stage1/$STAGE1_MODEL" \
  --save_model_path "$CHECKPOINT_DIR/sft_stage3/$SAVE_CKPT" \
  --dataset_root "$DATASET_DIR" \
  --deepspeed ./deepspeed/ds_zero2_gpu.json \
  --wandb_name "$SAVE_CKPT" \
  --latent_size "$LATENT_SIZE" \
  --alignment_weight "$ALIGNMENT_WEIGHT" \
  --ce_emphasize_factor "$CE_EMPHASIZE_FACTOR" \
  --teacher_latent_dir "$TEACHER_LATENT_DIR" \
  --alignment_layer all_layers
