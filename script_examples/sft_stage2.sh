#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/sft_config.sh"
cd "$REPO_DIR"

TEACHER="${TEACHER:-sft_stage1_ce2.0}"
TEACHER_REPS_DIR="$CHECKPOINT_DIR/monet_precomputed_observation_token_teacher_reps/$TEACHER"

# Step 1: cache observation-token representations from the stage-1 teacher.
torchrun --nproc-per-node="$NPROC_PER_NODE" --master-port="${PRECOMPUTE_PORT:-29505}" \
  -m src.precompute_teacher_reps \
  --bsz 1 \
  --data_path "${SFT_DATA_PATHS[@]}" \
  --load_model_path "$CHECKPOINT_DIR/sft_stage1/$TEACHER" \
  --save_model_path "$TEACHER_REPS_DIR" \
  --dataset_root "$DATASET_DIR" \
  --deepspeed ./deepspeed/ds_zero2_gpu.json \
  --output_hidden_states \
  --alignment_layer all_layers

# Step 2: train the latent student against the cached representations.
LATENT_SIZE="${LATENT_SIZE:-8}"
CE_EMPHASIZE_FACTOR="${CE_EMPHASIZE_FACTOR:-4.0}"
ALIGNMENT_WEIGHT="${ALIGNMENT_WEIGHT:-2.0}"
EMPHASIZE_LATENT_WEIGHT="${EMPHASIZE_LATENT_WEIGHT:-2.0}"
SAVE_CKPT="${SAVE_CKPT:-sft_stage2_latent${LATENT_SIZE}_ce${CE_EMPHASIZE_FACTOR}_al${ALIGNMENT_WEIGHT}_emph${EMPHASIZE_LATENT_WEIGHT}}"

torchrun --nproc-per-node="$NPROC_PER_NODE" --master-port="${MASTER_PORT:-29501}" -m src.main \
  --epochs 2 \
  --bsz 1 \
  --grad_accum_steps 16 \
  --stage "sft_stage2" \
  --data_path "${SFT_DATA_PATHS[@]}" \
  --log_file "$REPO_DIR/log.txt" \
  --load_model_path "$CHECKPOINT_DIR/sft_stage1/$TEACHER" \
  --save_model_path "$CHECKPOINT_DIR/sft_stage2/$SAVE_CKPT" \
  --dataset_root "$DATASET_DIR" \
  --deepspeed ./deepspeed/ds_zero2_gpu.json \
  --wandb_name "$SAVE_CKPT" \
  --latent_size "$LATENT_SIZE" \
  --alignment_weight "$ALIGNMENT_WEIGHT" \
  --ce_emphasize_factor "$CE_EMPHASIZE_FACTOR" \
  --emphasize_latent_weight "$EMPHASIZE_LATENT_WEIGHT" \
  --teacher_reps_dir "$TEACHER_REPS_DIR" \
  --alignment_layer all_layers
