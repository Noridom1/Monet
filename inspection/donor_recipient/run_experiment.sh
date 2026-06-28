#!/usr/bin/env bash
# Sequential MMVP donor-recipient experiment. One 7B checkpoint is resident at a time.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

ENV_NAME="${ENV_NAME:-monet}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

GPU_ID="${GPU_ID:-3}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATA_DIR="${DATA_DIR:-inspection/donor_recipient/data/mmvp}"
MANIFEST="${MANIFEST:-$DATA_DIR/manifest.json}"
OUTPUT_DIR="${OUTPUT_DIR:-inspection/donor_recipient/outputs/mmvp_seed0}"
DONOR_MODEL_PATH="${DONOR_MODEL_PATH:-models/Monet-7B}"
RECIPIENT_MODEL_PATH="${RECIPIENT_MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
LATENT_SIZE="${LATENT_SIZE:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
SEEDS="${SEEDS:-0}"
TARGET_LIMIT="${TARGET_LIMIT:-}"
DONOR_LIMIT="${DONOR_LIMIT:-}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"

if [ ! -f "$MANIFEST" ]; then
  python -m inspection.donor_recipient.prepare_mmvp --data_dir "$DATA_DIR"
fi
if [ ! -d "$DONOR_MODEL_PATH" ]; then
  echo "[experiment] ERROR: donor checkpoint not found: $DONOR_MODEL_PATH" >&2
  exit 1
fi

# For the default seed-0 smoke test, target i receives the next sample's wrong donor.
# Generate one extra donor when a target limit is requested. Other seed lists require
# explicit DONOR_LIMIT or the complete donor set.
if [ -z "$DONOR_LIMIT" ] && [ -n "$TARGET_LIMIT" ] && [ "$SEEDS" = "0" ]; then
  DONOR_LIMIT=$((TARGET_LIMIT + 1))
fi

DONOR_ARGS=()
RECIPIENT_ARGS=()
ANALYSIS_ARGS=()
if [ -n "$DONOR_LIMIT" ]; then
  DONOR_ARGS+=(--limit "$DONOR_LIMIT")
fi
if [ -n "$TARGET_LIMIT" ]; then
  RECIPIENT_ARGS+=(--limit "$TARGET_LIMIT")
  ANALYSIS_ARGS+=(--limit "$TARGET_LIMIT")
fi

echo "[experiment] physical GPU=$GPU_ID (visible as cuda:0), output=$OUTPUT_DIR"
echo "[experiment] === Stage 1: Monet donor capture ==="
python -m inspection.donor_recipient.generate_donors \
  --manifest "$MANIFEST" \
  --model_path "$DONOR_MODEL_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --latent_size "$LATENT_SIZE" \
  "${DONOR_ARGS[@]}"

echo "[experiment] === Stage 2: vanilla recipient interventions ==="
python -m inspection.donor_recipient.run_recipients \
  --manifest "$MANIFEST" \
  --donor_dir "$OUTPUT_DIR/donors" \
  --output_dir "$OUTPUT_DIR" \
  --model_path "$RECIPIENT_MODEL_PATH" \
  --latent_size "$LATENT_SIZE" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --seeds "$SEEDS" \
  --attn_implementation "$ATTN_IMPLEMENTATION" \
  "${RECIPIENT_ARGS[@]}"

echo "[experiment] === Analysis ==="
python -m inspection.donor_recipient.analyze_results \
  --manifest "$MANIFEST" \
  --output_dir "$OUTPUT_DIR" \
  --seeds "$SEEDS" \
  "${ANALYSIS_ARGS[@]}"

echo "[experiment] DONE: $OUTPUT_DIR/report.md"

