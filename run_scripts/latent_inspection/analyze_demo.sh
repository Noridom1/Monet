#!/usr/bin/env bash
# Phase B (Objective A — logit lens): replay a captured trace and decode latents.
# Run on the A100 box, after capture_demo.sh produced a trace.
set -euo pipefail

ENV_NAME="${ENV_NAME:-monet}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"   # the Monet patch resolves paths relative to CWD

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

export LATENT_SIZE="${LATENT_SIZE:-10}"
export MODEL_PATH="${MODEL_PATH:-$REPO_DIR/models/Monet-7B}"
TRACE="${TRACE:-inspection/outputs/demo/trace.pt}"

if [ ! -f "$TRACE" ]; then
  echo "[phaseB] ERROR: trace '$TRACE' not found. Run capture_demo.sh first." >&2
  exit 1
fi

echo "[phaseB] MODEL_PATH=$MODEL_PATH  TRACE=$TRACE"
python -m inspection.inspect \
  --trace "$TRACE" \
  --model_path "$MODEL_PATH" \
  --topk 20
