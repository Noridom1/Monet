#!/usr/bin/env bash
# Phase A: generate + capture Monet latent hidden states for one example.
# Run on the A100 box (the local machine is insufficient for the 7B model).
set -euo pipefail

ENV_NAME="${ENV_NAME:-monet}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"   # the Monet patch resolves paths relative to CWD

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

export LATENT_SIZE="${LATENT_SIZE:-10}"
export MODEL_PATH="${MODEL_PATH:-$REPO_DIR/models/Monet-7B}"

if [ ! -d "$MODEL_PATH" ]; then
  echo "[phaseA] ERROR: MODEL_PATH '$MODEL_PATH' not found. Run 01_download_model.sh first." >&2
  exit 1
fi

echo "[phaseA] MODEL_PATH=$MODEL_PATH  LATENT_SIZE=$LATENT_SIZE"
python -m inspection.generate_latents \
  --model_path "$MODEL_PATH" \
  --latent_size "$LATENT_SIZE" \
  --out "inspection/outputs/demo/trace.pt"
