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
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_K="${TOP_K:-50}"
TOP_P="${TOP_P:-0.8}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
SEED="${SEED:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
TRACE="${TRACE:-inspection/outputs/demo/trace.pt}"

if [ ! -d "$MODEL_PATH" ]; then
  echo "[phaseA] ERROR: MODEL_PATH '$MODEL_PATH' not found. Run 01_download_model.sh first." >&2
  exit 1
fi

echo "[phaseA] MODEL_PATH=$MODEL_PATH  LATENT_SIZE=$LATENT_SIZE  TRACE=$TRACE"
echo "[phaseA] temperature=$TEMPERATURE top_k=$TOP_K top_p=$TOP_P repetition_penalty=$REPETITION_PENALTY seed=$SEED max_new_tokens=$MAX_NEW_TOKENS"
python -m inspection.generate_latents \
  --model_path "$MODEL_PATH" \
  --latent_size "$LATENT_SIZE" \
  --temperature "$TEMPERATURE" \
  --top_k "$TOP_K" \
  --top_p "$TOP_P" \
  --repetition_penalty "$REPETITION_PENALTY" \
  --seed "$SEED" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --out "$TRACE"
