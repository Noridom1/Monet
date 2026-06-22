#!/usr/bin/env bash
# Run the Monet-7B latent-reasoning inference example.
set -euo pipefail

ENV_NAME="${ENV_NAME:-monet}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"   # the vLLM patch resolves paths relative to CWD

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# --- Monet knobs ----------------------------------------------------------
export LATENT_SIZE="${LATENT_SIZE:-10}"                      # # latent embeddings per latent block
export MODEL_PATH="${MODEL_PATH:-$REPO_DIR/models/Monet-7B}" # consumed by vllm_inference_example.py

# GPU memory fraction for vLLM.
#   RTX 4090 (24GB): 0.9 is a tight-but-OK fit for the 7B bf16 weights + KV cache.
#   A100 80GB:       0.9 leaves plenty of headroom; lower it if sharing the GPU.
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"

if [ ! -d "$MODEL_PATH" ]; then
  echo "[run] ERROR: MODEL_PATH '$MODEL_PATH' not found. Run 01_download_model.sh first." >&2
  exit 1
fi

echo "[run] MODEL_PATH=$MODEL_PATH  LATENT_SIZE=$LATENT_SIZE  GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION"
python -m inference.vllm_inference_example
