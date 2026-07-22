#!/usr/bin/env bash
# Download the Monet-7B model from Hugging Face for inference.
# Defaults to the RL-finished model NOVAglow646/Monet-7B.
# Override the repo with MODEL_REPO=... (e.g. NOVAglow646/Monet-SFT-7B).
set -euo pipefail

ENV_NAME="${ENV_NAME:-monet}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

MODEL_REPO="${MODEL_REPO:-NOVAglow646/Monet-7B}"
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/models/Monet-7B}"

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"   # set 0 if transfer backend stalls

mkdir -p "$MODEL_DIR"
echo "[download] ${MODEL_REPO} -> ${MODEL_DIR}"
# For gated/private repos, run `hf auth login` first.
hf download "$MODEL_REPO" --local-dir "$MODEL_DIR"

echo
echo "[download] DONE. Model at: ${MODEL_DIR}"
echo "        Next: MODEL_PATH=${MODEL_DIR} bash run_scripts/inference/run_example.sh"
