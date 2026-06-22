#!/usr/bin/env bash
# Set up the `monet` conda environment for INFERENCE only.
# Run once. Safe to re-run (pip install is idempotent).
set -euo pipefail

ENV_NAME="${ENV_NAME:-monet}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- locate & load conda --------------------------------------------------
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"

# --- create env if missing ------------------------------------------------
if ! conda env list | grep -qE "^\s*${ENV_NAME}\s"; then
  echo "[setup] creating conda env '${ENV_NAME}' (python=3.10)"
  conda create -y -n "$ENV_NAME" python=3.10
else
  echo "[setup] conda env '${ENV_NAME}' already exists, reusing"
fi

conda activate "$ENV_NAME"

# --- install deps ---------------------------------------------------------
# requirements.txt pins vllm==0.10.0, transformers==4.54.0 (required for inference).
echo "[setup] installing requirements (vllm==0.10.0, transformers==4.54.0, ...)"
pip install -r "$REPO_DIR/requirements.txt"

# Hugging Face downloader CLI + fast transfer (used by 01_download_model.sh).
# Pin <1.0: transformers==4.54.0 and tokenizers require huggingface-hub<1.0,>=0.34.0.
pip install "huggingface_hub[cli,hf_transfer]>=0.34.0,<1.0"

echo
echo "[setup] DONE. Environment '${ENV_NAME}' is ready."
echo "        Next: bash run_scripts/01_download_model.sh"
