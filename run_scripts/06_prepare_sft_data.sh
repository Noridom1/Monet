#!/usr/bin/env bash
# Download and validate the official Monet-SFT-125K training dataset.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_REPO="${DATASET_REPO:-NOVAglow646/Monet-SFT-125K}"
DATASET_DIR="${DATASET_DIR:-$REPO_DIR/data/Monet-SFT-125K}"

if ! command -v hf >/dev/null 2>&1; then
  echo "[prepare-sft] ERROR: Hugging Face CLI not found." >&2
  echo "Install it with: pip install 'huggingface_hub[cli,hf_transfer]>=0.34.0,<1.0'" >&2
  exit 1
fi

mkdir -p "$DATASET_DIR"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

echo "[prepare-sft] Downloading ${DATASET_REPO} -> ${DATASET_DIR}"
hf download "$DATASET_REPO" \
  --repo-type dataset \
  --local-dir "$DATASET_DIR"

python "$REPO_DIR/run_scripts/validate_sft_dataset.py" "$DATASET_DIR"

echo
echo "[prepare-sft] Dataset is ready."
echo "Run stage 1 with:"
echo "  DATASET_DIR=\"$DATASET_DIR\" bash script_examples/sft_stage1.sh"
