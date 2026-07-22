#!/usr/bin/env bash
# Prepare the local Monet environment and ensure the requested model is available.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/models/Monet-7B}"
FORCE_MODEL_DOWNLOAD="${FORCE_MODEL_DOWNLOAD:-0}"

if [ "$FORCE_MODEL_DOWNLOAD" != "0" ] && [ "$FORCE_MODEL_DOWNLOAD" != "1" ]; then
  echo "[setup-local] ERROR: FORCE_MODEL_DOWNLOAD must be 0 or 1." >&2
  exit 2
fi

bash "$REPO_DIR/run_scripts/setup/environment.sh"

has_weights=0
if [ -f "$MODEL_DIR/model.safetensors" ] || [ -f "$MODEL_DIR/model.safetensors.index.json" ]; then
  has_weights=1
fi

if [ "$FORCE_MODEL_DOWNLOAD" = "0" ] && [ -f "$MODEL_DIR/config.json" ] && [ "$has_weights" = "1" ]; then
  echo "[setup-local] reusing model at $MODEL_DIR"
else
  MODEL_DIR="$MODEL_DIR" bash "$REPO_DIR/run_scripts/setup/download_model.sh"
fi

echo
echo "[setup-local] DONE. Environment and model are ready."
echo "              Next: bash run_scripts/inference/run_example.sh"
