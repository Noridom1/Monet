#!/usr/bin/env bash
# Download a Hugging Face benchmark and normalize it to Monet's common layout:
#   data/<name>/images/        one PNG per image
#   data/<name>/samples.json   records with images replaced by relative paths
#
# Image columns are auto-detected, so most datasets work with just a name (+ repo).
# Known names live in prepare_dataset.py's REGISTRY; anything else takes --repo.
#
# Examples:
#   bash run_scripts/evaluation/prepare_data.sh VisualPuzzles
#   bash run_scripts/evaluation/prepare_data.sh MathVision --split testmini
#   bash run_scripts/evaluation/prepare_data.sh MyBench --repo org/MyBench --split test
#   bash run_scripts/evaluation/prepare_data.sh VisualPuzzles --limit 20   # quick smoke test
set -euo pipefail

ENV_NAME="${ENV_NAME:-monet}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"   # faster downloads

if [ "$#" -eq 0 ]; then
  echo "usage: bash run_scripts/evaluation/prepare_data.sh <name> [--repo org/name] [--split S] [--config C] [--limit N]" >&2
  exit 1
fi

# For gated datasets, run `hf auth login` first.
python "$REPO_DIR/run_scripts/evaluation/prepare_dataset.py" "$@"
