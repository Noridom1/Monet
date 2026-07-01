#!/usr/bin/env bash
# Shared local paths for the three SFT stages. Override any value in the environment.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_DIR="${DATASET_DIR:-$REPO_DIR/data/Monet-SFT-125K}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-$REPO_DIR/models/Qwen2.5-VL-7B-Instruct}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$REPO_DIR/checkpoints}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

SFT_DATA_PATHS=(
  "$DATASET_DIR/Visual_CoT/train.json"
  "$DATASET_DIR/CogCoM/train.json"
  "$DATASET_DIR/ReFocus/train.json"
  "$DATASET_DIR/Zebra_CoT_count/train.json"
  "$DATASET_DIR/Zebra_CoT_visual_search/train.json"
  "$DATASET_DIR/Zebra_CoT_geometry/train.json"
)

for data_path in "${SFT_DATA_PATHS[@]}"; do
  if [[ ! -f "$data_path" ]]; then
    echo "Missing training file: $data_path" >&2
    echo "Prepare the dataset with: bash run_scripts/06_prepare_sft_data.sh" >&2
    exit 1
  fi
done
