#!/usr/bin/env bash
# Adapt a prepared dataset (run_scripts/05_prepare_data.sh output) into an inspection
# manifest, then you can run_batch over it. Picks N samples (datasets have thousands of
# rows; inspection is expensive). Writes data/<name>/inspect_manifest.json reusing images/.
#
# env knobs: DATA_DIR=data/VisualPuzzles  N=10  MODE=head|random  SEED=0
set -euo pipefail

ENV_NAME="${ENV_NAME:-monet}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

DATA_DIR="${DATA_DIR:-data/VisualPuzzles}"
N="${N:-10}"
MODE="${MODE:-head}"
SEED="${SEED:-0}"

if [ ! -f "$DATA_DIR/samples.json" ]; then
  echo "[prepare-ds] ERROR: '$DATA_DIR/samples.json' not found. Run run_scripts/05_prepare_data.sh first." >&2
  exit 1
fi

echo "[prepare-ds] data_dir=$DATA_DIR n=$N mode=$MODE seed=$SEED"
python -m inspection.prepare_dataset_samples \
  --data_dir "$DATA_DIR" \
  --n "$N" \
  --mode "$MODE" \
  --seed "$SEED"
