#!/usr/bin/env bash
# Prepare eval samples (images + questions) into data/ for latent inspection.
# Picks 4 correct + 5 genuine-wrong samples from a VLMEvalKit result file.
# Run on the A100 box: needs the `monet` env (vlmeval) and the localized dataset TSV
# in ~/LMUData (the downloaded result xlsx has no images).
set -euo pipefail

ENV_NAME="${ENV_NAME:-monet}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

RESULTS="${RESULTS:-eval_outputs/full/Monet/T20260622-230719/Monet_MMBench_DEV_EN_gpt-4o-mini_result.xlsx}"
DATASET="${DATASET:-MMBench_DEV_EN}"
N_CORRECT="${N_CORRECT:-4}"
N_INCORRECT="${N_INCORRECT:-5}"
SEED="${SEED:-0}"
OUT="${OUT:-data/inspect_samples}"

if [ ! -f "$RESULTS" ]; then
  echo "[prepare] ERROR: results file '$RESULTS' not found (set RESULTS=...)." >&2
  exit 1
fi

echo "[prepare] results=$RESULTS dataset=$DATASET out=$OUT (correct=$N_CORRECT wrong=$N_INCORRECT seed=$SEED)"
python -m inspection.prepare_eval_samples \
  --results "$RESULTS" \
  --dataset "$DATASET" \
  --n_correct "$N_CORRECT" \
  --n_incorrect "$N_INCORRECT" \
  --seed "$SEED" \
  --out "$OUT"
