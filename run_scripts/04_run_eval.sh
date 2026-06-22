#!/usr/bin/env bash
# Evaluate Monet-7B on VLMEvalKit benchmarks — full dataset or a subset.
#
# Subset modes (set SUBSET):
#   SUBSET=full           run the whole dataset (default)
#   SUBSET=head           first k% / first N samples (deterministic)
#   SUBSET=random         random k% / random N samples (seeded by SEED)
# Size is given by exactly one of:
#   FRAC=0.1              fraction, e.g. 10%
#   N=200                 absolute count
#
# Each dataset's official TSV is restored to full before this run, then (for head/random)
# re-subsetted, so runs are reproducible and never accumulate. Official dataset names are
# used, so official evaluators/metrics apply.
set -euo pipefail

ENV_NAME="${ENV_NAME:-monet}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_DIR="${EVAL_DIR:-$REPO_DIR/VLMEvalKit}"

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# --- model + Monet knobs ---------------------------------------------------
export MODEL_PATH="${MODEL_PATH:-$REPO_DIR/models/Monet-7B}"
export LATENT_SIZE="${LATENT_SIZE:-10}"
# System prompt + token budget (run_monet.py reads these; defaults match README).
export MONET_MAX_NEW_TOKENS="${MONET_MAX_NEW_TOKENS:-2048}"

# --- datasets + subset selection -------------------------------------------
# Space-separated official VLMEvalKit dataset names.
DATASETS="${DATASETS:-MMBench_DEV_EN}"
SUBSET="${SUBSET:-full}"            # full | head | random
FRAC="${FRAC:-}"                   # e.g. 0.1
N="${N:-}"                         # e.g. 200
SEED="${SEED:-0}"

# --- judge (README: replace exact match with an API judge) -----------------
# Set JUDGE to an API model name (e.g. gpt-4o-mini, or a DeepSeek/Gemini-compatible
# model served via JUDGE_BASE_URL + JUDGE_KEY). Leave empty to use VLMEvalKit defaults.
JUDGE="${JUDGE:-}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-}"
JUDGE_KEY="${JUDGE_KEY:-}"

# --- output dir (kept distinct per subset so subset/full results don't collide) ---
case "$SUBSET" in
  full)   WORK_DIR="${WORK_DIR:-$REPO_DIR/eval_outputs/full}";;
  head)   WORK_DIR="${WORK_DIR:-$REPO_DIR/eval_outputs/head_${FRAC:-$N}}";;
  random) WORK_DIR="${WORK_DIR:-$REPO_DIR/eval_outputs/random_${FRAC:-$N}_seed${SEED}}";;
  *) echo "[run-eval] ERROR: SUBSET must be full|head|random (got '$SUBSET')" >&2; exit 1;;
esac

if [ ! -d "$MODEL_PATH" ]; then
  echo "[run-eval] ERROR: MODEL_PATH '$MODEL_PATH' not found. Run 01_download_model.sh first." >&2
  exit 1
fi
if [ ! -f "$EVAL_DIR/run_monet.py" ]; then
  echo "[run-eval] ERROR: VLMEvalKit not set up. Run 03_setup_eval.sh first." >&2
  exit 1
fi
if [ "$SUBSET" != "full" ]; then
  if { [ -n "$FRAC" ] && [ -n "$N" ]; } || { [ -z "$FRAC" ] && [ -z "$N" ]; }; then
    echo "[run-eval] ERROR: for SUBSET=$SUBSET set exactly one of FRAC or N." >&2; exit 1
  fi
fi

cd "$EVAL_DIR"
# Make sitecustomize.py importable from the very first interpreter byte (parent + workers).
export PYTHONPATH="$EVAL_DIR:${PYTHONPATH:-}"

echo "[run-eval] model=$MODEL_PATH  datasets='$DATASETS'  subset=$SUBSET  frac=${FRAC:-} n=${N:-} seed=$SEED"
echo "[run-eval] work-dir=$WORK_DIR"

# --- prepare each dataset's TSV (restore to full, then subset if requested) -
for d in $DATASETS; do
  python "$REPO_DIR/run_scripts/eval_subset.py" --dataset "$d" --mode restore
  if [ "$SUBSET" != "full" ]; then
    SIZE_ARGS=(); [ -n "$FRAC" ] && SIZE_ARGS+=(--frac "$FRAC"); [ -n "$N" ] && SIZE_ARGS+=(--n "$N")
    python "$REPO_DIR/run_scripts/eval_subset.py" --dataset "$d" --mode "$SUBSET" --seed "$SEED" "${SIZE_ARGS[@]}"
  fi
done

# --- run evaluation --------------------------------------------------------
JUDGE_ARGS=()
if [ -n "$JUDGE" ]; then
  JUDGE_ARGS+=(--judge "$JUDGE")
  [ -n "$JUDGE_BASE_URL" ] && JUDGE_ARGS+=(--judge-base-url "$JUDGE_BASE_URL")
  [ -n "$JUDGE_KEY" ]      && JUDGE_ARGS+=(--judge-key "$JUDGE_KEY")
else
  echo "[run-eval] WARNING: no JUDGE set. README recommends an API judge for accurate scoring."
fi

python run_monet.py \
  --model Monet \
  --data $DATASETS \
  --work-dir "$WORK_DIR" \
  --reuse \
  "${JUDGE_ARGS[@]}"

echo
echo "[run-eval] DONE. Results under: $WORK_DIR/Monet/"
