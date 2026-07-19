#!/usr/bin/env bash
# Force latent thinking only on natural-inactive, incorrect VStarBench baseline samples.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-monet}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

source "$REPO_DIR/run_scripts/eval_env.sh"

DATASETS="${DATASETS:-VStarBench}"
LATENT_SIZE="${LATENT_SIZE:-16}"
MODEL_PATH="${MODEL_PATH:-$REPO_DIR/models/Monet-7B}"
MONET_MAX_NEW_TOKENS="${MONET_MAX_NEW_TOKENS:-2048}"
MONET_MAX_PIXELS="${MONET_MAX_PIXELS:-1003520}"
MONET_SYSTEM_PROMPT="${MONET_SYSTEM_PROMPT:-}"
if [ -z "$MONET_SYSTEM_PROMPT" ]; then
  MONET_SYSTEM_PROMPT='You are a helpful multimodal assistant. You are required to answer the question based on the image provided. Put your final answer in \boxed{}.'
fi
BASELINE_RESULT="${BASELINE_RESULT:-$REPO_DIR/eval_outputs/table3_vstar/latent_16/Monet/T20260705-122733/Monet_VStarBench_nvidia_nemotron-3-nano-omni-30b-a3b-reasoning_result.xlsx}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_DIR/eval_outputs/vstar_latent_rescue/natural_inactive_wrong}"
JUDGE="${JUDGE:-${JUDGE_MODEL:-}}"

if [ "$DATASETS" != "VStarBench" ]; then
  echo "[latent-rescue] ERROR: this launcher requires DATASETS=VStarBench" >&2
  exit 1
fi
if [ ! -f "$BASELINE_RESULT" ]; then
  echo "[latent-rescue] ERROR: baseline result not found: $BASELINE_RESULT" >&2
  exit 1
fi
if [ -z "$JUDGE" ]; then
  echo "[latent-rescue] ERROR: set JUDGE to the same judge used for the baseline" >&2
  exit 1
fi

RUN_DIR="$OUTPUT_ROOT/run"
TARGETS_FILE="$OUTPUT_ROOT/target_samples.csv"
MANIFEST="$RUN_DIR/policy_manifest.json"

python "$REPO_DIR/run_scripts/latent_policy.py" create-rescue \
  --baseline "$BASELINE_RESULT" \
  --dataset VStarBench \
  --targets-output "$TARGETS_FILE" \
  --output "$MANIFEST" \
  --model-path "$MODEL_PATH" \
  --latent-size "$LATENT_SIZE" \
  --max-new-tokens "$MONET_MAX_NEW_TOKENS" \
  --max-pixels "$MONET_MAX_PIXELS" \
  --system-prompt "$MONET_SYSTEM_PROMPT"

DATASETS=VStarBench \
SUBSET=indices \
INDICES_FILE="$TARGETS_FILE" \
WORK_DIR="$RUN_DIR" \
MODEL_PATH="$MODEL_PATH" \
LATENT_SIZE="$LATENT_SIZE" \
MONET_MAX_NEW_TOKENS="$MONET_MAX_NEW_TOKENS" \
MONET_MAX_PIXELS="$MONET_MAX_PIXELS" \
MONET_SYSTEM_PROMPT="$MONET_SYSTEM_PROMPT" \
MONET_LATENT_POLICY_MANIFEST="$MANIFEST" \
JUDGE="$JUDGE" \
JUDGE_BASE_URL="${JUDGE_BASE_URL:-}" \
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-1}" \
JUDGE_TEMPERATURE=0 \
JUDGE_RETRY="${JUDGE_RETRY:-}" \
JUDGE_WAIT="${JUDGE_WAIT:-}" \
REUSE=0 \
  bash "$REPO_DIR/run_scripts/04_run_eval.sh"

python "$REPO_DIR/run_scripts/latent_policy.py" analyze-rescue \
  --baseline "$BASELINE_RESULT" \
  --targets "$TARGETS_FILE" \
  --run-dir "$RUN_DIR" \
  --output-dir "$OUTPUT_ROOT" \
  --dataset VStarBench

echo "[latent-rescue] DONE. Results under: $OUTPUT_ROOT"
