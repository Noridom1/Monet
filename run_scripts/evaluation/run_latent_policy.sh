#!/usr/bin/env bash
# Run deterministic forced-latent counterfactuals and summarize them.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_NAME="${ENV_NAME:-monet}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

source "$REPO_DIR/run_scripts/evaluation/eval_env.sh"

X_VALUES="${X_VALUES:-15 25}"
FORCE_SEED="${FORCE_SEED:-0}"
DATASETS="${DATASETS:-VStarBench}"
LATENT_SIZE="${LATENT_SIZE:-16}"
SUPPRESS_UNSELECTED="${SUPPRESS_UNSELECTED:-0}"
MODEL_PATH="${MODEL_PATH:-$REPO_DIR/models/Monet-7B}"
MONET_MAX_NEW_TOKENS="${MONET_MAX_NEW_TOKENS:-2048}"
MONET_MAX_PIXELS="${MONET_MAX_PIXELS:-1003520}"
MONET_SYSTEM_PROMPT="${MONET_SYSTEM_PROMPT:-}"
if [ -z "$MONET_SYSTEM_PROMPT" ]; then
  MONET_SYSTEM_PROMPT='You are a helpful multimodal assistant. You are required to answer the question based on the image provided. Put your final answer in \boxed{}.'
fi
BASELINE_RESULT="${BASELINE_RESULT:-$REPO_DIR/eval_outputs/table3_vstar/latent_16/Monet/T20260705-122733/Monet_VStarBench_nvidia_nemotron-3-nano-omni-30b-a3b-reasoning_result.xlsx}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_DIR/eval_outputs/vstar_latent_force}"
JUDGE="${JUDGE:-${JUDGE_MODEL:-}}"

if [ "$DATASETS" != "VStarBench" ]; then
  echo "[latent-policy] ERROR: this paired launcher currently requires DATASETS=VStarBench" >&2
  exit 1
fi
if [ ! -f "$BASELINE_RESULT" ]; then
  echo "[latent-policy] ERROR: baseline result not found: $BASELINE_RESULT" >&2
  exit 1
fi
if [ "$SUPPRESS_UNSELECTED" != "0" ] && [ "$SUPPRESS_UNSELECTED" != "1" ]; then
  echo "[latent-policy] ERROR: SUPPRESS_UNSELECTED must be 0 or 1" >&2
  exit 1
fi

CREATE_MODE_ARGS=()
if [ "$SUPPRESS_UNSELECTED" = "1" ]; then
  CREATE_MODE_ARGS+=(--suppress-unselected)
fi

RUN_DIRS=()
for x in $X_VALUES; do
  x_label="${x//./p}"
  run_dir="$OUTPUT_ROOT/x${x_label}_seed${FORCE_SEED}"
  manifest="$run_dir/policy_manifest.json"

  python "$REPO_DIR/run_scripts/evaluation/latent_policy.py" create \
    --dataset VStarBench \
    --indices-file "$BASELINE_RESULT" \
    --x-percent "$x" \
    --seed "$FORCE_SEED" \
    --output "$manifest" \
    --model-path "$MODEL_PATH" \
    --latent-size "$LATENT_SIZE" \
    --max-new-tokens "$MONET_MAX_NEW_TOKENS" \
    --max-pixels "$MONET_MAX_PIXELS" \
    --system-prompt "$MONET_SYSTEM_PROMPT" \
    "${CREATE_MODE_ARGS[@]}"

  DATASETS=VStarBench \
  SUBSET=full \
  WORK_DIR="$run_dir" \
  MODEL_PATH="$MODEL_PATH" \
  LATENT_SIZE="$LATENT_SIZE" \
  MONET_MAX_NEW_TOKENS="$MONET_MAX_NEW_TOKENS" \
  MONET_MAX_PIXELS="$MONET_MAX_PIXELS" \
  MONET_SYSTEM_PROMPT="$MONET_SYSTEM_PROMPT" \
  MONET_LATENT_POLICY_MANIFEST="$manifest" \
  JUDGE="$JUDGE" \
  JUDGE_BASE_URL="${JUDGE_BASE_URL:-}" \
  JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-1}" \
  JUDGE_TEMPERATURE=0 \
  JUDGE_RETRY="${JUDGE_RETRY:-}" \
  JUDGE_WAIT="${JUDGE_WAIT:-}" \
  REUSE="${REUSE:-0}" \
    bash "$REPO_DIR/run_scripts/evaluation/run.sh"
  RUN_DIRS+=(--run-dir "$run_dir")
done

python "$REPO_DIR/run_scripts/evaluation/latent_policy.py" analyze \
  --baseline "$BASELINE_RESULT" \
  --dataset VStarBench \
  --output-dir "$OUTPUT_ROOT" \
  "${RUN_DIRS[@]}"
