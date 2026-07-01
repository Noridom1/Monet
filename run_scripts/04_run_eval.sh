#!/usr/bin/env bash
# Monet Table 3 evaluation. MODE=infer, score, all, or summarize.
set -euo pipefail

ENV_NAME="${ENV_NAME:-monet}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_DIR="${EVAL_DIR:-$REPO_DIR/VLMEvalKit}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

MODEL_PATH="${MODEL_PATH:-$REPO_DIR/models/Monet-7B}"
MODEL_PATH="$(realpath -m "$MODEL_PATH" 2>/dev/null || readlink -f "$MODEL_PATH")"
export MODEL_PATH
export MONET_MAX_NEW_TOKENS="${MONET_MAX_NEW_TOKENS:-2048}"
export MONET_MAX_PIXELS="${MONET_MAX_PIXELS:-$((1280 * 28 * 28))}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"

DATASETS="${DATASETS:-VStarBench HRBench4K HRBench8K MME-RealWorld-Lite}"
LATENT_SIZES="${LATENT_SIZES:-${LATENT_SIZE:-10}}"
MODE="${MODE:-all}"                    # infer | score | all | summarize
SUBSET="${SUBSET:-full}"               # full | head | random (smoke tests only)
FRAC="${FRAC:-}"
N="${N:-}"
SEED="${SEED:-0}"
WORK_DIR="${WORK_DIR:-$REPO_DIR/eval_outputs/table3}"

JUDGE_MODEL="${JUDGE_MODEL:-google/gemma-4-31b-it}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-https://maas-llm-aiplatform-hcm.api.vngcloud.vn/v1}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-1}"
export JUDGE_MODEL JUDGE_BASE_URL JUDGE_CONCURRENCY
export JUDGE_RPM="${JUDGE_RPM:-6}"
export JUDGE_HTTP_RETRIES="${JUDGE_HTTP_RETRIES:-6}"

case "$MODE" in infer|score|all|summarize) ;; *) echo "ERROR: MODE must be infer|score|all|summarize" >&2; exit 2;; esac
if [ ! -f "$EVAL_DIR/run_monet.py" ] && [ "$MODE" != summarize ]; then
  echo "ERROR: VLMEvalKit is not set up. Run 03_setup_eval.sh first." >&2; exit 1
fi
if [ ! -d "$MODEL_PATH" ] && { [ "$MODE" = infer ] || [ "$MODE" = all ]; }; then
  echo "ERROR: MODEL_PATH '$MODEL_PATH' does not exist." >&2; exit 1
fi
if [ "$SUBSET" != full ] && { { [ -n "$FRAC" ] && [ -n "$N" ]; } || { [ -z "$FRAC" ] && [ -z "$N" ]; }; }; then
  echo "ERROR: for a subset, set exactly one of FRAC or N." >&2; exit 2
fi

read -ra DATASET_ARRAY <<< "$DATASETS"
read -ra LATENT_ARRAY <<< "$LATENT_SIZES"
for size in "${LATENT_ARRAY[@]}"; do
  [[ "$size" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid latent size '$size'" >&2; exit 2; }
done

if [ "$MODE" = summarize ]; then
  python "$REPO_DIR/run_scripts/summarize_table3.py" --work-dir "$WORK_DIR" \
    --latent-sizes "${LATENT_ARRAY[@]}" --datasets "${DATASET_ARRAY[@]}"
  exit 0
fi

cd "$EVAL_DIR"
export PYTHONPATH="$EVAL_DIR:$REPO_DIR/run_scripts:${PYTHONPATH:-}"

if [ "$MODE" = score ] || [ "$MODE" = all ]; then
  : "${AI_PLATFORM_API_KEY:?Set AI_PLATFORM_API_KEY for MODE=$MODE}"
  python "$REPO_DIR/run_scripts/judge_preflight.py" --base-url "$JUDGE_BASE_URL" --model "$JUDGE_MODEL"
fi

# Download/restore canonical datasets once. Subsets are explicitly smoke-test artifacts.
for dataset in "${DATASET_ARRAY[@]}"; do
  python "$REPO_DIR/run_scripts/eval_subset.py" --dataset "$dataset" --mode restore
  if [ "$SUBSET" != full ]; then
    SIZE_ARGS=(); [ -n "$FRAC" ] && SIZE_ARGS+=(--frac "$FRAC"); [ -n "$N" ] && SIZE_ARGS+=(--n "$N")
    python "$REPO_DIR/run_scripts/eval_subset.py" --dataset "$dataset" --mode "$SUBSET" --seed "$SEED" "${SIZE_ARGS[@]}"
  fi
done

for size in "${LATENT_ARRAY[@]}"; do
  export LATENT_SIZE="$size"
  RUN_DIR="$WORK_DIR/latent_$size"
  mkdir -p "$RUN_DIR"
  python "$REPO_DIR/run_scripts/record_eval_metadata.py" --output "$RUN_DIR/run_config.json" \
    --eval-dir "$EVAL_DIR" --model-path "$MODEL_PATH" --latent-size "$size" --datasets "${DATASET_ARRAY[@]}"

  if [ "$MODE" = infer ] || [ "$MODE" = all ]; then
    echo "[eval] inference latent_size=$size datasets=$DATASETS"
    python run_monet.py --model Monet --data "${DATASET_ARRAY[@]}" --work-dir "$RUN_DIR" --mode infer --reuse
  fi

  if [ "$MODE" = score ] || [ "$MODE" = all ]; then
    echo "[eval] scoring latent_size=$size judge=$JUDGE_MODEL rpm=$JUDGE_RPM concurrency=$JUDGE_CONCURRENCY"
    MONET_RATE_LIMIT_JUDGE=1 JUDGE_STATS_FILE="$RUN_DIR/judge_http_stats.json" python run_monet.py \
      --model Monet --data "${DATASET_ARRAY[@]}" --work-dir "$RUN_DIR" --mode eval --reuse \
      --judge "$JUDGE_MODEL" --judge-base-url "$JUDGE_BASE_URL" \
      --judge-api-nproc "$JUDGE_CONCURRENCY" --judge-args '{"temperature": 0}'
  fi
done

python "$REPO_DIR/run_scripts/summarize_table3.py" --work-dir "$WORK_DIR" \
  --latent-sizes "${LATENT_ARRAY[@]}" --datasets "${DATASET_ARRAY[@]}"
