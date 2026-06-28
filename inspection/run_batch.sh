#!/usr/bin/env bash
# Batch latent inspection over a prepared manifest: Phase A (capture) then Phase B (analyze).
# Run on the A100 box (the local machine is insufficient for the 7B model).
# Prerequisite: bash inspection/run_prepare.sh  (writes data/inspect_samples/samples.json)
set -euo pipefail

ENV_NAME="${ENV_NAME:-monet}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"   # the Monet patch resolves paths relative to CWD

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

export LATENT_SIZE="${LATENT_SIZE:-10}"
export MODEL_PATH="${MODEL_PATH:-$REPO_DIR/models/Monet-7B}"
MANIFEST="${MANIFEST:-data/inspect_samples/samples.json}"
OUT_DIR="${OUT_DIR:-inspection/outputs/eval_samples}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_K="${TOP_K:-50}"
TOP_P="${TOP_P:-0.8}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
SEED="${SEED:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"

if [ ! -d "$MODEL_PATH" ]; then
  echo "[batch] ERROR: MODEL_PATH '$MODEL_PATH' not found. Run 01_download_model.sh first." >&2
  exit 1
fi
if [ ! -f "$MANIFEST" ]; then
  echo "[batch] ERROR: manifest '$MANIFEST' not found. Run inspection/run_prepare.sh first." >&2
  exit 1
fi

echo "[batch] MODEL_PATH=$MODEL_PATH  LATENT_SIZE=$LATENT_SIZE  MANIFEST=$MANIFEST"
echo "[batch] OUT_DIR=$OUT_DIR temperature=$TEMPERATURE top_k=$TOP_K top_p=$TOP_P repetition_penalty=$REPETITION_PENALTY seed=$SEED max_new_tokens=$MAX_NEW_TOKENS"

echo "[batch] === Phase A: capture latents per sample ==="
python -m inspection.generate_latents \
  --model_path "$MODEL_PATH" \
  --latent_size "$LATENT_SIZE" \
  --temperature "$TEMPERATURE" \
  --top_k "$TOP_K" \
  --top_p "$TOP_P" \
  --repetition_penalty "$REPETITION_PENALTY" \
  --seed "$SEED" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --manifest "$MANIFEST" \
  --out_dir "$OUT_DIR"

echo "[batch] === Phase B: logit lens + attention per sample ==="
python -m inspection.inspect \
  --model_path "$MODEL_PATH" \
  --manifest "$MANIFEST" \
  --out_dir "$OUT_DIR" \
  --topk 20

echo "[batch] === Summary: extract \\boxed{} answers + score vs gold ==="
python -m inspection.summarize_eval \
  --manifest "$MANIFEST" \
  --out_dir "$OUT_DIR" \
  --model_path "$MODEL_PATH"

echo "[batch] DONE. See $OUT_DIR/{index.md, eval_summary.json} and $OUT_DIR/<id>/report.md"
