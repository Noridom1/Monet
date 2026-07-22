#!/usr/bin/env bash
# Set up VLMEvalKit for evaluating Monet-7B with latent reasoning.
#
# What it does (all per README.md "Evaluation" section, with two fixes):
#   1. Clones VLMEvalKit and installs it into the `monet` conda env.
#      (The eval env needs vllm==0.10.0 — the same pin as inference — so we reuse `monet`.)
#   2. Drops the Monet vLLM runner into VLMEvalKit/Monet_models/ as an importable package.
#   3. Writes `sitecustomize.py` (NOTE: README calls it `sitecustomized.py`, which Python
#      will NOT auto-import — the correct name is `sitecustomize.py`, no "d"). It runs in
#      every Python process (parent + vLLM workers) and swaps in the Monet GPU model runner.
#   4. Writes `run_monet.py`, a thin wrapper around VLMEvalKit's run.py that
#        - registers "Monet" as a model (Qwen2VLChat + use_vllm + the required system prompt), and
#        - disables VLMEvalKit's md5-triggered TSV re-download, so locally-subsetted datasets
#          (see eval_subset.py / run.sh) are honored instead of being re-downloaded.
#
# Run once. Safe to re-run.
set -euo pipefail

ENV_NAME="${ENV_NAME:-monet}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EVAL_DIR="${EVAL_DIR:-$REPO_DIR/VLMEvalKit}"
VLMEVALKIT_REF="${VLMEVALKIT_REF:-3fb3b5b}"

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# --- 1. clone or reuse the pinned external checkout ------------------------
if [ ! -d "$EVAL_DIR/.git" ]; then
  echo "[setup-eval] cloning VLMEvalKit -> $EVAL_DIR"
  git clone https://github.com/open-compass/VLMEvalKit.git "$EVAL_DIR"
  git -C "$EVAL_DIR" checkout --detach "$VLMEVALKIT_REF"
else
  echo "[setup-eval] VLMEvalKit already present at $EVAL_DIR, reusing"
fi

actual_ref="$(git -C "$EVAL_DIR" rev-parse HEAD)"
expected_ref="$(git -C "$EVAL_DIR" rev-parse "$VLMEVALKIT_REF^{commit}")"
if [ "$actual_ref" != "$expected_ref" ]; then
  echo "[setup-eval] ERROR: VLMEvalKit is at $actual_ref, expected $VLMEVALKIT_REF" >&2
  echo "[setup-eval] Set VLMEVALKIT_REF explicitly or initialize a clean checkout." >&2
  exit 1
fi

# --- 2. install VLMEvalKit, then restore the inference pins -----------------
echo "[setup-eval] pip install -e VLMEvalKit"
pip install -e "$EVAL_DIR"
# VLMEvalKit's deps may bump vllm/transformers; the Monet runner requires these exact pins.
echo "[setup-eval] restoring required pins and evaluation extras"
pip install "vllm==0.10.0" "transformers==4.54.0" qwen-vl-utils rouge-score

# --- 3. drop the Monet runner in as an importable package ------------------
mkdir -p "$EVAL_DIR/Monet_models"
cp "$REPO_DIR/inference/vllm/monet_gpu_model_runner.py" "$EVAL_DIR/Monet_models/"
cp "$REPO_DIR/inference/vllm/latent_policy_logits.py" "$EVAL_DIR/Monet_models/"
: > "$EVAL_DIR/Monet_models/__init__.py"   # make it a package
echo "[setup-eval] copied Monet runner and latent-policy logits helper -> Monet_models/"

# Apply checked patches so upstream drift fails visibly instead of silently
# changing which code receives the Monet integration.
apply_patch_once() {
  local patch_file="$1"
  if git -C "$EVAL_DIR" apply --reverse --check "$patch_file" >/dev/null 2>&1; then
    echo "[setup-eval] patch already applied: $(basename "$patch_file")"
  elif git -C "$EVAL_DIR" apply --check "$patch_file" >/dev/null 2>&1; then
    git -C "$EVAL_DIR" apply "$patch_file"
    echo "[setup-eval] applied: $(basename "$patch_file")"
  else
    echo "[setup-eval] ERROR: cannot apply $(basename "$patch_file")" >&2
    exit 1
  fi
}
apply_patch_once "$REPO_DIR/run_scripts/evaluation/vlmeval_sample_context.patch"
apply_patch_once "$REPO_DIR/run_scripts/evaluation/vlmeval_api_diagnostics.patch"

# --- 4a. sitecustomize.py (runs in EVERY python process; keep it minimal) --
cat > "$EVAL_DIR/sitecustomize.py" <<'PYCODE'
# sitecustomize.py (top-level) — Python auto-imports this at interpreter startup,
# in the parent process AND every spawned vLLM worker. Keep it lean: just env + the
# runner swap. (Heavier customizations live in run_monet.py, which runs parent-only.)
import os, sys, importlib

os.environ["VLLM_USE_V1"] = "1"          # force the V1 engine (Monet runner is V1)
os.environ["VLLM_NO_USAGE_STATS"] = "1"
workspace = os.path.abspath(".")
old_path = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = f"{workspace}:{old_path}" if old_path else workspace
os.environ["LATENT_START_ID"] = "151666"   # <abs_vis_token>
os.environ["LATENT_END_ID"] = "151667"      # </abs_vis_token>
# Replace stock vLLM GPU model runner with the Monet one: on emitting 151666 the
# decoder switches to latent mode for LATENT_SIZE steps.
sys.modules["vllm.v1.worker.gpu_model_runner"] = importlib.import_module(
    "Monet_models.monet_gpu_model_runner"
)
PYCODE
echo "[setup-eval] wrote sitecustomize.py"

# --- 4b. parent-only adapter and helpers -----------------------------------
cp "$REPO_DIR/run_scripts/evaluation/vlmeval_run_monet.py" "$EVAL_DIR/run_monet.py"
cp "$REPO_DIR/run_scripts/evaluation/latent_activation.py" "$EVAL_DIR/latent_activation.py"
cp "$REPO_DIR/run_scripts/evaluation/latent_policy.py" "$EVAL_DIR/latent_policy.py"
cp "$REPO_DIR/run_scripts/evaluation/secret_redaction.py" "$EVAL_DIR/secret_redaction.py"
echo "[setup-eval] copied tracked run_monet.py and policy helpers"

echo
echo "[setup-eval] DONE. VLMEvalKit ready at: $EVAL_DIR"
echo "        Next: MODEL_PATH=... bash run_scripts/evaluation/run.sh"
