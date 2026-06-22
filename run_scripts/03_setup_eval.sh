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
#          (see eval_subset.py / 04_run_eval.sh) are honored instead of being re-downloaded.
#
# Run once. Safe to re-run.
set -euo pipefail

ENV_NAME="${ENV_NAME:-monet}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_DIR="${EVAL_DIR:-$REPO_DIR/VLMEvalKit}"

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# --- 1. clone VLMEvalKit ---------------------------------------------------
if [ ! -d "$EVAL_DIR/.git" ]; then
  echo "[setup-eval] cloning VLMEvalKit -> $EVAL_DIR"
  git clone https://github.com/open-compass/VLMEvalKit.git "$EVAL_DIR"
else
  echo "[setup-eval] VLMEvalKit already present at $EVAL_DIR, reusing"
fi

# --- 2. install VLMEvalKit, then restore the inference pins -----------------
echo "[setup-eval] pip install -e VLMEvalKit"
pip install -e "$EVAL_DIR"
# VLMEvalKit's deps may bump vllm/transformers; the Monet runner requires these exact pins.
echo "[setup-eval] restoring required pins (vllm==0.10.0, transformers==4.54.0)"
pip install "vllm==0.10.0" "transformers==4.54.0" qwen-vl-utils

# --- 3. drop the Monet runner in as an importable package ------------------
mkdir -p "$EVAL_DIR/Monet_models"
cp "$REPO_DIR/inference/vllm/monet_gpu_model_runner.py" "$EVAL_DIR/Monet_models/"
: > "$EVAL_DIR/Monet_models/__init__.py"   # make it a package
echo "[setup-eval] copied monet_gpu_model_runner.py -> Monet_models/"

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

# --- 4b. run_monet.py wrapper (parent-only: register model + patch md5) ----
cat > "$EVAL_DIR/run_monet.py" <<'PYCODE'
#!/usr/bin/env python
"""Thin wrapper around VLMEvalKit's run.py for evaluating Monet-7B.

Run it exactly like run.py, e.g.:
    python run_monet.py --model Monet --data MMBench_DEV_EN --work-dir outputs

It does two parent-process-only things before delegating to run.py:
  1. Registers "Monet" in vlmeval's supported_VLM (Qwen2.5-VL + vLLM + the required
     Monet system prompt), so you can pass `--model Monet` and use plain official
     dataset names with their official evaluators.
  2. Disables VLMEvalKit's md5-triggered TSV re-download, so a locally-subsetted
     dataset TSV is used as-is instead of being re-downloaded over the top.

Config via env vars (set by 04_run_eval.sh):
  MODEL_PATH               path to the Monet model (required)
  LATENT_SIZE              latent embeddings per block (read by the runner; default 10)
  MONET_SYSTEM_PROMPT      system prompt (defaults to the README-recommended one)
  MONET_MAX_NEW_TOKENS     max new tokens (default 2048)
  MONET_MAX_PIXELS         max image pixels for the processor (default 1280*28*28)
"""
import os
import sys
import runpy
from functools import partial

# 1) honor local (possibly subsetted) TSVs: never re-download on md5 mismatch.
from vlmeval.dataset import image_base as _ib
_orig_prepare_tsv = _ib.ImageBaseDataset.prepare_tsv
def _prepare_tsv_no_md5(self, url, file_md5=None):
    return _orig_prepare_tsv(self, url, None)
_ib.ImageBaseDataset.prepare_tsv = _prepare_tsv_no_md5

# 2) register the Monet model.
from vlmeval import config as _cfg
from vlmeval import vlm as _vlm

_MODEL_PATH = os.environ.get("MODEL_PATH")
assert _MODEL_PATH, "MODEL_PATH env var must point to the Monet model directory"
_SYSTEM_PROMPT = os.environ.get(
    "MONET_SYSTEM_PROMPT",
    "You are a helpful multimodal assistant. You are required to answer the "
    "question based on the image provided. Put your final answer in \\boxed{}.",
)
_cfg.supported_VLM["Monet"] = partial(
    _vlm.Qwen2VLChat,
    model_path=_MODEL_PATH,
    use_vllm=True,
    system_prompt=_SYSTEM_PROMPT,
    post_process=True,  # extract the final \boxed{...} answer
    max_new_tokens=int(os.environ.get("MONET_MAX_NEW_TOKENS", "2048")),
    max_pixels=int(os.environ.get("MONET_MAX_PIXELS", str(1280 * 28 * 28))),
)

# 3) hand off to the real run.py (it does `from vlmeval.config import supported_VLM`,
#    which is the same dict object we just edited).
_here = os.path.dirname(os.path.abspath(__file__))
sys.argv[0] = os.path.join(_here, "run.py")
runpy.run_path(sys.argv[0], run_name="__main__")
PYCODE
echo "[setup-eval] wrote run_monet.py"

echo
echo "[setup-eval] DONE. VLMEvalKit ready at: $EVAL_DIR"
echo "        Next: MODEL_PATH=... bash run_scripts/04_run_eval.sh"
