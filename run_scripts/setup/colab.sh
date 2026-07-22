#!/usr/bin/env bash
# Prepare a Colab session end to end: source, environment, and a fresh model download.
set -euo pipefail

SESSION="monet"
GPU="A100"
REUSE=0
TIMEOUT="3600"
MODEL_REPO="NOVAglow646/Monet-7B"
MODEL_DIR="/content/Monet/models/Monet-7B"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

usage() {
  cat <<'EOF'
Usage: bash run_scripts/setup/colab.sh [options]

Options:
  --session NAME       Colab session name (default: monet)
  --gpu TYPE           Requested accelerator (default: A100)
  --reuse              Reuse an existing session before replacing its Monet source tree
  --timeout SECONDS    Timeout for each remote provisioning step (default: 3600)
  --model-repo REPO    Hugging Face model repository
  --model-dir PATH     Remote model destination
  -h, --help           Show this help

This command always runs `hf download` in the remote runtime. Evaluation kit and
dataset provisioning remain separate under run_scripts/evaluation/.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --session) [ "$#" -ge 2 ] || { echo "ERROR: --session requires a value" >&2; exit 2; }; SESSION="$2"; shift 2 ;;
    --gpu) [ "$#" -ge 2 ] || { echo "ERROR: --gpu requires a value" >&2; exit 2; }; GPU="$2"; shift 2 ;;
    --reuse) REUSE=1; shift ;;
    --timeout) [ "$#" -ge 2 ] || { echo "ERROR: --timeout requires a value" >&2; exit 2; }; TIMEOUT="$2"; shift 2 ;;
    --model-repo) [ "$#" -ge 2 ] || { echo "ERROR: --model-repo requires a value" >&2; exit 2; }; MODEL_REPO="$2"; shift 2 ;;
    --model-dir) [ "$#" -ge 2 ] || { echo "ERROR: --model-dir requires a value" >&2; exit 2; }; MODEL_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$TIMEOUT" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: --timeout must be a positive integer." >&2; exit 2; }
[[ "$MODEL_DIR" = /content/* ]] || { echo "ERROR: --model-dir must be an absolute path under /content." >&2; exit 2; }

SESSION_ARGS=(--session "$SESSION" --gpu "$GPU")
[ "$REUSE" = "1" ] && SESSION_ARGS+=(--reuse)
bash "$REPO_DIR/run_scripts/setup/colab/prepare_session.sh" "${SESSION_ARGS[@]}"
bash "$REPO_DIR/run_scripts/setup/colab/prepare_dependencies.sh" \
  --session "$SESSION" --timeout "$TIMEOUT"

MODEL_CONFIG="$(mktemp "${TMPDIR:-/tmp}/monet-model.XXXXXX.conf")"
trap 'rm -f "$MODEL_CONFIG"' EXIT
printf '%s\n%s\n' "$MODEL_REPO" "$MODEL_DIR" > "$MODEL_CONFIG"

echo "[setup-colab] uploading model-download configuration"
colab upload -s "$SESSION" "$MODEL_CONFIG" /content/monet-model-download.conf
echo "[setup-colab] downloading $MODEL_REPO -> $MODEL_DIR"
OUTPUT="$(colab exec -s "$SESSION" \
  -f "$REPO_DIR/run_scripts/setup/colab/download_model.py" \
  --timeout "$TIMEOUT")"
printf '%s\n' "$OUTPUT"
if ! grep -Fq "[monet-colab] MODEL_READY" <<<"$OUTPUT"; then
  echo "ERROR: remote model download did not complete successfully." >&2
  exit 1
fi

echo
echo "[setup-colab] DONE. Session '$SESSION' has source, environment, and model."
echo "[setup-colab] next: colab console -s $SESSION"
