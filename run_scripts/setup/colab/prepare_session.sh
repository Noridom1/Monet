#!/usr/bin/env bash
# Package this Monet checkout, create/reuse a Colab session, upload it, and unpack it.
set -euo pipefail

SESSION="monet"
GPU="A100"
CREATE_SESSION=1
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
REMOTE_ARCHIVE="/content/monet-source.tar.gz"
REMOTE_DIR="/content/Monet"

usage() {
  cat <<'EOF'
Usage: bash run_scripts/setup/colab/prepare_session.sh [options]

Options:
  --session NAME    Colab session name (default: monet)
  --gpu TYPE        Requested GPU, for example A100 or T4 (default: A100)
  --reuse           Reuse an existing session instead of creating one
  -h, --help        Show this help

The session uses Colab's standard system-memory configuration; this script does
not request a high-RAM runtime. The source archive excludes models, datasets,
outputs, caches, VLMEvalKit, and Git metadata. Model and evaluation-data
provisioning are separate workflows.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --session)
      [ "$#" -ge 2 ] || { echo "ERROR: --session requires a value" >&2; exit 2; }
      SESSION="$2"
      shift 2
      ;;
    --gpu)
      [ "$#" -ge 2 ] || { echo "ERROR: --gpu requires a value" >&2; exit 2; }
      GPU="$2"
      shift 2
      ;;
    --reuse)
      CREATE_SESSION=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

command -v colab >/dev/null 2>&1 || {
  echo "ERROR: colab CLI is not installed or is not on PATH." >&2
  exit 1
}
command -v tar >/dev/null 2>&1 || {
  echo "ERROR: tar is required to package the repository." >&2
  exit 1
}

ARCHIVE="$(mktemp "${TMPDIR:-/tmp}/monet-source.XXXXXX.tar.gz")"
trap 'rm -f "$ARCHIVE"' EXIT

echo "[colab] packaging source from $REPO_DIR"
tar -C "$REPO_DIR" \
  --exclude='./.git' \
  --exclude='./models' \
  --exclude='./data' \
  --exclude='./eval_outputs' \
  --exclude='./VLMEvalKit' \
  --exclude='./inspection/outputs' \
  --exclude='./checkpoints' \
  --exclude='./__pycache__' \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.pyo' \
  --exclude='.pytest_cache' \
  --exclude='.ruff_cache' \
  --exclude='.mypy_cache' \
  -czf "$ARCHIVE" .

echo "[colab] archive size: $(du -h "$ARCHIVE" | cut -f1)"
sha256sum "$ARCHIVE"

if [ "$CREATE_SESSION" -eq 1 ]; then
  echo "[colab] creating session '$SESSION' with requested GPU '$GPU'"
  colab new -s "$SESSION" --gpu "$GPU"
else
  echo "[colab] reusing session '$SESSION'"
  colab status -s "$SESSION"
fi

echo "[colab] uploading source archive"
colab upload -s "$SESSION" "$ARCHIVE" "$REMOTE_ARCHIVE"

echo "[colab] unpacking source into $REMOTE_DIR"
colab exec -s "$SESSION" -f "$REPO_DIR/run_scripts/setup/colab/unpack_source.py"

echo "[colab] requested accelerator: $GPU"
colab status -s "$SESSION"

echo
echo "[colab] source is ready in session '$SESSION' at $REMOTE_DIR"
echo "[colab] next: colab console -s $SESSION"
echo "[colab] stop when finished: colab stop -s $SESSION"
