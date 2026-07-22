#!/usr/bin/env bash
# Mount Drive and restore (or create) the four-dataset VLMEvalKit bundle.
set -euo pipefail

SESSION="monet"
TIMEOUT="7200"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

usage() {
  cat <<'EOF'
Usage: bash run_scripts/evaluation/colab/prepare_data.sh [options]

Options:
  --session NAME     Existing Colab session name (default: monet)
  --timeout SECONDS  Remote execution timeout (default: 7200)
  -h, --help         Show this help

Mounts Google Drive at /content/drive and uses this persistent archive:
  /content/drive/MyDrive/Monet/monet_eval_datasets.zip

If the archive exists, its manifest is verified while it is restored to
/content/LMUData. If it is absent, the four Table 3 datasets are downloaded
directly into a new archive first. This operation does not install Conda or
download model weights.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --session)
      [ "$#" -ge 2 ] || { echo "ERROR: --session requires a value" >&2; exit 2; }
      SESSION="$2"
      shift 2
      ;;
    --timeout)
      [ "$#" -ge 2 ] || { echo "ERROR: --timeout requires a value" >&2; exit 2; }
      TIMEOUT="$2"
      shift 2
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
[[ "$TIMEOUT" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: --timeout must be a positive integer." >&2
  exit 2
}

echo "[colab] checking session '$SESSION'"
colab status -s "$SESSION"
echo "[colab] mounting Google Drive at /content/drive"
colab drivemount -s "$SESSION" /content/drive

echo "[colab] preparing the four VLMEvalKit datasets (no Conda or model required)"
OUTPUT="$(colab exec \
  -s "$SESSION" \
  -f "$REPO_DIR/run_scripts/evaluation/colab/setup_eval_data.py" \
  --timeout "$TIMEOUT")"
printf '%s\n' "$OUTPUT"
if ! grep -Fq "[monet-colab] EVAL_DATA_READY" <<<"$OUTPUT"; then
  echo "ERROR: remote evaluation-data preparation did not complete successfully." >&2
  exit 1
fi

echo
echo "[colab] evaluation data is ready at /content/LMUData"
echo "[colab] persistent archive: /content/drive/MyDrive/Monet/monet_eval_datasets.zip"
echo "[colab] use: export LMUData=/content/LMUData"
