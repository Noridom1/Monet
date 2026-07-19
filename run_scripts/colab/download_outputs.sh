#!/usr/bin/env bash
# Download one remote Monet evaluation-output directory into its matching local path.
set -euo pipefail

SESSION="monet"
REMOTE_DIR=""
TIMEOUT="7200"
OVERWRITE=0
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE_ROOT="/content/Monet/eval_outputs"
LOCAL_ROOT="$REPO_DIR/eval_outputs"

usage() {
  cat <<'EOF'
Usage: bash run_scripts/colab/download_outputs.sh --remote-dir PATH [options]

Options:
  --session NAME      Existing Colab session name (default: monet)
  --remote-dir PATH   Remote output directory below /content/Monet/eval_outputs (required)
  --timeout SECONDS   Remote packaging timeout (default: 7200)
  --overwrite         Replace an existing non-empty matching local directory
  -h, --help          Show this help

Packages the requested remote output directory, downloads it, and extracts it
to the matching path in this checkout. For example:

  bash run_scripts/colab/download_outputs.sh \
    --session monet-a100 \
    --remote-dir /content/Monet/eval_outputs/table3_hrbench4k/latent_10

places the files at:

  eval_outputs/table3_hrbench4k/latent_10
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --session)
      [ "$#" -ge 2 ] || { echo "ERROR: --session requires a value" >&2; exit 2; }
      SESSION="$2"
      shift 2
      ;;
    --remote-dir)
      [ "$#" -ge 2 ] || { echo "ERROR: --remote-dir requires a value" >&2; exit 2; }
      REMOTE_DIR="${2%/}"
      shift 2
      ;;
    --timeout)
      [ "$#" -ge 2 ] || { echo "ERROR: --timeout requires a value" >&2; exit 2; }
      TIMEOUT="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE=1
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
command -v python3 >/dev/null 2>&1 || {
  echo "ERROR: python3 is required to extract the downloaded ZIP archive." >&2
  exit 1
}
[[ "$TIMEOUT" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: --timeout must be a positive integer." >&2
  exit 2
}
[[ -n "$REMOTE_DIR" ]] || {
  echo "ERROR: --remote-dir is required." >&2
  usage >&2
  exit 2
}
[[ "$REMOTE_DIR" == "$REMOTE_ROOT"/* ]] || {
  echo "ERROR: --remote-dir must be below $REMOTE_ROOT." >&2
  exit 2
}

RELATIVE_DIR="${REMOTE_DIR#"$REMOTE_ROOT"/}"
LOCAL_DIR="$LOCAL_ROOT/$RELATIVE_DIR"
LOCAL_PARENT="$(dirname "$LOCAL_DIR")"
CONFIG_FILE="$(mktemp "${TMPDIR:-/tmp}/monet-download-source.XXXXXX")"
ARCHIVE_FILE="$(mktemp "${TMPDIR:-/tmp}/monet-download.XXXXXX.zip")"
trap 'rm -f "$CONFIG_FILE" "$ARCHIVE_FILE"' EXIT

if [ -d "$LOCAL_DIR" ] && [ -n "$(find "$LOCAL_DIR" -mindepth 1 -print -quit)" ]; then
  if [ "$OVERWRITE" -eq 0 ]; then
    echo "ERROR: local destination is not empty: $LOCAL_DIR" >&2
    echo "Re-run with --overwrite to replace it." >&2
    exit 1
  fi
  rm -rf "$LOCAL_DIR"
fi

COLAB_BIN="$(command -v colab)"
COLAB_PY="$(head -n 1 "$COLAB_BIN" | sed 's/^#!//')"
if [ ! -x "$COLAB_PY" ]; then
  echo "ERROR: could not determine the Python interpreter used by colab." >&2
  exit 1
fi

echo "[colab] refreshing session '$SESSION' credentials"
"$COLAB_PY" "$REPO_DIR/run_scripts/colab/refresh_session_proxy.py" --session "$SESSION"

printf '%s\n' "$REMOTE_DIR" > "$CONFIG_FILE"
echo "[colab] packaging remote output directory: $REMOTE_DIR"
colab upload -s "$SESSION" "$CONFIG_FILE" /tmp/monet-download-source.txt
OUTPUT="$(colab exec \
  -s "$SESSION" \
  -f "$REPO_DIR/run_scripts/colab/package_remote_directory.py" \
  --timeout "$TIMEOUT")"
printf '%s\n' "$OUTPUT"
if ! grep -Fq "[monet-colab] OUTPUT_ARCHIVE_READY" <<<"$OUTPUT"; then
  echo "ERROR: remote output packaging did not complete successfully." >&2
  exit 1
fi

mkdir -p "$LOCAL_PARENT"
echo "[colab] downloading output archive"
colab download -s "$SESSION" /tmp/monet-download.zip "$ARCHIVE_FILE"
python3 -m zipfile -e "$ARCHIVE_FILE" "$LOCAL_PARENT"

echo
echo "[colab] outputs downloaded to: $LOCAL_DIR"
