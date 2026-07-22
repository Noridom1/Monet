#!/usr/bin/env bash
# Install the Monet inference environment inside an existing Colab session.
set -euo pipefail

SESSION="monet"
TIMEOUT="3600"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

usage() {
  cat <<'EOF'
Usage: bash run_scripts/setup/colab/prepare_dependencies.sh [options]

Options:
  --session NAME     Existing Colab session name (default: monet)
  --timeout SECONDS  Remote execution timeout (default: 3600)
  -h, --help         Show this help

The remote Monet source must already exist at /content/Monet. Run
prepare_session.sh first. This installs Miniconda under /root/miniconda3 and
uses run_scripts/setup/environment.sh to create/update the pinned `monet` env.
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
echo "[colab] preparing the Monet inference environment (this can take a while)"
OUTPUT="$(colab exec \
  -s "$SESSION" \
  -f "$REPO_DIR/run_scripts/setup/colab/setup_inference_env.py" \
  --timeout "$TIMEOUT")"
printf '%s\n' "$OUTPUT"
if ! grep -Fq "[monet-colab] ENVIRONMENT_READY" <<<"$OUTPUT"; then
  echo "ERROR: remote dependency preparation did not complete successfully." >&2
  exit 1
fi

echo
echo "[colab] inference environment is ready in session '$SESSION'"
echo "[colab] environment: /root/miniconda3/envs/monet"
echo "[colab] next: download the model, then run the example inference"
