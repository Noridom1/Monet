#!/usr/bin/env bash
# Run the full inspection batch pipeline once per sampling temperature.
#
# Environment variables:
#   TEMPERATURES="0.1 0.3 0.5 0.7"  space-separated temperatures to sweep
#   MANIFEST=...                       manifest passed to run_batch.sh
#   OUT_ROOT=...                       parent directory for sweep outputs
#
# All other sampling/model variables are inherited by run_batch.sh unchanged.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

TEMPERATURES="${TEMPERATURES:-0.1 0.3 0.5 0.7}"
MANIFEST="${MANIFEST:-data/inspect_samples/samples.json}"
OUT_ROOT="${OUT_ROOT:-inspection/outputs/temp_sweep}"

if [ ! -f "$MANIFEST" ]; then
  echo "[temp-sweep] ERROR: manifest '$MANIFEST' not found." >&2
  exit 1
fi

echo "[temp-sweep] MANIFEST=$MANIFEST"
echo "[temp-sweep] TEMPERATURES=$TEMPERATURES"
echo "[temp-sweep] OUT_ROOT=$OUT_ROOT"

for temperature in $TEMPERATURES; do
  # Produce filesystem-friendly names such as temperature-0p1.
  temperature_tag="${temperature//./p}"
  temperature_tag="${temperature_tag//-/m}"
  out_dir="$OUT_ROOT/temperature-$temperature_tag"

  echo "[temp-sweep] === temperature=$temperature -> $out_dir ==="
  TEMPERATURE="$temperature" \
  MANIFEST="$MANIFEST" \
  OUT_DIR="$out_dir" \
  bash inspection/run_batch.sh
done

echo "[temp-sweep] DONE. Results are under $OUT_ROOT"
