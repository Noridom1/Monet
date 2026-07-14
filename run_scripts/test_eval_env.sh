#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

printf '%s\n' 'JUDGE_BASE_URL=https://from-file.invalid/v1' 'JUDGE_API_KEY=file-secret' > "$TMP_DIR/current.env"
JUDGE_BASE_URL="https://from-shell.invalid/v1"
JUDGE_API_KEY="shell-secret"
ENV_FILE="$TMP_DIR/current.env"
source "$REPO_DIR/run_scripts/eval_env.sh"
[ "$JUDGE_BASE_URL" = "https://from-shell.invalid/v1" ]
[ "$JUDGE_API_KEY" = "shell-secret" ]

unset JUDGE_BASE_URL JUDGE_API_KEY AI_PLATFORM_API_KEY
printf '%s\n' 'JUDGE_BASE_URL=https://legacy.invalid/v1' 'AI_PLATFORM_API_KEY=legacy-secret' > "$TMP_DIR/legacy.env"
ENV_FILE="$TMP_DIR/legacy.env"
source "$REPO_DIR/run_scripts/eval_env.sh" 2> "$TMP_DIR/warning.log"
[ "$JUDGE_BASE_URL" = "https://legacy.invalid/v1" ]
[ "$JUDGE_API_KEY" = "legacy-secret" ]
grep -q 'AI_PLATFORM_API_KEY is deprecated' "$TMP_DIR/warning.log"
if grep -q 'legacy-secret' "$TMP_DIR/warning.log"; then
  echo "ERROR: judge key leaked into warning output" >&2
  exit 1
fi

echo "eval env tests passed"
