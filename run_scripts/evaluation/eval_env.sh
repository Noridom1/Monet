#!/usr/bin/env bash
# Load local judge settings while preserving values explicitly exported by the caller.

ENV_FILE="${ENV_FILE:-$REPO_DIR/.env}"
_shell_judge_base_url_set="${JUDGE_BASE_URL+x}"; _shell_judge_base_url="${JUDGE_BASE_URL:-}"
_shell_judge_api_key_set="${JUDGE_API_KEY+x}"; _shell_judge_api_key="${JUDGE_API_KEY:-}"
_shell_legacy_key_set="${AI_PLATFORM_API_KEY+x}"; _shell_legacy_key="${AI_PLATFORM_API_KEY:-}"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi
[ -n "$_shell_judge_base_url_set" ] && JUDGE_BASE_URL="$_shell_judge_base_url"
[ -n "$_shell_judge_api_key_set" ] && JUDGE_API_KEY="$_shell_judge_api_key"
[ -n "$_shell_legacy_key_set" ] && AI_PLATFORM_API_KEY="$_shell_legacy_key"
if [ -z "${JUDGE_API_KEY:-}" ] && [ -n "${AI_PLATFORM_API_KEY:-}" ]; then
  JUDGE_API_KEY="$AI_PLATFORM_API_KEY"
  echo "[eval] warning: AI_PLATFORM_API_KEY is deprecated; use JUDGE_API_KEY" >&2
fi
export JUDGE_BASE_URL JUDGE_API_KEY
unset _shell_judge_base_url_set _shell_judge_base_url _shell_judge_api_key_set _shell_judge_api_key
unset _shell_legacy_key_set _shell_legacy_key
