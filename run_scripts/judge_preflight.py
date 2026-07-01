#!/usr/bin/env python3
"""Fail fast unless the configured OpenAI-compatible judge is usable."""

import argparse
import os

import requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    key = os.environ.get("AI_PLATFORM_API_KEY")
    if not key:
        raise SystemExit("AI_PLATFORM_API_KEY is not set")
    url = f'{args.base_url.rstrip("/")}/chat/completions'
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": args.model, "messages": [{"role": "user", "content": "Reply with A only."}],
              "max_tokens": 2, "temperature": 0},
        timeout=args.timeout,
    )
    if not response.ok:
        raise SystemExit(f"Judge preflight failed: HTTP {response.status_code}: {response.text[:500]}")
    payload = response.json()
    if not payload.get("choices"):
        raise SystemExit("Judge preflight returned no choices")
    print("[eval] judge preflight passed")


if __name__ == "__main__":
    main()
