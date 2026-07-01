"""Process-wide requests rate limiter for VLMEvalKit judge calls."""

from __future__ import annotations

import email.utils
import atexit
import json
import os
import random
import threading
import time
from datetime import datetime, timezone

import requests

_LOCK = threading.Lock()
_NEXT_REQUEST = 0.0
_ORIGINAL_REQUEST = requests.sessions.Session.request
_STATS = {"http_requests": 0, "retries": 0, "rate_limit_responses": 0, "server_error_responses": 0}


def _write_stats() -> None:
    path = os.environ.get("JUDGE_STATS_FILE")
    if path:
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(_STATS, stream, indent=2, sort_keys=True)
            stream.write("\n")


def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            parsed = email.utils.parsedate_to_datetime(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def install(rpm: float | None = None, max_retries: int | None = None) -> None:
    """Patch requests once with a global start-rate limiter and transient retries."""
    rpm = float(rpm if rpm is not None else os.environ.get("JUDGE_RPM", "6"))
    max_retries = int(max_retries if max_retries is not None else os.environ.get("JUDGE_HTTP_RETRIES", "6"))
    if rpm <= 0:
        raise ValueError("JUDGE_RPM must be greater than zero")
    if getattr(requests.sessions.Session.request, "_monet_rate_limited", False):
        return
    interval = 60.0 / rpm
    atexit.register(_write_stats)

    def limited_request(self, method, url, **kwargs):
        global _NEXT_REQUEST
        for attempt in range(max_retries + 1):
            with _LOCK:
                now = time.monotonic()
                wait = max(0.0, _NEXT_REQUEST - now)
                _NEXT_REQUEST = max(now, _NEXT_REQUEST) + interval
            if wait:
                time.sleep(wait)
            response = _ORIGINAL_REQUEST(self, method, url, **kwargs)
            with _LOCK:
                _STATS["http_requests"] += 1
                if response.status_code == 429:
                    _STATS["rate_limit_responses"] += 1
                elif response.status_code in {500, 502, 503, 504}:
                    _STATS["server_error_responses"] += 1
            if response.status_code not in {429, 500, 502, 503, 504} or attempt == max_retries:
                return response
            with _LOCK:
                _STATS["retries"] += 1
            retry_after = _retry_after_seconds(response.headers.get("Retry-After"))
            delay = retry_after if retry_after is not None else min(60.0, 2**attempt) + random.uniform(0.0, 0.5)
            time.sleep(delay)
        raise AssertionError("unreachable")

    limited_request._monet_rate_limited = True
    requests.sessions.Session.request = limited_request
