"""Encode Monet latent activation without confusing MCQ answer extractors."""

from __future__ import annotations

import re

LATENT_START = "<abs_vis_token>"
LATENT_END = "</abs_vis_token>"
ACTIVATION_PATTERN = re.compile(r"<ltnt:(\d+)>")


def annotate_latent_response(response: str, block_count: int | None = None) -> str:
    """Remove latent control tokens and append an evaluator-safe block count."""
    if block_count is None:
        block_count = response.count(LATENT_START)
    if block_count < 0:
        raise ValueError("block_count must be non-negative")
    cleaned = response.replace(LATENT_START, "").replace(LATENT_END, "").rstrip()
    return f"{cleaned}\n<ltnt:{block_count}>"


def latent_block_count(response: str) -> int | None:
    """Return the captured block count, or None for stale/unannotated output."""
    matches = ACTIVATION_PATTERN.findall(response)
    if len(matches) != 1:
        return None
    return int(matches[0])
