"""Small, testable logits operations for Monet latent-policy inference."""

from __future__ import annotations

import torch


FORCE_FIRST_POLICY = "force_first"
SUPPRESS_LATENT_START_POLICY = "suppress_latent_start"
POLICY_EXTRA_ARG = "monet_latent_policy"


def force_token_for_row(logits: torch.Tensor, row: int, token_id: int) -> None:
    """Mutate one logits row so greedy or random sampling must choose ``token_id``."""
    if logits.ndim != 2:
        raise ValueError(f"expected rank-2 logits, got shape {tuple(logits.shape)}")
    if not 0 <= row < logits.shape[0]:
        raise IndexError(f"logits row {row} is outside [0, {logits.shape[0]})")
    if not 0 <= token_id < logits.shape[1]:
        raise ValueError(
            f"latent start token {token_id} is outside vocabulary [0, {logits.shape[1]})"
        )

    logits[row].fill_(float("-inf"))
    logits[row, token_id] = 0.0


def suppress_token_for_row(logits: torch.Tensor, row: int, token_id: int) -> None:
    """Prevent one token from being sampled without changing other logits."""
    if logits.ndim != 2:
        raise ValueError(f"expected rank-2 logits, got shape {tuple(logits.shape)}")
    if not 0 <= row < logits.shape[0]:
        raise IndexError(f"logits row {row} is outside [0, {logits.shape[0]})")
    if not 0 <= token_id < logits.shape[1]:
        raise ValueError(
            f"latent start token {token_id} is outside vocabulary [0, {logits.shape[1]})"
        )

    logits[row, token_id] = float("-inf")
