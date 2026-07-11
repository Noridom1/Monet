"""Objective A: logit-lens decoding of latent hidden states.

A latent in Monet is a hidden-state vector that the model feeds back as an input
embedding. The logit lens answers: *what vocabulary tokens does this vector most
resemble?* — by projecting it through the model's final RMSNorm + ``lm_head`` and
reading the top-k softmax.

Two normalisation regimes (see ``modeling_qwen2_5_vl_monet.py:1293-1297``):
  - The captured latent vectors and the LAST entry of ``output_hidden_states`` are
    **post-norm** (``self.norm`` already applied) -> decode with ``lm_head`` directly.
  - The other ``output_hidden_states`` entries (layers 0..N-1) are **pre-norm residual
    streams** -> apply ``self.norm`` first (the standard logit-lens recipe; an un-normed
    lens is a known pitfall, see plan §5.2).
"""
from typing import List

import torch


def _norm_module(model):
    """The final RMSNorm the model uses right before ``lm_head``."""
    # text model lives at model.model.language_model (checkpoint_conversion_mapping)
    return model.model.language_model.norm


@torch.no_grad()
def logit_lens(hidden: torch.Tensor, model, k: int = 20, already_normed: bool = False):
    """Project hidden states to vocab and return (top_ids, top_probs).

    hidden: ``[N, H]``. Returns ``top_ids`` ``[N, k]`` (long, cpu) and ``top_probs``
    ``[N, k]`` (float32, cpu).
    """
    lm_head = model.lm_head
    dev = lm_head.weight.device
    h = hidden.to(device=dev, dtype=lm_head.weight.dtype)
    if not already_normed:
        h = _norm_module(model)(h)
    logits = lm_head(h)                      # [N, V]
    probs = torch.softmax(logits.float(), dim=-1)
    top_probs, top_ids = probs.topk(k, dim=-1)
    return top_ids.cpu(), top_probs.cpu()


def decode_tokens(tokenizer, ids_row: List[int]) -> List[str]:
    """Readable per-token strings (keeps leading-space markers visible)."""
    out = []
    for t in ids_row:
        s = tokenizer.decode([int(t)])
        # make whitespace-only / empty tokens visible in tables
        if s == "":
            s = tokenizer.convert_ids_to_tokens(int(t))
        out.append(s)
    return out
