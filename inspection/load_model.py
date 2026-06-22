"""Load the Monet HF model + processor for inspection.

Mirrors the model/processor/config setup in ``src/main.py`` (lines ~50-125) but
configured for *inference* (use_cache, eval, cuda) rather than training. The Monet
forward patch MUST be applied before importing transformers' Qwen2.5-VL symbols,
exactly as ``src/main.py`` does via ``from monet_qwen_model import apply_qwen2_5_monet``.
"""
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from monet_qwen_model import apply_qwen2_5_monet  # noqa: F401  (applies the patch on import)
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLConfig,
    AutoProcessor,
)

# The five Monet special tokens, added in src/main.py:64-68.
MONET_SPECIAL_TOKENS = [
    "<abs_vis_token_pad>",
    "<abs_vis_token>",
    "</abs_vis_token>",
    "<observation>",
    "</observation>",
]


def load_monet(model_path: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    """Load model + processor and wire up the Monet config attributes.

    Returns ``(model, processor, special_ids)`` where ``special_ids`` is a dict of
    the resolved single-token ids (matching ``src/main.py:110-125``).
    """
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    tokenizer = processor.tokenizer

    # The released Monet checkpoint already contains these tokens; add_tokens is a
    # no-op (returns 0) if they exist. We resize only if the embedding table is
    # actually smaller than the tokenizer, to avoid clobbering trained rows.
    n_added = tokenizer.add_tokens(MONET_SPECIAL_TOKENS, special_tokens=True)

    def _single_id(tok: str) -> int:
        ids = tokenizer(tok, return_tensors="pt")["input_ids"][0]
        assert ids.numel() == 1, f"{tok!r} did not map to a single token id: {ids.tolist()}"
        return int(ids[0])

    special_ids = {
        "latent_pad": _single_id("<abs_vis_token_pad>"),   # == config.latent_token_id
        "latent_start": _single_id("<abs_vis_token>"),     # 151666
        "latent_end": _single_id("</abs_vis_token>"),      # 151667
        "obs_start": _single_id("<observation>"),
        "obs_end": _single_id("</observation>"),
        "image_pad": _single_id("<|image_pad|>"),          # == config.image_token_id
        "vision_start": _single_id("<|vision_start|>"),
        "vision_end": _single_id("<|vision_end|>"),
    }
    special_ids["answer_start_pattern"] = tokenizer(
        "<|im_start|>assistant", return_tensors="pt"
    )["input_ids"][0].tolist()

    config = Qwen2_5_VLConfig.from_pretrained(model_path)
    config.use_cache = True  # inference: we want the KV cache for autoregressive decode

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        config=config,
        torch_dtype=dtype,
    )

    # Resize embeddings only if the checkpoint really lacks the new rows.
    if n_added > 0 and model.get_input_embeddings().weight.shape[0] < len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    # Wire the config attributes the Monet forward relies on (src/main.py:122-125).
    model.config.latent_token_id = special_ids["latent_pad"]
    model.config.latent_start_id = special_ids["latent_start"]
    model.config.latent_end_id = special_ids["latent_end"]
    model.config.answer_start_pattern = special_ids["answer_start_pattern"]
    # image_token_id / vision ids come from the base config already.

    model = model.to(device=device, dtype=dtype).eval()
    return model, processor, special_ids
