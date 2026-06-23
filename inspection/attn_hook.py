"""Phase B instrument — memory-safe attention capture (Objective B).

``output_attentions=True`` would materialise ``[layers x heads x S x S]`` (tens of GB at
VL sequence lengths). Instead we register a **forward pre-hook on each live attention
module** of the loaded model and, for a chosen set of query rows + key columns, recompute
only the sliced attention probabilities — reusing the exact memory-safe recipe already in
the model (``modeling_qwen2_5_vl_monet.py:1008-1024``):

    q_b    = query_states[b, :, QUERY_POSS, :]              # [H, Q, D]  post-RoPE
    k_rep  = repeat_kv(key_states, num_key_value_groups)    # [H, S, D]  GQA-expanded
    logits = einsum('hqd,hsd->hqs', q_b, k_rep) * scaling   # [H, Q, S]  — never S x S kept
    probs  = softmax(logits + causal_mask_rows, dim=-1)     # [H, Q, S]
    store  probs[:, :, KEY_COLS].to('cpu', float16)         # slice keys, offload, free

Peak kept memory is ``O(layers * H * Q * |KEY_COLS|)`` (megabytes), not ``O(layers*H*S^2)``.

We attach to the *instances* of the loaded model (selected by class name), so capture is
independent of any module-identity subtleties from the sys.modules patch. The pre-hook only
reads the attention inputs; it does not change the model's own computation. ``install`` and
``remove`` add / drop the hooks, leaving the model exactly as found.
"""
import torch

_apply_mrope = None
_repeat_kv = None
_ATTN_CLS_NAME = "Qwen2_5_VLAttention"

_CAPTURE = {
    "enabled": False,
    "query_poss": [],     # List[int] query (row) positions to keep
    "key_cols": None,     # List[int] key (column) positions to keep; None = keep all (costly)
    "buffer": {},         # layer_idx -> Tensor [H, Q, K] float16 on cpu
    "handles": [],        # registered hook handles
}


def _resolve_symbols():
    global _apply_mrope, _repeat_kv
    if _apply_mrope is not None:
        return
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as m
    _apply_mrope = m.apply_multimodal_rotary_pos_emb
    _repeat_kv = m.repeat_kv


def configure(query_poss, key_cols):
    """Set which query rows / key columns to capture for the next replay."""
    _CAPTURE["query_poss"] = list(query_poss)
    _CAPTURE["key_cols"] = None if key_cols is None else list(key_cols)
    _CAPTURE["buffer"] = {}


def get_buffer():
    """layer_idx -> [H, Q, K] float16 cpu tensor (Q = len(query_poss), K = len(key_cols))."""
    return _CAPTURE["buffer"]


def _extract_mask_rows(attention_mask, poss, b, S, device):
    """Additive [Q, S] bias for the selected query rows; causal fallback if mask is None."""
    am = attention_mask
    if am is not None and torch.is_tensor(am):
        if am.dim() == 4:        # [B, 1, S, S]
            rows = am[b, 0, poss, :]
        elif am.dim() == 3:      # [B, S, S]
            rows = am[b, poss, :]
        elif am.dim() == 2:      # [S, S]
            rows = am[poss, :]
        else:
            rows = None
        if rows is not None:
            if rows.dtype == torch.bool:
                rows = torch.where(rows, 0.0, float("-inf"))
            return rows.to(device=device, dtype=torch.float32)
    # Fallback: plain causal (correct for a single unpadded sequence).
    ar = torch.arange(S, device=device)[None, :]
    q_idx = torch.tensor(poss, device=device)[:, None]
    return torch.where(ar <= q_idx, 0.0, float("-inf")).to(torch.float32)


@torch.no_grad()
def _capture(attn, hidden_states, attention_mask, position_embeddings):
    bsz, q_len, _ = hidden_states.size()
    poss = _CAPTURE["query_poss"]
    if not poss:
        return
    q = attn.q_proj(hidden_states).view(bsz, q_len, -1, attn.head_dim).transpose(1, 2)
    k = attn.k_proj(hidden_states).view(bsz, q_len, -1, attn.head_dim).transpose(1, 2)
    cos, sin = position_embeddings
    q, k = _apply_mrope(q, k, cos, sin, attn.rope_scaling["mrope_section"])
    k_rep = _repeat_kv(k, attn.num_key_value_groups)        # [B, H, S, D]

    b = 0  # inspection runs batch size 1
    q_b = q[b, :, poss, :]                                  # [H, Q, D]
    logits = torch.einsum("hqd,hsd->hqs", q_b.float(), k_rep[b].float()) * attn.scaling
    mask_rows = _extract_mask_rows(attention_mask, poss, b, q_len, logits.device)
    logits = logits + mask_rows[None, :, :]                 # broadcast over heads
    probs = torch.softmax(logits, dim=-1)                   # [H, Q, S]

    key_cols = _CAPTURE["key_cols"]
    if key_cols is not None:
        idx = torch.tensor(key_cols, device=probs.device)
        probs = probs.index_select(-1, idx)                 # [H, Q, K]
    _CAPTURE["buffer"][int(attn.layer_idx)] = probs.to("cpu", dtype=torch.float16)


def _pre_hook(module, args, kwargs):
    if not _CAPTURE["enabled"]:
        return
    hs = kwargs.get("hidden_states", args[0] if args else None)
    pe = kwargs.get("position_embeddings")
    am = kwargs.get("attention_mask")
    pkv = kwargs.get("past_key_value")
    if hs is None or pe is None or pkv is not None:   # only the single full-seq pass
        return
    _capture(module, hs, am, pe)


def install(model):
    """Register pre-hooks on every attention module of the loaded model. Idempotent."""
    _resolve_symbols()
    if _CAPTURE["handles"]:
        _CAPTURE["enabled"] = True
        return len(_CAPTURE["handles"])
    n = 0
    for mod in model.modules():
        if type(mod).__name__ == _ATTN_CLS_NAME and hasattr(mod, "q_proj"):
            h = mod.register_forward_pre_hook(_pre_hook, with_kwargs=True)
            _CAPTURE["handles"].append(h)
            n += 1
    _CAPTURE["enabled"] = True
    if n == 0:
        print("[attn_hook] WARNING: no attention modules matched "
              f"'{_ATTN_CLS_NAME}'; Objective B will be empty.")
    return n


def remove():
    """Remove all hooks and disable capture."""
    _CAPTURE["enabled"] = False
    for h in _CAPTURE["handles"]:
        h.remove()
    _CAPTURE["handles"] = []
