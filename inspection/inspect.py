"""Phase B driver — Objectives A (logit lens) and B (attention maps).

Loads a Trace from ``inspection.generate_latents`` and runs ONE teacher-forced forward
over the full recorded sequence (plain causal mask, no cache) that simultaneously yields:
  * ``output_hidden_states`` -> Objective A (logit lens, final + by-layer)
  * sliced attention via the ``attn_hook`` monkey-patch -> Objective B
    (B.1 text->latent, B.2 latent->image), never materialising the full S x S matrix.

A built-in numerical gate checks the replay reproduces generation (replay last-layer hidden
at ``latent_positions[k]-1`` == captured ``latent_hidden_states[k]``).

Artifacts (next to the trace):
  logit_lens_final.json, logit_lens_by_layer.npz, logit_lens.md,
  attn_text2latent.npz, attn_latent2image.npz, heatmaps/*.png, report.md

Run on the A100:
    python -m inspection.inspect --trace inspection/outputs/demo/trace.pt
"""
import os
import json
import argparse
from typing import List, Optional

import numpy as np
import torch

from inspection.load_model import load_monet
from inspection.logit_lens import logit_lens, decode_tokens
from inspection import attn_hook

MAX_TEXT_Q = 512  # cap on text query rows for B.1 (logged if exceeded)


def _block_index_of(pos: int, blocks: List[List[int]]):
    for bi, blk in enumerate(blocks):
        if pos in blk:
            return bi, blk.index(pos)
    return -1, -1


def _select_text_queries(trace, latent_positions):
    """Generated, non-latent positions at/after the first latent block — the tokens that
    can attend *back* to the latents. Capped to MAX_TEXT_Q (evenly subsampled)."""
    if not latent_positions:
        return []
    first = min(latent_positions)
    S = trace["input_ids"].shape[0]
    latset = set(latent_positions)
    cand = [p for p in range(first, S) if p not in latset]
    if len(cand) > MAX_TEXT_Q:
        idx = np.linspace(0, len(cand) - 1, MAX_TEXT_Q).round().astype(int)
        cand = [cand[i] for i in sorted(set(idx.tolist()))]
        print(f"[Phase B] text queries capped {MAX_TEXT_Q} (of available); evenly subsampled.")
    return cand


@torch.no_grad()
def teacher_forced_pass(model, trace, capture_query, capture_keys, device="cuda"):
    """One full-sequence causal forward. Returns (hidden_states tuple, last_hidden, attn_buffer).

    Latent vectors and image features are already in ``trace['inputs_embeds']``, so we drive
    the text model with inputs_embeds + precomputed mrope positions (no pixel_values).
    """
    inner = model.model
    text = inner.language_model
    dtype = model.lm_head.weight.dtype

    input_ids = trace["input_ids"].to(device).unsqueeze(0)
    embeds = trace["inputs_embeds"].to(device=device, dtype=dtype).unsqueeze(0)
    grid = trace["image_grid_thw"]
    grid = grid.to(device) if grid is not None else None
    S = input_ids.shape[1]
    attn = torch.ones((1, S), dtype=torch.long, device=device)
    position_ids, _ = inner.get_rope_index(input_ids, grid, None, attention_mask=attn)

    attn_hook.configure(capture_query, capture_keys)
    attn_hook.install()
    try:
        out = text(
            inputs_embeds=embeds,
            position_ids=position_ids,
            attention_mask=attn,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        buffer = {l: t.clone() for l, t in attn_hook.get_buffer().items()}
    finally:
        attn_hook.remove()
    return out.hidden_states, out.last_hidden_state, buffer


@torch.no_grad()
def _replay_gate(last_hidden, latent_hidden, latent_positions, device):
    lh = latent_hidden.to(device=device, dtype=last_hidden.dtype)
    diffs, coss = [], []
    for k, pos in enumerate(latent_positions):
        produced = last_hidden[pos - 1]
        captured = lh[k]
        diffs.append((produced - captured).abs().max().item())
        coss.append(torch.nn.functional.cosine_similarity(
            produced.float(), captured.float(), dim=0).item())
    return {"max_abs_diff": max(diffs), "min_cosine": min(coss),
            "ok": bool(min(coss) > 0.99 and max(diffs) < 5e-2)}


def _stack_buffer(buffer, n_layers):
    """layer-keyed [H,Q,K] -> [n_layers, H, Q, K] float32 numpy (missing layers -> zeros)."""
    sample = next(iter(buffer.values()))
    Hh, Q, K = sample.shape
    arr = np.zeros((n_layers, Hh, Q, K), dtype=np.float16)
    for l, t in buffer.items():
        arr[l] = t.numpy()
    return arr


def run(trace_path, model_path, k=20, image_path=None, device="cuda", do_attention=True):
    trace = torch.load(trace_path, map_location="cpu", weights_only=False)
    out_dir = os.path.dirname(trace_path)
    meta = trace.get("meta", {})

    latent_positions = list(trace["latent_positions"])
    latent_blocks = [list(b) for b in trace["latent_blocks"]]
    latent_hidden = trace["latent_hidden_states"]
    image_positions = list(trace["image_positions"])
    n_latent = latent_hidden.shape[0]
    print(f"[Phase B] seq_len={trace['input_ids'].shape[0]} n_latent={n_latent} "
          f"blocks={len(latent_blocks)} n_image={len(image_positions)}")
    if n_latent == 0:
        print("[Phase B] No latent tokens — nothing to inspect.")
        return

    model, processor, _ = load_monet(model_path, device=device)
    tokenizer = processor.tokenizer

    # ---- A.1 final logit lens on captured latents (already post-norm) ----
    top_ids, top_probs = logit_lens(latent_hidden, model, k=k, already_normed=True)
    final_records = []
    for li in range(n_latent):
        pos = latent_positions[li]
        bi, step = _block_index_of(pos, latent_blocks)
        toks = decode_tokens(tokenizer, top_ids[li].tolist())
        final_records.append({
            "latent_idx": li, "position": int(pos), "block_idx": bi, "step_in_block": step,
            "topk": [{"token_id": int(top_ids[li, j]), "token_str": toks[j],
                      "prob": float(top_probs[li, j])} for j in range(k)],
        })
    with open(os.path.join(out_dir, "logit_lens_final.json"), "w") as f:
        json.dump(final_records, f, indent=2, ensure_ascii=False)

    # ---- build capture index sets, then ONE replay ----
    text_positions = _select_text_queries(trace, latent_positions) if do_attention else []
    query_poss = sorted(set(text_positions) | set(latent_positions))
    key_cols = sorted(set(latent_positions) | set(image_positions))
    hs, last_hidden, buffer = teacher_forced_pass(
        model, trace, query_poss if do_attention else [], key_cols if do_attention else [],
        device=device)
    last_hidden = last_hidden[0]

    gate = _replay_gate(last_hidden, latent_hidden, latent_positions, device)
    print(f"[Phase B] replay==generation gate: max|Δ|={gate['max_abs_diff']:.4e} "
          f"min_cos={gate['min_cosine']:.6f} -> {'OK' if gate['ok'] else 'CHECK'}")

    # ---- A.2 depth-resolved logit lens at latent positions ----
    n_lp1 = len(hs)
    by_layer_ids = torch.zeros(n_latent, n_lp1, k, dtype=torch.long)
    by_layer_probs = torch.zeros(n_latent, n_lp1, k, dtype=torch.float32)
    lat_idx = torch.tensor(latent_positions, device=device)
    for l in range(n_lp1):
        h_at = hs[l][0].index_select(0, lat_idx)
        ids_l, probs_l = logit_lens(h_at, model, k=k, already_normed=(l == n_lp1 - 1))
        by_layer_ids[:, l, :] = ids_l
        by_layer_probs[:, l, :] = probs_l
    np.savez_compressed(
        os.path.join(out_dir, "logit_lens_by_layer.npz"),
        token_ids=by_layer_ids.numpy(), probs=by_layer_probs.numpy(),
        latent_positions=np.array(latent_positions))

    # ---- Objective B: split the captured buffer into the two relationships ----
    attn_summary = None
    if do_attention and buffer:
        n_layers = model.config.text_config.num_hidden_layers if hasattr(
            model.config, "text_config") else model.config.num_hidden_layers
        full = _stack_buffer(buffer, n_layers)                 # [L, H, Q, K]
        qmap = {p: i for i, p in enumerate(query_poss)}
        kmap = {p: i for i, p in enumerate(key_cols)}
        t_rows = [qmap[p] for p in text_positions]
        l_rows = [qmap[p] for p in latent_positions]
        l_cols = [kmap[p] for p in latent_positions]
        i_cols = [kmap[p] for p in image_positions]

        text2latent = full[:, :, t_rows, :][:, :, :, l_cols]   # [L, H, Q_text, N_latent]
        latent2image = full[:, :, l_rows, :][:, :, :, i_cols]  # [L, H, N_latent, N_image]

        grid = trace["image_grid_thw"]
        grid_np = grid.numpy() if grid is not None else None
        np.savez_compressed(os.path.join(out_dir, "attn_text2latent.npz"),
                            attn=text2latent, text_positions=np.array(text_positions),
                            latent_positions=np.array(latent_positions))
        np.savez_compressed(os.path.join(out_dir, "attn_latent2image.npz"),
                            attn=latent2image, latent_positions=np.array(latent_positions),
                            image_positions=np.array(image_positions),
                            image_grid_thw=grid_np,
                            spatial_merge_size=int(trace["spatial_merge_size"]))
        attn_summary = {"text2latent": text2latent, "latent2image": latent2image,
                        "text_positions": text_positions, "grid_thw": grid_np,
                        "merge": int(trace["spatial_merge_size"])}

        # ---- visualisations ----
        try:
            from inspection import visualize
            visualize.render_all(out_dir, attn_summary, latent_positions, tokenizer,
                                  trace, image_path)
        except Exception as e:  # pragma: no cover - plotting is best-effort
            print(f"[Phase B] visualisation skipped ({type(e).__name__}: {e}); "
                  f"npz artifacts still written.")

    _write_markdown(out_dir, trace, final_records, by_layer_ids, by_layer_probs,
                    tokenizer, gate, k, attn_summary, latent_positions)
    _write_report(out_dir, trace, final_records, gate, attn_summary, latent_positions,
                  tokenizer)
    print(f"[Phase B] artifacts written -> {out_dir}")


def _write_markdown(out_dir, trace, final_records, by_layer_ids, by_layer_probs,
                    tokenizer, gate, k, attn_summary, latent_positions):
    meta = trace.get("meta", {})
    L = []
    L.append("# Latent logit-lens report (Objective A)\n")
    L.append(f"- sequence length: **{trace['input_ids'].shape[0]}**")
    L.append(f"- latent tokens: **{meta.get('num_latent', len(final_records))}** "
             f"in **{meta.get('num_latent_blocks', '?')}** block(s)")
    L.append(f"- LATENT_SIZE: **{meta.get('latent_size', '?')}**")
    L.append(f"- replay==generation gate: max|Δ|=`{gate['max_abs_diff']:.3e}` "
             f"min_cos=`{gate['min_cosine']:.5f}` -> "
             f"**{'PASS' if gate['ok'] else 'CHECK — replay diverged'}**\n")
    L.append("## A.1 — Final-layer logit lens (top-8 per latent)\n")
    L.append("| latent | block:step | top tokens (prob) |")
    L.append("|---|---|---|")
    for r in final_records:
        cells = ", ".join(f"`{t['token_str']}` {t['prob']:.2f}" for t in r["topk"][:8])
        L.append(f"| {r['latent_idx']} (pos {r['position']}) "
                 f"| {r['block_idx']}:{r['step_in_block']} | {cells} |")
    L.append("\n## A.2 — Depth trajectory (top-1 token per layer)\n")
    n_latent, n_lp1, _ = by_layer_ids.shape
    L.append("| layer | " + " | ".join(f"L{li}" for li in range(n_latent)) + " |")
    L.append("|---|" + "---|" * n_latent)
    for l in range(n_lp1):
        row = [f"{l}"]
        for li in range(n_latent):
            tid = int(by_layer_ids[li, l, 0])
            p = float(by_layer_probs[li, l, 0])
            tok = tokenizer.decode([tid]).replace("|", "\\|").strip() or "·"
            row.append(f"`{tok}` {p:.2f}")
        L.append("| " + " | ".join(row) + " |")
    with open(os.path.join(out_dir, "logit_lens.md"), "w") as f:
        f.write("\n".join(L) + "\n")


def _write_report(out_dir, trace, final_records, gate, attn_summary, latent_positions,
                  tokenizer):
    meta = trace.get("meta", {})
    L = []
    L.append("# Monet latent inspection report\n")
    L.append(f"- sequence length **{trace['input_ids'].shape[0]}**, "
             f"latents **{len(final_records)}** in **{meta.get('num_latent_blocks','?')}** "
             f"block(s), LATENT_SIZE **{meta.get('latent_size','?')}**")
    L.append(f"- replay==generation gate: "
             f"**{'PASS' if gate['ok'] else 'CHECK'}** "
             f"(min_cos `{gate['min_cosine']:.5f}`, max|Δ| `{gate['max_abs_diff']:.2e}`)\n")
    L.append("## Generated text\n")
    L.append("```\n" + trace["generated_text"].strip() + "\n```\n")
    L.append("## What each latent represents (final logit lens, top-5)\n")
    L.append("| latent | top tokens |")
    L.append("|---|---|")
    for r in final_records:
        cells = ", ".join(f"`{t['token_str']}` ({t['prob']:.2f})" for t in r["topk"][:5])
        L.append(f"| {r['latent_idx']} | {cells} |")
    L.append("")
    if attn_summary is not None:
        t2l = attn_summary["text2latent"].astype(np.float32).mean(axis=(0, 1))  # [Q_text, N_lat]
        per_latent = t2l.mean(axis=0)                                            # [N_lat]
        L.append("## Objective B — attention summary\n")
        L.append("**text → latent** (mean attention from generated tokens to each latent, "
                 "averaged over layers/heads):\n")
        L.append("| latent | mean text→latent attn |")
        L.append("|---|---|")
        for li in range(per_latent.shape[0]):
            L.append(f"| {li} | {per_latent[li]:.4f} |")
        L.append("")
        L.append("See `attn_text2latent.png` and `heatmaps/` for latent→image overlays "
                 "(full per-layer/head tensors in the `.npz` files).\n")
        if os.path.exists(os.path.join(out_dir, "attn_text2latent.png")):
            L.append("![text to latent](attn_text2latent.png)\n")
        hdir = os.path.join(out_dir, "heatmaps")
        if os.path.isdir(hdir):
            for li in range(len(final_records)):
                fn = f"latent{li}_overlay.png"
                if os.path.exists(os.path.join(hdir, fn)):
                    L.append(f"latent {li}: ![latent {li}](heatmaps/{fn})")
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="inspection/outputs/demo/trace.pt")
    ap.add_argument("--model_path", default=os.environ.get("MODEL_PATH", "models/Monet-7B"))
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--image", default="images/example_question.png",
                    help="original image for latent->image overlays")
    ap.add_argument("--no_attention", action="store_true", help="logit lens only")
    args = ap.parse_args()
    run(args.trace, args.model_path, k=args.topk, image_path=args.image,
        do_attention=not args.no_attention)


if __name__ == "__main__":
    main()
