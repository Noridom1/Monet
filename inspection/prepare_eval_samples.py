"""Prepare a small set of eval samples (images + questions) into ``data/`` for latent
inspection.

Picks 4 *correct* + 5 *genuine-wrong* samples from a VLMEvalKit result file and writes,
for each, the localized image and the **exact** prompt the model saw at eval time, so the
inspection pipeline (Phase A / B) reproduces the eval prediction faithfully.

Must run where VLMEvalKit + the dataset are localized (the A100 eval machine, ``monet``
env): it imports ``vlmeval`` and reads the source dataset TSV from ``~/LMUData``. The
downloaded result xlsx carries *no* image column — images live only in that TSV.

Correctness is re-derived here (not taken from the ``hit`` column): the judge fell back to
``exact_matching`` and logs "Failed in Prefetch ..." even on apparent matches, so we extract
the answer letter from ``prediction`` and compare to the gold ``answer``. ``hit`` is kept in
the manifest for reference only.

Usage (on the A100):
    python -m inspection.prepare_eval_samples \
        --results eval_outputs/full/Monet/T20260622-230719/Monet_MMBench_DEV_EN_gpt-4o-mini_result.xlsx \
        --dataset MMBench_DEV_EN --n_correct 4 --n_incorrect 5 --seed 0 \
        --out data/inspect_samples
"""
import os
import re
import json
import shutil
import argparse
import random

import pandas as pd

# The Monet eval system prompt (must match VLMEvalKit/run_monet.py).
SYSTEM_PROMPT = (
    "You are a helpful multimodal assistant. You are required to answer the "
    "question based on the image provided. Put your final answer in \\boxed{}."
)

OPTION_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]


def _valid_options(row):
    """Option letters present (non-empty) in this MCQ row."""
    out = []
    for L in OPTION_LETTERS:
        if L in row and not _is_blank(row[L]):
            out.append(L)
    return out


def _is_blank(v):
    if v is None:
        return True
    if isinstance(v, float) and pd.isna(v):
        return True
    return str(v).strip().lower() in ("", "nan", "none")


def extract_letter(prediction, valid):
    """Best-effort answer letter from a prediction string. Returns a letter or None.

    Handles: a bare letter ("C"), ``\\boxed{C}``, "(C)", "Answer: C", etc. Only letters in
    ``valid`` (the row's actual options) are accepted.
    """
    if _is_blank(prediction):
        return None
    s = str(prediction).strip()
    vset = set(valid)
    # 1) exactly a single option letter
    if s.upper() in vset:
        return s.upper()
    # 2) \boxed{ X }
    m = re.search(r"\\boxed\{\s*([A-Za-z])", s)
    if m and m.group(1).upper() in vset:
        return m.group(1).upper()
    # 3) first *standalone* option letter (not embedded in a word), e.g. "(C)", "C.",
    #    "Answer: A" — the negative look-arounds reject letters inside words like "blah".
    for m in re.finditer(r"(?<![A-Za-z])([A-Ha-h])(?![A-Za-z])", s):
        if m.group(1).upper() in vset:
            return m.group(1).upper()
    return None


def _msgs_to_text_and_images(msgs):
    """Split a VLMEvalKit build_prompt() message list into (text, [image_paths])."""
    texts, images = [], []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        if m.get("type") == "image":
            images.append(m.get("value"))
        elif m.get("type") == "text":
            texts.append(m.get("value", ""))
    return "\n".join(t for t in texts if t), images


def _save_image(src_path, dst_path):
    """Copy/convert a localized image to dst_path as PNG."""
    from PIL import Image
    img = Image.open(src_path).convert("RGB")
    img.save(dst_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="VLMEvalKit *_result.xlsx (has `hit`)")
    ap.add_argument("--dataset", default="MMBench_DEV_EN")
    ap.add_argument("--n_correct", type=int, default=4)
    ap.add_argument("--n_incorrect", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/inspect_samples")
    args = ap.parse_args()

    # ---- result rows (prediction / answer / hit / options) ----
    res = pd.read_excel(args.results)
    res = res.set_index("index", drop=False)
    print(f"[prepare] read {len(res)} result rows from {args.results}")

    # ---- localized dataset (base64 images + the official prompt builder) ----
    from vlmeval.dataset import build_dataset
    dataset = build_dataset(args.dataset)
    if dataset is None:
        raise SystemExit(f"[prepare] could not build dataset '{args.dataset}'.")
    data = dataset.data.set_index("index", drop=False)

    # ---- derive correctness per result row, build buckets ----
    correct_idx, wrong_idx = [], []
    for idx, row in res.iterrows():
        valid = _valid_options(row)
        gold = None if _is_blank(row.get("answer")) else str(row["answer"]).strip().upper()
        letter = extract_letter(row.get("prediction"), valid)
        if gold is None or letter is None:
            continue  # unparseable / no gold -> skip (e.g. format failures)
        if letter == gold:
            correct_idx.append(idx)
        elif letter in valid:
            wrong_idx.append(idx)  # genuine wrong: valid option, != gold

    print(f"[prepare] pools: correct={len(correct_idx)} genuine-wrong={len(wrong_idx)}")

    rng = random.Random(args.seed)

    def pick(pool, n, kind):
        if len(pool) < n:
            print(f"[prepare] WARNING: only {len(pool)} {kind} samples available (<{n}).")
        return rng.sample(pool, min(n, len(pool)))

    chosen = ([("correct", i) for i in pick(correct_idx, args.n_correct, "correct")]
              + [("incorrect", i) for i in pick(wrong_idx, args.n_incorrect, "genuine-wrong")])

    # ---- materialize images + manifest ----
    out_root = args.out
    img_dir = os.path.join(out_root, "images")
    os.makedirs(img_dir, exist_ok=True)

    samples = []
    counters = {"correct": 0, "incorrect": 0}
    for bucket, idx in chosen:
        if idx not in data.index:
            print(f"[prepare] WARNING: index {idx} not in dataset (subset TSV?); skipped. "
                  f"Restore full: python run_scripts/eval_subset.py --dataset {args.dataset} "
                  f"--mode restore")
            continue
        ds_row = data.loc[idx]
        msgs = dataset.build_prompt(ds_row)
        text, images = _msgs_to_text_and_images(msgs)
        if not images:
            print(f"[prepare] WARNING: no image for index {idx}; skipped.")
            continue
        if len(images) > 1:
            print(f"[prepare] WARNING: index {idx} has {len(images)} images; using the first.")

        sid = f"{bucket}_{counters[bucket]}"
        counters[bucket] += 1
        rel_img = os.path.join("images", f"{sid}.png")
        _save_image(images[0], os.path.join(out_root, rel_img))

        res_row = res.loc[idx]
        valid = _valid_options(res_row)
        samples.append({
            "id": sid,
            "bucket": bucket,
            "index": int(idx),
            "image": rel_img,
            "question_text": text,
            "gold": str(res_row["answer"]).strip().upper(),
            "pred_letter": extract_letter(res_row.get("prediction"), valid),
            "hit": (None if _is_blank(res_row.get("hit")) else int(res_row["hit"])),
            "category": (None if "category" not in res_row or _is_blank(res_row["category"])
                         else str(res_row["category"])),
        })

    manifest = {"dataset": args.dataset, "system_prompt": SYSTEM_PROMPT, "samples": samples}
    with open(os.path.join(out_root, "samples.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    n_c = sum(s["bucket"] == "correct" for s in samples)
    n_w = sum(s["bucket"] == "incorrect" for s in samples)
    print(f"[prepare] wrote {len(samples)} samples ({n_c} correct, {n_w} incorrect) "
          f"-> {out_root}/samples.json")
    for s in samples:
        print(f"    {s['id']:<12} idx={s['index']:<6} gold={s['gold']} "
              f"pred={s['pred_letter']} hit={s['hit']}")


if __name__ == "__main__":
    main()
