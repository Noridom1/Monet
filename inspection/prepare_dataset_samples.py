"""Adapt a prepared dataset (``run_scripts/prepare_dataset.py`` output) into an inspection
manifest the Phase A/B pipeline can consume.

``prepare_dataset.py`` writes ``data/<name>/{images/, samples.json}`` where ``samples.json``
is a flat list of records carrying ``question`` + ``options`` (separate) and ``answer``.
Phase A/B (``generate_latents.py`` / ``inspect.py``) instead expect a manifest shaped like
``prepare_eval_samples.py``'s output::

    {"dataset", "system_prompt", "samples": [
        {"id", "image", "question_text", "gold", "pred_letter", "category", ...}, ...]}

So this script:
  * picks N samples (head or seeded-random — these datasets have thousands of rows and
    inspection is expensive),
  * builds ``question_text`` by inlining the options as ``A. ... / B. ...``,
  * maps ``answer`` -> ``gold`` (there is no prior prediction, so ``pred_letter`` is None;
    the model's freshly captured answer lives in each ``report.md`` / ``trace.pt``),
  * writes the manifest INSIDE ``data/<name>/`` so the existing ``images/`` are reused
    (image paths in the manifest stay relative to that dir).

Usage:
    python -m inspection.prepare_dataset_samples --data_dir data/VisualPuzzles --n 10
    python -m inspection.prepare_dataset_samples --data_dir data/MathVision --n 10 --mode random --seed 0

Then inspect:
    MANIFEST=data/VisualPuzzles/inspect_manifest.json \
    OUT_DIR=inspection/outputs/VisualPuzzles bash inspection/run_batch.sh
"""
import os
import json
import argparse
import random

# Must match VLMEvalKit/run_monet.py and inspection/prepare_eval_samples.py so the captured
# generation reproduces how Monet answers under the eval system prompt.
SYSTEM_PROMPT = (
    "You are a helpful multimodal assistant. You are required to answer the "
    "question based on the image provided. Put your final answer in \\boxed{}."
)

OPTION_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def build_question_text(question, options):
    """Inline MCQ options as ``A. ...`` lines, mirroring how a VLM benchmark prompt reads.

    Open-ended rows (no options) just use the question text verbatim.
    """
    text = str(question).strip()
    if options:
        lines = [f"{OPTION_LETTERS[i]}. {opt}" for i, opt in enumerate(options)]
        text = text + "\n" + "\n".join(lines)
    return text


def normalize_gold(answer):
    """A single-letter answer -> uppercase letter; anything else kept verbatim (open-ended)."""
    if answer is None:
        return None
    s = str(answer).strip()
    return s.upper() if len(s) == 1 and s.isalpha() else s


def first_present(row, *names):
    for n in names:
        if n in row and row[n] not in (None, ""):
            return row[n]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="prepared dataset dir, e.g. data/VisualPuzzles (holds samples.json + images/)")
    ap.add_argument("--n", type=int, default=10, help="number of samples to inspect (0 = all)")
    ap.add_argument("--mode", choices=["head", "random"], default="head",
                    help="head = first N (deterministic); random = seeded sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="manifest path (default: <data_dir>/inspect_manifest.json)")
    # Field overrides for datasets that name things differently.
    ap.add_argument("--id_field", default="id")
    ap.add_argument("--question_field", default="question")
    ap.add_argument("--options_field", default="options")
    ap.add_argument("--answer_field", default="answer")
    ap.add_argument("--category_field", default=None,
                    help="defaults to whichever of 'category'/'subject' exists")
    args = ap.parse_args()

    src = os.path.join(args.data_dir, "samples.json")
    if not os.path.isfile(src):
        raise SystemExit(f"[prepare-ds] '{src}' not found. Run run_scripts/05_prepare_data.sh first.")
    with open(src) as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise SystemExit(f"[prepare-ds] expected a list in {src} (prepare_dataset.py format).")
    print(f"[prepare-ds] {len(rows)} rows in {src}")

    # ---- select N ----
    idxs = list(range(len(rows)))
    if args.n and args.n > 0 and args.n < len(rows):
        if args.mode == "random":
            idxs = sorted(random.Random(args.seed).sample(idxs, args.n))
        else:
            idxs = idxs[:args.n]

    dataset_name = os.path.basename(os.path.normpath(args.data_dir))
    samples = []
    for sid_counter, i in enumerate(idxs):
        row = rows[i]
        image = first_present(row, "image")
        if image is None:
            images = row.get("images")
            if images:
                print(f"[prepare-ds] row {i} is multi-image; using the first.")
                image = images[0]
        if image is None:
            print(f"[prepare-ds] WARNING: row {i} has no image; skipped.")
            continue

        options = row.get(args.options_field) or []
        category = (row.get(args.category_field) if args.category_field
                    else first_present(row, "category", "subject"))
        # keep the dataset's own id for traceability, but make a filesystem-safe sample id
        orig_id = first_present(row, args.id_field)
        sid = f"{dataset_name}_{i:06d}"

        samples.append({
            "id": sid,
            "bucket": None,
            "index": orig_id if orig_id is not None else i,
            "image": image,
            "question_text": build_question_text(row.get(args.question_field, ""), options),
            "gold": normalize_gold(row.get(args.answer_field)),
            "pred_letter": None,           # no prior eval; captured answer is in report.md
            "hit": None,
            "category": (str(category) if category is not None else None),
        })

    out_path = args.out or os.path.join(args.data_dir, "inspect_manifest.json")
    manifest = {"dataset": dataset_name, "system_prompt": SYSTEM_PROMPT, "samples": samples}
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[prepare-ds] wrote {len(samples)} samples ({args.mode} of {len(rows)}) -> {out_path}")
    print(f"[prepare-ds] next: MANIFEST={out_path} "
          f"OUT_DIR=inspection/outputs/{dataset_name} bash inspection/run_batch.sh")


if __name__ == "__main__":
    main()
