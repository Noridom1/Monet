#!/usr/bin/env python
"""Download a Hugging Face VLM benchmark and normalize it to Monet's common layout:

    data/<name>/
      images/            # one PNG per image, extracted from the dataset
      samples.json       # list of records, images replaced by relative paths

Each record in samples.json keeps every original column, except that image columns
are saved to images/ and replaced by a relative path string:
  - exactly one image in a row  -> "image": "images/000123.png"
  - several images in a row      -> "images": ["images/000123_0.png", ...]

Image columns are AUTO-DETECTED from the dataset features (anything HF types as
`datasets.Image`), so most datasets work with no per-dataset code. The REGISTRY
below only pins convenient defaults (repo id, split, config) for known names; any
dataset can also be prepared ad-hoc via --repo/--split/--config.

Usage:
  # known datasets (see REGISTRY)
  python prepare_dataset.py VisualPuzzles
  python prepare_dataset.py MathVision --split testmini

  # any other HF dataset, ad-hoc
  python prepare_dataset.py MyBench --repo org/MyBench --split test --config default

  # quick smoke test: only the first 20 rows
  python prepare_dataset.py VisualPuzzles --limit 20
"""
import argparse
import json
import os
import sys

# Known datasets -> default load args. Image handling is auto-detected, so entries
# only need to pin the repo and the split/config you actually want to evaluate on.
REGISTRY = {
    "VisualPuzzles": {"repo": "neulab/VisualPuzzles", "split": "train"},
    "MathVision":    {"repo": "MathLLMs/MathVision",  "split": "test"},
}


def detect_image_fields(features):
    """Return the names of all columns HF types as images (incl. lists of images)."""
    from datasets import Image, Sequence

    names = []
    for name, feat in features.items():
        if isinstance(feat, Image):
            names.append(name)
        elif isinstance(feat, Sequence) and isinstance(feat.feature, Image):
            names.append(name)
        elif isinstance(feat, list) and len(feat) == 1 and isinstance(feat[0], Image):
            names.append(name)
    return names


def save_image(img, path):
    """Save a PIL image as PNG (PNG is lossless and handles every mode)."""
    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGB")
    img.save(path, format="PNG")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="dataset name (key in REGISTRY, or any label for ad-hoc --repo)")
    ap.add_argument("--repo", default=None, help="HF repo id (overrides/required if name not in REGISTRY)")
    ap.add_argument("--config", default=None, help="HF dataset config name")
    ap.add_argument("--split", default=None, help="split to export (e.g. train, test, testmini)")
    ap.add_argument("--out-root", default=None, help="output root (default: <repo>/data)")
    ap.add_argument("--image-fields", default=None,
                    help="comma-separated image column names to override auto-detection")
    ap.add_argument("--limit", type=int, default=None, help="only export the first N rows (smoke test)")
    args = ap.parse_args()

    from datasets import load_dataset

    cfg = dict(REGISTRY.get(args.name, {}))
    repo = args.repo or cfg.get("repo")
    if not repo:
        sys.exit(f"[prepare] ERROR: '{args.name}' is not in REGISTRY; pass --repo org/name.")
    config = args.config or cfg.get("config")
    split = args.split or cfg.get("split", "test")

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_root = args.out_root or os.path.join(repo_root, "data")
    out_dir = os.path.join(out_root, args.name)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    print(f"[prepare] loading {repo} (config={config}, split={split}) ...")
    ds = load_dataset(repo, name=config, split=split)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    if args.image_fields is not None:
        image_fields = [f.strip() for f in args.image_fields.split(",") if f.strip()]
    else:
        image_fields = detect_image_fields(ds.features)
    print(f"[prepare] {len(ds)} rows; image column(s): {image_fields or '(none detected)'}")

    samples = []
    for idx, row in enumerate(ds):
        rec = dict(row)
        paths = []
        for field in image_fields:
            val = rec.pop(field, None)
            imgs = val if isinstance(val, list) else [val]
            for img in imgs:
                if img is None:
                    continue
                suffix = f"_{len(paths)}" if (len(imgs) > 1 or len(image_fields) > 1) else ""
                rel = os.path.join("images", f"{idx:06d}{suffix}.png")
                save_image(img, os.path.join(out_dir, rel))
                paths.append(rel)

        # Normalize to a single 'image' (one image) or 'images' (many); drop both keys
        # first so a stale original (e.g. MathVision's 'image' path string) can't linger.
        rec.pop("image", None)
        rec.pop("images", None)
        if len(paths) == 1:
            rec["image"] = paths[0]
        elif len(paths) > 1:
            rec["images"] = paths
        samples.append(rec)

        if (idx + 1) % 200 == 0:
            print(f"[prepare]   {idx + 1}/{len(ds)} rows")

    samples_path = os.path.join(out_dir, "samples.json")
    with open(samples_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"[prepare] DONE. {len(samples)} samples -> {samples_path}")
    print(f"[prepare]        images -> {img_dir}/")


if __name__ == "__main__":
    main()
