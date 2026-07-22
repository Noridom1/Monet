#!/usr/bin/env python
"""Build a row-subset of a VLMEvalKit dataset, in place, reversibly.

Why in place (same dataset name) instead of a new name:
  VLMEvalKit picks the evaluator from the dataset NAME. A renamed dataset becomes a
  "Custom" dataset; CustomMCQDataset evaluates fine, but CustomVQADataset.evaluate
  raises NotImplementedError. Keeping the official name preserves the official scorer
  for every dataset type (MCQ, VQA, MME, ...). So we overwrite `<name>.tsv` with the
  subset and keep a one-time backup `<name>.full.tsv` to restore from.

VLMEvalKit's md5-triggered re-download (which would clobber the subset) is disabled by
run_monet.py, so the subset TSV is used as-is at eval time.

Usage:
  python eval_subset.py --dataset MMBench_DEV_EN --mode head   --frac 0.1
  python eval_subset.py --dataset MMBench_DEV_EN --mode random --n 200 --seed 0
  python eval_subset.py --dataset VStarBench --mode indices --indices-file targets.csv
  python eval_subset.py --dataset MMBench_DEV_EN --mode restore        # undo, back to full
"""
import argparse
import os
import shutil
import sys

from vlmeval.smp import LMUDataRoot, load, dump
from vlmeval.dataset import build_dataset

from latent_policy import canonical_index


def ensure_full(name):
    """Return (root, tsv_path, full_backup_path); download + back up once if needed."""
    root = LMUDataRoot()
    tsv = os.path.join(root, f"{name}.tsv")
    full = os.path.join(root, f"{name}.full.tsv")
    if not os.path.exists(full):
        if not os.path.exists(tsv):
            print(f"[subset] {name}.tsv not found; downloading via VLMEvalKit ...")
            if build_dataset(name) is None:
                sys.exit(f"[subset] ERROR: could not build/download dataset '{name}'.")
        print(f"[subset] backing up full dataset -> {os.path.basename(full)}")
        shutil.copy(tsv, full)
    return root, tsv, full


def drop_localized(root, name):
    """Remove the >1GB localized cache so it regenerates from whatever <name>.tsv now holds."""
    loc = os.path.join(root, f"{name}_local.tsv")
    if os.path.exists(loc):
        os.remove(loc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--mode", required=True, choices=["head", "random", "indices", "restore"])
    ap.add_argument("--frac", type=float, default=None, help="fraction in (0, 1], e.g. 0.1 for first/random 10%%")
    ap.add_argument("--n", type=int, default=None, help="absolute number of samples")
    ap.add_argument("--seed", type=int, default=0, help="seed for --mode random")
    ap.add_argument("--indices-file", default=None, help="CSV/XLSX/TSV containing an index column")
    args = ap.parse_args()

    root, tsv, full = ensure_full(args.dataset)

    if args.mode == "restore":
        shutil.copy(full, tsv)
        drop_localized(root, args.dataset)
        print(f"[subset] restored full dataset: {args.dataset} ({len(load(tsv))} rows)")
        return

    if args.mode == "indices":
        if args.indices_file is None:
            sys.exit("[subset] ERROR: --mode indices requires --indices-file.")
        if args.frac is not None or args.n is not None:
            sys.exit("[subset] ERROR: --mode indices cannot be combined with --frac or --n.")

        from latent_policy import _load_table

        requested = _load_table(args.indices_file)
        if "index" not in requested.columns:
            sys.exit(f"[subset] ERROR: indices file lacks an index column: {args.indices_file}")
        if requested.empty:
            sys.exit("[subset] ERROR: indices file contains no rows.")

        requested_keys = [canonical_index(index) for index in requested["index"]]
        if len(requested_keys) != len(set(requested_keys)):
            sys.exit("[subset] ERROR: indices file contains duplicate indices.")

        df = load(full)
        dataset_keys = df["index"].map(canonical_index)
        if dataset_keys.duplicated().any():
            sys.exit(f"[subset] ERROR: full dataset {args.dataset} contains duplicate indices.")
        missing = sorted(set(requested_keys) - set(dataset_keys))
        if missing:
            preview = ", ".join(missing[:5])
            sys.exit(f"[subset] ERROR: {len(missing)} requested indices are absent: {preview}")

        requested_set = set(requested_keys)
        sub = df[dataset_keys.isin(requested_set)]
        dump(sub, tsv)
        drop_localized(root, args.dataset)
        print(
            f"[subset] {args.dataset}: indices {len(sub)}/{len(df)} rows "
            f"-> {os.path.basename(tsv)}"
        )
        return

    if (args.frac is None) == (args.n is None):
        sys.exit("[subset] ERROR: pass exactly one of --frac or --n.")

    df = load(full)
    total = len(df)
    k = args.n if args.n is not None else max(1, int(round(args.frac * total)))
    k = min(k, total)

    if args.mode == "head":
        sub = df.iloc[:k]
    else:  # random
        sub = df.sample(n=k, random_state=args.seed).sort_index()

    dump(sub, tsv)
    drop_localized(root, args.dataset)
    print(f"[subset] {args.dataset}: {args.mode} {k}/{total} rows -> {os.path.basename(tsv)}")


if __name__ == "__main__":
    main()
