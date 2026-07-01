#!/usr/bin/env python3
"""Summarize VLMEvalKit Table 3 scores and Monet latent activation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DATASETS = ("VStarBench", "HRBench4K", "HRBench8K", "MME-RealWorld-Lite")
TARGETS = {"VStarBench": 83.25, "HRBench4K": 71.00, "HRBench8K": 68.00, "MME-RealWorld-Lite": 55.50}
LATENT_MARKER = "<abs_vis_token>"


def percentage(value: float) -> float:
    """VLMEvalKit commonly writes accuracy as [0, 1]; Table 3 uses percent."""
    return value * 100.0 if abs(value) <= 1.0 else value


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".xlsx":
        return pd.read_excel(path)
    return pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")


def prediction_file(run_dir: Path, dataset: str) -> Path | None:
    candidates = []
    for suffix in (".xlsx", ".tsv", ".csv"):
        candidates.extend(run_dir.rglob(f"*{dataset}*{suffix}"))
    candidates = [p for p in candidates if not any(x in p.stem for x in ("_acc", "_result", "_judge"))]
    for path in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            if "prediction" in read_table(path).columns:
                return path
        except Exception:
            continue
    return None


def score_metrics(run_dir: Path, dataset: str) -> dict[str, float]:
    files = list(run_dir.rglob(f"*{dataset}*_acc*.csv")) + list(run_dir.rglob(f"*{dataset}*_score*.csv"))
    if not files:
        return {}
    frame = read_table(max(files, key=lambda p: p.stat().st_mtime))
    metrics = {}
    if "split" in frame.columns:
        numeric_columns = [c for c in frame.columns if c != "split" and pd.api.types.is_numeric_dtype(frame[c])]
        for _, row in frame.iterrows():
            for column in numeric_columns:
                if pd.notna(row[column]):
                    metrics[f'{row["split"]}.{column}'] = percentage(float(row[column]))
    for column in ("Overall", "overall", "acc", "accuracy"):
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            if len(values):
                metrics["overall"] = percentage(float(values.iloc[0]))
                return metrics
    numeric = frame.select_dtypes(include="number")
    if not numeric.empty:
        metrics["overall"] = percentage(float(numeric.iloc[0, -1]))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--latent-sizes", nargs="+", type=int, required=True)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    args = parser.parse_args()
    rows, details = [], []
    for latent_size in args.latent_sizes:
        run_dir = args.work_dir / f"latent_{latent_size}"
        for dataset in args.datasets:
            pred_path = prediction_file(run_dir, dataset)
            expected = successful = activated = blocks = 0
            if pred_path is not None:
                frame = read_table(pred_path)
                expected = len(frame)
                predictions = frame["prediction"]
                ok = predictions.notna() & predictions.astype(str).str.strip().ne("")
                successful = int(ok.sum())
                counts = predictions[ok].astype(str).str.count(LATENT_MARKER)
                activated, blocks = int(counts.gt(0).sum()), int(counts.sum())
                index_col = "index" if "index" in frame.columns else None
                for idx, count in counts.items():
                    details.append({"dataset": dataset, "latent_size": latent_size,
                                    "index": frame.loc[idx, index_col] if index_col else idx,
                                    "latent_activated": bool(count), "latent_block_count": int(count)})
            metrics = score_metrics(run_dir, dataset)
            score = metrics.get("overall")
            rows.append({"dataset": dataset, "latent_size": latent_size, "overall": score,
                         "paper_target": TARGETS.get(dataset), "activation_count": activated,
                         "successful": successful, "expected": expected,
                         "activation_ratio": activated / successful if successful else None,
                         "latent_block_count": blocks, "metrics_json": json.dumps(metrics, sort_keys=True),
                         **{f"metric.{key}": value for key, value in metrics.items()}})

    args.work_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.work_dir / "table3_by_latent_size.csv", index=False)
    pd.DataFrame(details).to_csv(args.work_dir / "latent_activation_details.csv", index=False)
    best = []
    for dataset in args.datasets:
        candidates = [r for r in rows if r["dataset"] == dataset and r["overall"] is not None]
        if candidates:
            best.append(max(candidates, key=lambda r: r["overall"]))
    pd.DataFrame(best).to_csv(args.work_dir / "table3_best.csv", index=False)
    metadata = {"latent_sizes": args.latent_sizes, "datasets": args.datasets, "paper_targets": TARGETS}
    (args.work_dir / "summary_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    shown = best or rows
    lines = ["# Monet Table 3 evaluation", "", "| Dataset | Latent size | Overall | Paper | Activation | Completion |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in shown:
        score = "n/a" if row["overall"] is None else f'{row["overall"]:.2f}'
        ratio = "n/a" if row["activation_ratio"] is None else f'{100 * row["activation_ratio"]:.2f}%'
        lines.append(f'| {row["dataset"]} | {row["latent_size"]} | {score} | {row["paper_target"]:.2f} | '
                     f'{ratio} | {row["successful"]}/{row["expected"]} |')
    report = "\n".join(lines) + "\n"
    (args.work_dir / "table3_summary.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
