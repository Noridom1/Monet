#!/usr/bin/env python3
"""Build Table 3 score and latent-activation reports from completed evals.

This script is deliberately separate from ``run.sh``: it only reads
existing evaluation artifacts and can be rerun after additional latent sizes
finish.  Monet predictions carry ``<ltnt:N>`` metadata, where ``N`` is the
number of captured latent blocks for that sample.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from run_scripts.evaluation.latent_activation import latent_block_count
except ModuleNotFoundError:  # Support direct execution from run_scripts/evaluation/.
    from latent_activation import latent_block_count


DATASETS = ("VStarBench", "HRBench4K", "HRBench8K", "MME-RealWorld-Lite")
TARGETS = {
    "VStarBench": 83.25,
    "HRBench4K": 71.00,
    "HRBench8K": 68.00,
    "MME-RealWorld-Lite": 55.50,
}
TABLE_SUFFIXES = (".xlsx", ".csv", ".tsv")


def percentage(value: float) -> float:
    """Convert VLMEvalKit's fractional metrics to the percent scale."""
    return value * 100.0 if abs(value) <= 1.0 else value


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".xlsx":
        return pd.read_excel(path)
    return pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")


def matching_files(run_dir: Path, dataset: str, suffixes: tuple[str, ...] = TABLE_SUFFIXES) -> list[Path]:
    files = [path for path in run_dir.rglob(f"*{dataset}*") if path.is_file() and path.suffix in suffixes]
    return sorted(files, key=lambda path: (len(path.relative_to(run_dir).parts), str(path)))


def prediction_file(run_dir: Path, dataset: str) -> Path | None:
    """Find the canonical prediction sheet, excluding evaluator intermediates."""
    excluded = ("_acc", "_score", "_result", "_judge", "_rating")
    for path in matching_files(run_dir, dataset):
        if any(fragment in path.stem for fragment in excluded):
            continue
        try:
            if "prediction" in read_table(path).columns:
                return path
        except Exception:
            continue
    return None


def numeric_column(frame: pd.DataFrame, names: tuple[str, ...]) -> pd.Series | None:
    for name in names:
        if name in frame.columns:
            values = pd.to_numeric(frame[name], errors="coerce").dropna()
            if not values.empty:
                return values
    return None


def table_score(path: Path, dataset: str) -> float | None:
    """Read a benchmark score using its official VLMEvalKit output convention."""
    try:
        frame = read_table(path)
    except Exception:
        return None

    # VStar's judge-result sheet stores one ``hit`` value per sample.
    if dataset == "VStarBench":
        values = numeric_column(frame, ("hit",))
        if values is not None:
            return percentage(float(values.mean()))

    # HRBench's official summary is the Average/all accuracy row.
    if dataset in {"HRBench4K", "HRBench8K"} and {"cycle", "type", "accuracy"}.issubset(frame.columns):
        average_all = frame[
            frame["cycle"].astype(str).str.casefold().eq("average")
            & frame["type"].astype(str).str.casefold().eq("all")
        ]
        values = numeric_column(average_all, ("accuracy",))
        if values is not None:
            return percentage(float(values.iloc[-1]))

    values = numeric_column(frame, ("Overall", "overall", "accuracy", "acc"))
    if values is not None:
        return percentage(float(values.iloc[-1]))
    return None


def score_for_dataset(run_dir: Path, dataset: str) -> tuple[float | None, Path | None]:
    """Return the benchmark's overall score and the artifact that supplied it."""
    if dataset == "MME-RealWorld-Lite":
        rating_files = sorted(run_dir.rglob(f"*{dataset}*_rating.json"), key=lambda path: str(path))
        for path in rating_files:
            try:
                overall = json.loads(path.read_text(encoding="utf-8")).get("Overall")
                if overall is not None:
                    return percentage(float(overall)), path
            except (OSError, ValueError, TypeError):
                continue

    files = matching_files(run_dir, dataset)
    preference = ("_judge_result", "_result", "_acc", "_score")
    preferred = [
        path
        for tag in preference
        for path in files
        if tag in path.stem
    ]
    for path in [*preferred, *[path for path in files if path not in preferred]]:
        score = table_score(path, dataset)
        if score is not None:
            return score, path
    return None, None


def expected_rows(run_dir: Path, dataset: str, fallback: int) -> int:
    """Use run metadata when available, otherwise the prediction row count."""
    for path in sorted(run_dir.rglob("run_config.json"), key=lambda item: str(item)):
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))["datasets"][dataset]["rows"]
            return int(rows)
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return fallback


def activation_stats(
    prediction_path: Path | None, run_dir: Path, dataset: str, latent_size: int
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """Extract sample-level latent markers and validate capture completeness."""
    empty = {
        "prediction_file": "",
        "activation_count": 0,
        "successful": 0,
        "expected": 0,
        "activation_ratio_all": None,
        "activation_ratio_successful": None,
        "latent_block_count": 0,
    }
    if prediction_path is None:
        return empty, [], [f"missing {dataset} prediction file for latent_{latent_size}"]

    frame = read_table(prediction_path)
    if "prediction" not in frame.columns:
        return empty, [], [f"missing prediction column in {prediction_path}"]

    details: list[dict[str, Any]] = []
    failures: list[str] = []
    predictions = frame["prediction"]
    completed = predictions.notna() & predictions.astype(str).str.strip().ne("")
    counts: list[int | None] = []
    index_column = "index" if "index" in frame.columns else None
    for row_number, response in predictions.items():
        is_completed = bool(completed.loc[row_number])
        count = latent_block_count(str(response)) if is_completed else None
        counts.append(count)
        if is_completed and count is None:
            failures.append(f"{prediction_path}: row {row_number} is missing <ltnt:N> activation metadata")
        details.append(
            {
                "dataset": dataset,
                "latent_size": latent_size,
                "index": frame.loc[row_number, index_column] if index_column else row_number,
                "completed": is_completed,
                "activation_captured": count is not None,
                "latent_activated": bool(count and count > 0),
                "latent_block_count": count,
            }
        )

    expected = expected_rows(run_dir, dataset, len(frame))
    successful = int(completed.sum())
    activated = sum(count is not None and count > 0 for count in counts)
    blocks = sum(count or 0 for count in counts)
    stats = {
        "prediction_file": str(prediction_path.resolve()),
        "activation_count": int(activated),
        "successful": successful,
        "expected": expected,
        "activation_ratio_all": activated / expected if expected else None,
        "activation_ratio_successful": activated / successful if successful else None,
        "latent_block_count": int(blocks),
    }
    return stats, details, failures


def write_reports(args: argparse.Namespace) -> list[str]:
    """Write all Table 3 report artifacts and return validation failures."""
    work_dir = Path(args.work_dir)
    datasets = tuple(args.datasets)
    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    failures: list[str] = []

    for latent_size in args.latent_sizes:
        run_dir = work_dir / f"latent_{latent_size}"
        for dataset in datasets:
            prediction_path = prediction_file(run_dir, dataset) if run_dir.is_dir() else None
            stats, activation_details, activation_failures = activation_stats(
                prediction_path, run_dir, dataset, latent_size
            )
            score, score_path = score_for_dataset(run_dir, dataset) if run_dir.is_dir() else (None, None)
            issues = list(activation_failures)
            if score is None and not args.allow_missing_scores:
                issues.append(f"missing {dataset} score for latent_{latent_size}")
            failures.extend(issues)
            details.extend(activation_details)
            rows.append(
                {
                    "dataset": dataset,
                    "latent_size": latent_size,
                    "overall": score,
                    "paper_target": TARGETS.get(dataset),
                    "score_delta": score - TARGETS[dataset] if score is not None and dataset in TARGETS else None,
                    **stats,
                    "score_file": str(score_path.resolve()) if score_path else "",
                    "issues": "; ".join(issues),
                }
            )

    report = pd.DataFrame(rows)
    report.to_csv(work_dir / "table3_by_latent_size.csv", index=False)
    pd.DataFrame(details).to_csv(work_dir / "latent_activation_details.csv", index=False)

    best_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        candidates = [row for row in rows if row["dataset"] == dataset and row["overall"] is not None]
        if candidates:
            best_rows.append(min(candidates, key=lambda row: (-float(row["overall"]), int(row["latent_size"]))))
    pd.DataFrame(best_rows).to_csv(work_dir / "table3_best.csv", index=False)

    metadata = {
        "activation_capture": "inline_ltnt_v1",
        "complete": not failures,
        "datasets": list(datasets),
        "latent_sizes": list(args.latent_sizes),
        "paper_targets": {dataset: TARGETS.get(dataset) for dataset in datasets},
        "scores_required": not args.allow_missing_scores,
        "score_selection": "highest overall per dataset; smaller latent size wins ties",
        "failures": failures,
    }
    (work_dir / "summary_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    shown = best_rows or rows
    lines = [
        "# Monet Table 3 evaluation",
        "",
        "| Dataset | Latent size | Overall | Paper | Delta | Activation (all) | Activation (successful) | Completion | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in shown:
        score = "n/a" if row["overall"] is None else f'{row["overall"]:.2f}'
        target = "n/a" if row["paper_target"] is None else f'{row["paper_target"]:.2f}'
        delta = "n/a" if row["score_delta"] is None else f'{row["score_delta"]:+.2f}'
        ratio_all = "n/a" if row["activation_ratio_all"] is None else f'{100 * row["activation_ratio_all"]:.2f}%'
        ratio_success = (
            "n/a" if row["activation_ratio_successful"] is None else f'{100 * row["activation_ratio_successful"]:.2f}%'
        )
        status = "ok" if not row["issues"] else "issues"
        lines.append(
            f'| {row["dataset"]} | {row["latent_size"]} | {score} | {target} | {delta} | '
            f'{ratio_all} | {ratio_success} | {row["successful"]}/{row["expected"]} | {status} |'
        )
    (work_dir / "table3_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True, help="Directory containing latent_<size>/ runs")
    parser.add_argument("--latent-sizes", nargs="+", type=int, required=True)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--allow-missing-scores", action="store_true", help="Write an activation-only report")
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    failures = write_reports(args)
    print((args.work_dir / "table3_summary.md").read_text(encoding="utf-8"), end="")
    if failures:
        print("\n[table3-summary] issues:")
        print("\n".join(f"- {failure}" for failure in failures))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
