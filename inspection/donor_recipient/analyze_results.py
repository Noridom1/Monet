"""Aggregate donor-recipient outputs into JSON, CSV, and Markdown reports."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from inspection.donor_recipient.common import (
    CONDITIONS,
    HYBRID_SCORING_PROTOCOL,
    SCORING_PROTOCOL,
    atomic_json_dump,
    load_manifest,
    parse_seeds,
    percentile,
    result_path,
    score_response,
    stored_hybrid_score,
)


def paired_bootstrap(deltas: list[int], samples: int, seed: int) -> list[float | None]:
    if not deltas:
        return [None, None]
    generator = random.Random(seed)
    estimates = []
    size = len(deltas)
    for _ in range(samples):
        estimates.append(sum(deltas[generator.randrange(size)] for _ in range(size)) / size)
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)]


def summarize_records(
    records: dict[tuple[str, int], dict[str, dict]],
    sample_ids: list[str],
    indices: dict[str, int],
    seeds: list[int],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict]:
    rows = []
    for seed in seeds:
        baseline = records.get(("vanilla_baseline", seed), {})
        for condition in CONDITIONS:
            condition_records = records.get((condition, seed), {})
            paired_ids = [sid for sid in sample_ids if sid in baseline and sid in condition_records]
            correct = sum(bool(condition_records[sid]["correct"]) for sid in condition_records)
            count = len(condition_records)
            deltas = [
                int(bool(condition_records[sid]["correct"])) - int(bool(baseline[sid]["correct"]))
                for sid in paired_ids
            ]
            gains = sum(delta == 1 for delta in deltas)
            losses = sum(delta == -1 for delta in deltas)
            pair_groups: dict[int, list[str]] = {}
            for sid in condition_records:
                pair_groups.setdefault((indices[sid] - 1) // 2, []).append(sid)
            complete_pairs = [members for members in pair_groups.values() if len(members) == 2]
            pair_correct = sum(
                all(bool(condition_records[sid]["correct"]) for sid in members) for members in complete_pairs
            )
            rows.append({
                "condition": condition,
                "seed": seed,
                "count": count,
                "correct": correct,
                "accuracy": correct / count if count else None,
                "paired_count": len(paired_ids),
                "delta_from_vanilla": sum(deltas) / len(deltas) if deltas else None,
                "delta_ci95": paired_bootstrap(
                    deltas, bootstrap_samples, bootstrap_seed + seed * 100 + CONDITIONS.index(condition)
                ),
                "wrong_to_right": gains,
                "right_to_wrong": losses,
                "unchanged": len(deltas) - gains - losses,
                "complete_pairs": len(complete_pairs),
                "pair_correct": pair_correct,
                "pair_accuracy": pair_correct / len(complete_pairs) if complete_pairs else None,
            })
    return rows


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "condition", "seed", "count", "correct", "accuracy", "paired_count",
        "delta_from_vanilla", "ci95_low", "ci95_high", "wrong_to_right",
        "right_to_wrong", "unchanged", "complete_pairs", "pair_correct", "pair_accuracy",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flattened = dict(row)
            flattened["ci95_low"], flattened["ci95_high"] = flattened.pop("delta_ci95")
            writer.writerow(flattened)


def _format_percent(value) -> str:
    return "—" if value is None else f"{100 * value:.2f}%"


def _write_markdown(summary: dict, path: Path) -> None:
    lines = [
        "# Donor–recipient intervention results",
        "",
        f"- Dataset: **{summary['metadata']['dataset']}**",
        f"- Expected samples per condition: **{summary['metadata']['expected_samples']}**",
        f"- Missing results: **{len(summary['missing_results'])}**",
        f"- Validation failures: **{len(summary['validation_failures'])}**",
        "- Parser methods: **"
        + ", ".join(
            f"{method}={count}" for method, count in summary["metadata"]["parser_methods"].items()
        )
        + "**",
        "",
        "| condition | seed | accuracy | Δ vanilla (95% CI) | wrong→right | right→wrong | pair accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["conditions"]:
        low, high = row["delta_ci95"]
        interval = "—" if low is None else f"{_format_percent(row['delta_from_vanilla'])} [{_format_percent(low)}, {_format_percent(high)}]"
        lines.append(
            f"| {row['condition']} | {row['seed']} | {_format_percent(row['accuracy'])} "
            f"({row['correct']}/{row['count']}) | {interval} | {row['wrong_to_right']} | "
            f"{row['right_to_wrong']} | {_format_percent(row['pair_accuracy'])} |"
        )
    if summary["missing_results"]:
        lines.extend(["", "## Missing results", "", *[f"- {item}" for item in summary["missing_results"]]])
    if summary["validation_failures"]:
        lines.extend(["", "## Validation failures", "", *[f"- {item}" for item in summary["validation_failures"]]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=2026)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap_samples must be positive")
    try:
        seeds = parse_seeds(args.seeds)
    except ValueError as error:
        parser.error(str(error))

    manifest = load_manifest(args.manifest)
    samples = manifest["samples"][: args.limit]
    sample_ids = [sample["id"] for sample in samples]
    indices = {sample["id"]: int(sample["index"]) for sample in samples}
    records: dict[tuple[str, int], dict[str, dict]] = {}
    missing = []
    validation_failures = []
    rescored_results = 0
    scoring_protocols = set()
    parser_methods: dict[str, int] = {}
    for seed in seeds:
        for condition in CONDITIONS:
            bucket = records.setdefault((condition, seed), {})
            for sample in samples:
                path = result_path(args.output_dir, condition, seed, sample["id"])
                if not path.is_file():
                    missing.append(str(path))
                    continue
                with open(path, encoding="utf-8") as handle:
                    result = json.load(handle)
                hybrid_score = stored_hybrid_score(result, sample["gold"])
                if hybrid_score is None:
                    parsed, correct = score_response(result.get("response"), sample["gold"])
                    scoring_protocol = SCORING_PROTOCOL
                else:
                    parsed, correct = hybrid_score
                    scoring_protocol = HYBRID_SCORING_PROTOCOL
                parser_method = (
                    result.get("parsing", {}).get("method")
                    if scoring_protocol == HYBRID_SCORING_PROTOCOL
                    else "deterministic"
                )
                parser_methods[parser_method] = parser_methods.get(parser_method, 0) + 1
                if (
                    result.get("parsed") != parsed
                    or result.get("correct") != correct
                    or result.get("scoring_protocol") != scoring_protocol
                ):
                    result["parsed"] = parsed
                    result["correct"] = correct
                    result["scoring_protocol"] = scoring_protocol
                    atomic_json_dump(result, path)
                    rescored_results += 1
                scoring_protocols.add(scoring_protocol)
                bucket[sample["id"]] = result
                if condition == "wrong_sample" and result.get("latent_source_id") == sample["id"]:
                    validation_failures.append(f"{path}: wrong-sample self donation")
                if condition == "order_shuffled":
                    permutation = result.get("intervention", {}).get("permutation", [])
                    if permutation == list(range(len(permutation))):
                        validation_failures.append(f"{path}: identity permutation")
                if condition == "norm_matched_random":
                    error = result.get("intervention", {}).get("max_norm_relative_error")
                    if error is None or error > 0.01:
                        validation_failures.append(f"{path}: norm relative error {error}")

    rows = summarize_records(
        records, sample_ids, indices, seeds, args.bootstrap_samples, args.bootstrap_seed
    )
    summary = {
        "metadata": {
            "dataset": manifest.get("dataset"),
            "expected_samples": len(samples),
            "seeds": seeds,
            "bootstrap_samples": args.bootstrap_samples,
            "paper_vanilla_reference_accuracy": 0.6867,
            "scoring_protocol": (
                next(iter(scoring_protocols)) if len(scoring_protocols) == 1 else "mixed"
            ),
            "parser_methods": parser_methods,
            "rescored_results": rescored_results,
        },
        "conditions": rows,
        "missing_results": missing,
        "validation_failures": validation_failures,
    }
    output = Path(args.output_dir)
    atomic_json_dump(summary, output / "summary.json")
    _write_csv(rows, output / "summary.csv")
    _write_markdown(summary, output / "report.md")
    print(f"[analysis] wrote {output / 'summary.json'}, summary.csv, and report.md")
    if missing or validation_failures:
        print(f"[analysis] WARNING: missing={len(missing)} validation_failures={len(validation_failures)}")


if __name__ == "__main__":
    main()
