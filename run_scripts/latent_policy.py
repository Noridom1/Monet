#!/usr/bin/env python
"""Create, validate, and analyze forced-latent evaluation policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
NATURAL_ELSEWHERE_MODE = "force_selected_natural_elsewhere"
SUPPRESS_ELSEWHERE_MODE = "force_selected_suppress_elsewhere"
SUPPORTED_POLICY_MODES = frozenset({NATURAL_ELSEWHERE_MODE, SUPPRESS_ELSEWHERE_MODE})
SELECTION_METHOD = "sha256_rank_v1"
POLICY_EXTRA_ARG = "monet_latent_policy"
FORCE_FIRST_POLICY = "force_first"
SUPPRESS_LATENT_START_POLICY = "suppress_latent_start"
SUPPORTED_REQUEST_POLICIES = frozenset({FORCE_FIRST_POLICY, SUPPRESS_LATENT_START_POLICY})
ACTIVATION_PATTERN = re.compile(r"<ltnt:(\d+)>")


def _json_scalar(value: Any) -> int | float | str | bool | None:
    if hasattr(value, "item"):
        value = value.item()
    if value is None or isinstance(value, (int, float, str, bool)):
        return value
    raise TypeError(f"dataset index must be a JSON scalar, got {type(value).__name__}")


def canonical_index(value: Any) -> str:
    """Return a type-preserving, stable representation of a dataset index."""
    value = _json_scalar(value)
    if value is None:
        return "null:"
    if isinstance(value, bool):
        return f"bool:{str(value).lower()}"
    if isinstance(value, int):
        return f"int:{value}"
    if isinstance(value, float):
        return f"float:{value.hex()}"
    return f"str:{value}"


def index_set_sha256(indices: Iterable[Any]) -> str:
    keys = sorted(canonical_index(index) for index in indices)
    payload = json.dumps(keys, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ranking_digest(seed: int, dataset: str, index: Any) -> bytes:
    index_text = json.dumps(_json_scalar(index), ensure_ascii=True, separators=(",", ":"))
    payload = f"{seed}\0{dataset}\0{index_text}"
    return hashlib.sha256(payload.encode("utf-8")).digest()


def select_forced_indices(
    indices: Iterable[Any], dataset: str, x_percent: float, seed: int
) -> list[int | float | str | bool | None]:
    values = [_json_scalar(index) for index in indices]
    keys = [canonical_index(index) for index in values]
    if len(keys) != len(set(keys)):
        raise ValueError(f"dataset {dataset!r} contains duplicate indices")
    if not 0 <= x_percent <= 100:
        raise ValueError("x_percent must be in [0, 100]")
    forced_count = round(x_percent / 100.0 * len(values))
    ranked = sorted(
        values,
        key=lambda index: (ranking_digest(seed, dataset, index), canonical_index(index)),
    )
    return ranked[:forced_count]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_sha256(manifest: dict[str, Any]) -> str:
    payload = json.dumps(manifest, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_manifest(
    *,
    dataset: str,
    indices: Iterable[Any],
    x_percent: float,
    seed: int,
    model_path: str | Path,
    latent_size: int,
    max_new_tokens: int,
    max_pixels: int,
    system_prompt: str,
    latent_start_id: int = 151666,
    latent_end_id: int = 151667,
    suppress_unselected: bool = False,
) -> dict[str, Any]:
    values = [_json_scalar(index) for index in indices]
    forced = select_forced_indices(values, dataset, x_percent, seed)
    model_path = Path(model_path).expanduser().resolve()
    model_config = model_path / "config.json"
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": SUPPRESS_ELSEWHERE_MODE if suppress_unselected else NATURAL_ELSEWHERE_MODE,
        "selection_method": SELECTION_METHOD,
        "seed": seed,
        "x_percent": x_percent,
        "datasets": {
            dataset: {
                "total": len(values),
                "indices_sha256": index_set_sha256(values),
                "forced_count": len(forced),
                "forced_indices": forced,
            }
        },
        "runtime": {
            "model_path": str(model_path),
            "model_config_sha256": file_sha256(model_config) if model_config.is_file() else None,
            "latent_size": latent_size,
            "max_new_tokens": max_new_tokens,
            "max_pixels": max_pixels,
            "system_prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
            "latent_start_id": latent_start_id,
            "latent_end_id": latent_end_id,
        },
    }


def write_manifest(manifest: dict[str, Any], output: str | Path) -> str:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError(f"existing manifest differs from requested policy: {output}")
    else:
        output.write_text(rendered, encoding="utf-8")
    return manifest_sha256(manifest)


@dataclass(frozen=True)
class DatasetPolicy:
    total: int
    indices_sha256: str
    forced_indices: frozenset[str]


class LatentPolicyManifest:
    """Validated runtime view of a latent-policy manifest."""

    def __init__(self, raw: dict[str, Any], source: str | Path | None = None):
        if raw.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported latent-policy schema: {raw.get('schema_version')!r}")
        if raw.get("mode") not in SUPPORTED_POLICY_MODES:
            raise ValueError(f"unsupported latent-policy mode: {raw.get('mode')!r}")
        if raw.get("selection_method") != SELECTION_METHOD:
            raise ValueError(f"unsupported selection method: {raw.get('selection_method')!r}")
        datasets = raw.get("datasets")
        if not isinstance(datasets, dict) or not datasets:
            raise ValueError("manifest must contain at least one dataset policy")

        parsed: dict[str, DatasetPolicy] = {}
        for dataset, policy in datasets.items():
            forced_raw = policy.get("forced_indices")
            if not isinstance(forced_raw, list):
                raise ValueError(f"forced_indices for {dataset!r} must be a list")
            forced_keys = [canonical_index(index) for index in forced_raw]
            if len(forced_keys) != len(set(forced_keys)):
                raise ValueError(f"forced_indices for {dataset!r} contains duplicates")
            if policy.get("forced_count") != len(forced_keys):
                raise ValueError(f"forced_count for {dataset!r} does not match forced_indices")
            total = policy.get("total")
            if not isinstance(total, int) or total < len(forced_keys):
                raise ValueError(f"invalid total for {dataset!r}: {total!r}")
            parsed[dataset] = DatasetPolicy(
                total=total,
                indices_sha256=str(policy.get("indices_sha256", "")),
                forced_indices=frozenset(forced_keys),
            )

        self.raw = raw
        self.source = str(source) if source is not None else None
        self.mode = str(raw["mode"])
        self.datasets = parsed
        self.runtime = raw.get("runtime", {})
        self.digest = manifest_sha256(raw)

    @classmethod
    def load(cls, path: str | Path) -> "LatentPolicyManifest":
        path = Path(path).expanduser().resolve()
        return cls(json.loads(path.read_text(encoding="utf-8")), source=path)

    def is_forced(self, dataset: str, index: Any) -> bool:
        if dataset not in self.datasets:
            raise KeyError(f"dataset {dataset!r} is absent from policy manifest")
        return canonical_index(index) in self.datasets[dataset].forced_indices

    def policy_for(self, dataset: str, index: Any) -> str | None:
        if self.is_forced(dataset, index):
            return FORCE_FIRST_POLICY
        if self.mode == SUPPRESS_ELSEWHERE_MODE:
            return SUPPRESS_LATENT_START_POLICY
        return None

    def validate_dataset_indices(self, dataset: str, indices: Iterable[Any]) -> None:
        if dataset not in self.datasets:
            raise KeyError(f"dataset {dataset!r} is absent from policy manifest")
        values = list(indices)
        policy = self.datasets[dataset]
        if len(values) != policy.total:
            raise ValueError(
                f"dataset {dataset!r} has {len(values)} rows; manifest expects {policy.total}"
            )
        digest = index_set_sha256(values)
        if digest != policy.indices_sha256:
            raise ValueError(
                f"dataset {dataset!r} index hash {digest} does not match manifest "
                f"{policy.indices_sha256}"
            )

    def validate_runtime(
        self,
        *,
        model_path: str | Path,
        latent_size: int,
        max_new_tokens: int,
        max_pixels: int,
        system_prompt: str,
        latent_start_id: int,
        latent_end_id: int,
    ) -> None:
        expected = self.runtime
        actual_path = str(Path(model_path).expanduser().resolve())
        checks = {
            "model_path": actual_path,
            "latent_size": latent_size,
            "max_new_tokens": max_new_tokens,
            "max_pixels": max_pixels,
            "system_prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
            "latent_start_id": latent_start_id,
            "latent_end_id": latent_end_id,
        }
        config_path = Path(actual_path) / "config.json"
        checks["model_config_sha256"] = file_sha256(config_path) if config_path.is_file() else None
        mismatches = [
            f"{key}: expected {expected.get(key)!r}, got {value!r}"
            for key, value in checks.items()
            if expected.get(key) != value
        ]
        if mismatches:
            raise ValueError("latent-policy runtime mismatch: " + "; ".join(mismatches))


def attach_policy_to_sampling_params(sampling_params: Any, policy: str | None) -> None:
    """Attach request policy metadata while preserving existing metadata."""
    if policy is None:
        return
    if policy not in SUPPORTED_REQUEST_POLICIES:
        raise ValueError(f"unsupported Monet latent request policy: {policy!r}")
    extra_args = dict(sampling_params.extra_args or {})
    extra_args[POLICY_EXTRA_ARG] = policy
    sampling_params.extra_args = extra_args


def validate_policy_block_count(policy: str | None, block_count: int) -> None:
    if policy == FORCE_FIRST_POLICY and block_count == 0:
        raise RuntimeError("force_first completed without a latent activation")
    if policy == SUPPRESS_LATENT_START_POLICY and block_count != 0:
        raise RuntimeError(
            "suppress_latent_start completed with "
            f"{block_count} latent activation block(s)"
        )


def latent_block_count(prediction: Any) -> int | None:
    matches = ACTIVATION_PATTERN.findall(str(prediction))
    if len(matches) != 1:
        return None
    return int(matches[0])


def _load_table(path: str | Path):
    import pandas as pd

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"unsupported table format: {path}")


def _find_result(run_dir: Path, dataset: str) -> Path:
    candidates = list(run_dir.glob(f"Monet/T*/Monet_{dataset}_*_result.xlsx"))
    if not candidates:
        candidates = list(run_dir.glob(f"Monet/Monet_{dataset}_*_result.xlsx"))
    if not candidates:
        raise FileNotFoundError(f"no judged result found under {run_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def analyze_runs(
    baseline_path: str | Path,
    run_dirs: Iterable[str | Path],
    output_dir: str | Path,
    dataset: str,
) -> tuple[Path, Path, Path]:
    import pandas as pd

    baseline = _load_table(baseline_path).copy()
    required = {"index", "prediction", "hit", "category"}
    if not required.issubset(baseline.columns):
        raise ValueError(f"baseline is missing columns: {sorted(required - set(baseline.columns))}")
    baseline["_key"] = baseline["index"].map(canonical_index)
    baseline["baseline_blocks"] = baseline["prediction"].map(latent_block_count)
    if baseline["baseline_blocks"].isna().any():
        raise ValueError("baseline contains missing or ambiguous <ltnt:N> metadata")
    baseline["baseline_activated"] = baseline["baseline_blocks"].gt(0)

    sample_frames = []
    summary_rows = [{
        "scope": "overall",
        "mode": "natural_baseline",
        "x_percent": 0.0,
        "seed": None,
        "samples": len(baseline),
        "forced": 0,
        "suppressed": 0,
        "realized_activated": int(baseline["baseline_activated"].sum()),
        "activation_rate": float(baseline["baseline_activated"].mean()),
        "accuracy": float(baseline["hit"].mean()),
        "accuracy_delta": 0.0,
        "force_compliance": None,
        "suppression_compliance": None,
        "corrected": 0,
        "broken": 0,
        "prediction_drift": 0,
    }]
    run_data: dict[float, tuple[LatentPolicyManifest, pd.DataFrame]] = {}
    run_modes: set[str] = set()

    for run_dir_value in run_dirs:
        run_dir = Path(run_dir_value)
        manifest = LatentPolicyManifest.load(run_dir / "policy_manifest.json")
        run_modes.add(manifest.mode)
        if len(run_modes) > 1:
            raise ValueError("cannot analyze natural-elsewhere and suppress-elsewhere runs together")
        manifest.validate_dataset_indices(dataset, baseline["index"])
        result_path = _find_result(run_dir, dataset)
        result = _load_table(result_path).copy()
        if not required.issubset(result.columns):
            raise ValueError(f"result is missing columns: {sorted(required - set(result.columns))}")
        result["_key"] = result["index"].map(canonical_index)
        result["blocks"] = result["prediction"].map(latent_block_count)
        if result["blocks"].isna().any():
            raise ValueError(f"{result_path} contains missing or ambiguous <ltnt:N> metadata")

        merged = baseline[[
            "_key", "index", "category", "prediction", "hit", "baseline_blocks", "baseline_activated"
        ]].merge(
            result[["_key", "prediction", "hit", "blocks"]],
            on="_key",
            how="outer",
            validate="one_to_one",
            suffixes=("_baseline", "_run"),
            indicator=True,
        )
        if not merged["_merge"].eq("both").all():
            raise ValueError(f"result indices do not match baseline: {result_path}")
        merged.drop(columns="_merge", inplace=True)
        merged["assigned_policy"] = merged["index"].map(
            lambda idx: manifest.policy_for(dataset, idx) or "natural"
        )
        merged["forced"] = merged["assigned_policy"].eq(FORCE_FIRST_POLICY)
        merged["suppressed"] = merged["assigned_policy"].eq(SUPPRESS_LATENT_START_POLICY)
        merged["activated"] = merged["blocks"].gt(0)
        merged["hit_delta"] = merged["hit_run"] - merged["hit_baseline"]
        merged["prediction_match"] = merged["prediction_run"].eq(merged["prediction_baseline"])
        merged["outcome"] = "unchanged_wrong"
        merged.loc[(merged["hit_baseline"] == 1) & (merged["hit_run"] == 1), "outcome"] = "unchanged_correct"
        merged.loc[(merged["hit_baseline"] == 0) & (merged["hit_run"] == 1), "outcome"] = "corrected"
        merged.loc[(merged["hit_baseline"] == 1) & (merged["hit_run"] == 0), "outcome"] = "broken"
        merged["x_percent"] = float(manifest.raw["x_percent"])
        merged["seed"] = int(manifest.raw["seed"])
        merged["mode"] = manifest.mode
        merged["manifest_sha256"] = manifest.digest
        merged["result_path"] = str(result_path)
        sample_frames.append(merged)

        forced = merged[merged["forced"]]
        unforced = merged[~merged["forced"]]
        suppressed = merged[merged["suppressed"]]
        x_percent = float(manifest.raw["x_percent"])
        run_data[x_percent] = (manifest, merged)
        summary_rows.append({
            "scope": "overall",
            "mode": manifest.mode,
            "x_percent": x_percent,
            "seed": int(manifest.raw["seed"]),
            "samples": len(merged),
            "forced": len(forced),
            "suppressed": len(suppressed),
            "realized_activated": int(merged["activated"].sum()),
            "activation_rate": float(merged["activated"].mean()),
            "accuracy": float(merged["hit_run"].mean()),
            "accuracy_delta": float(merged["hit_delta"].mean()),
            "force_compliance": float(forced["activated"].mean()),
            "suppression_compliance": (
                float((~suppressed["activated"]).mean()) if len(suppressed) else None
            ),
            "corrected": int((merged["outcome"] == "corrected").sum()),
            "broken": int((merged["outcome"] == "broken").sum()),
            "prediction_drift": (
                int((~unforced["prediction_match"]).sum())
                if manifest.mode == NATURAL_ELSEWHERE_MODE
                else None
            ),
        })
        subsets = [
            ("forced", forced),
            ("forced_baseline_inactive", forced[~forced["baseline_activated"]]),
            ("forced_baseline_active", forced[forced["baseline_activated"]]),
        ]
        if manifest.mode == SUPPRESS_ELSEWHERE_MODE:
            subsets.extend([
                ("suppressed", suppressed),
                ("suppressed_baseline_inactive", suppressed[~suppressed["baseline_activated"]]),
                ("suppressed_baseline_active", suppressed[suppressed["baseline_activated"]]),
            ])
        else:
            subsets.append(("unforced", unforced))
        for scope, subset in subsets:
            summary_rows.append({
                "scope": scope,
                "mode": manifest.mode,
                "x_percent": x_percent,
                "seed": int(manifest.raw["seed"]),
                "samples": len(subset),
                "forced": int(subset["forced"].sum()),
                "suppressed": int(subset["suppressed"].sum()),
                "realized_activated": int(subset["activated"].sum()),
                "activation_rate": float(subset["activated"].mean()) if len(subset) else None,
                "accuracy": float(subset["hit_run"].mean()) if len(subset) else None,
                "accuracy_delta": float(subset["hit_delta"].mean()) if len(subset) else None,
                "force_compliance": float(subset["activated"].mean()) if scope == "forced" else None,
                "suppression_compliance": (
                    float((~subset["activated"]).mean()) if scope == "suppressed" else None
                ),
                "corrected": int((subset["outcome"] == "corrected").sum()),
                "broken": int((subset["outcome"] == "broken").sum()),
                "prediction_drift": (
                    int((~subset["prediction_match"]).sum())
                    if manifest.mode == NATURAL_ELSEWHERE_MODE and scope == "unforced"
                    else None
                ),
            })
        for category, subset in merged.groupby("category", dropna=False):
            summary_rows.append({
                "scope": f"category:{category}",
                "mode": manifest.mode,
                "x_percent": x_percent,
                "seed": int(manifest.raw["seed"]),
                "samples": len(subset),
                "forced": int(subset["forced"].sum()),
                "suppressed": int(subset["suppressed"].sum()),
                "realized_activated": int(subset["activated"].sum()),
                "activation_rate": float(subset["activated"].mean()),
                "accuracy": float(subset["hit_run"].mean()),
                "accuracy_delta": float(subset["hit_delta"].mean()),
                "force_compliance": None,
                "suppression_compliance": None,
                "corrected": int((subset["outcome"] == "corrected").sum()),
                "broken": int((subset["outcome"] == "broken").sum()),
                "prediction_drift": None,
            })

    x_values = sorted(run_data)
    shared_forced_match = None
    shared_suppressed_match = None
    marginal = None
    marginal_delta = None
    if len(x_values) >= 2:
        low_x, high_x = x_values[0], x_values[-1]
        low_manifest, low = run_data[low_x]
        high_manifest, high = run_data[high_x]
        low_forced_keys = set(low.loc[low["forced"], "_key"])
        high_forced_keys = set(high.loc[high["forced"], "_key"])
        if not low_forced_keys.issubset(high_forced_keys):
            raise ValueError(f"X={low_x:g} forced set is not nested inside X={high_x:g}")
        shared = low[low["_key"].isin(low_forced_keys)][["_key", "prediction_run"]].merge(
            high[high["_key"].isin(low_forced_keys)][["_key", "prediction_run"]],
            on="_key",
            suffixes=("_low", "_high"),
            validate="one_to_one",
        )
        shared_forced_match = int(
            shared["prediction_run_low"].eq(shared["prediction_run_high"]).sum()
        )

        marginal_keys = high_forced_keys - low_forced_keys
        marginal = low[low["_key"].isin(marginal_keys)][
            ["_key", "hit_run", "prediction_run"]
        ].merge(
            high[high["_key"].isin(marginal_keys)][
                ["_key", "hit_run", "prediction_run", "baseline_activated"]
            ],
            on="_key",
            suffixes=("_low", "_high"),
            validate="one_to_one",
        )
        marginal["hit_delta"] = marginal["hit_run_high"] - marginal["hit_run_low"]
        marginal_delta = float(marginal["hit_delta"].mean()) if len(marginal) else None
        summary_rows.append({
            "scope": f"policy_switch_x{low_x:g}_to_x{high_x:g}",
            "mode": high_manifest.mode,
            "x_percent": high_x,
            "seed": int(high_manifest.raw["seed"]),
            "samples": len(marginal),
            "forced": len(marginal),
            "suppressed": (
                len(marginal) if high_manifest.mode == SUPPRESS_ELSEWHERE_MODE else 0
            ),
            "realized_activated": int(
                high[high["_key"].isin(marginal_keys)]["activated"].sum()
            ),
            "activation_rate": float(
                high[high["_key"].isin(marginal_keys)]["activated"].mean()
            ),
            "accuracy": float(marginal["hit_run_high"].mean()),
            "accuracy_delta": marginal_delta,
            "force_compliance": None,
            "suppression_compliance": None,
            "corrected": int(
                ((marginal["hit_run_low"] == 0) & (marginal["hit_run_high"] == 1)).sum()
            ),
            "broken": int(
                ((marginal["hit_run_low"] == 1) & (marginal["hit_run_high"] == 0)).sum()
            ),
            "prediction_drift": None,
        })

        if low_manifest.mode == SUPPRESS_ELSEWHERE_MODE:
            shared_suppressed_keys = set(high.loc[high["suppressed"], "_key"])
            shared_suppressed = low[
                low["_key"].isin(shared_suppressed_keys)
            ][["_key", "prediction_run"]].merge(
                high[high["_key"].isin(shared_suppressed_keys)][
                    ["_key", "prediction_run"]
                ],
                on="_key",
                suffixes=("_low", "_high"),
                validate="one_to_one",
            )
            shared_suppressed_match = int(
                shared_suppressed["prediction_run_low"].eq(
                    shared_suppressed["prediction_run_high"]
                ).sum()
            )

    samples = pd.concat(sample_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "latent_policy_summary.csv"
    samples_path = output_dir / "latent_policy_samples.csv"
    markdown_path = output_dir / "latent_policy_summary.md"
    summary.to_csv(summary_path, index=False)
    samples.to_csv(samples_path, index=False)

    overall = summary[summary["scope"] == "overall"].sort_values("x_percent")
    controlled = bool(run_modes and next(iter(run_modes)) == SUPPRESS_ELSEWHERE_MODE)
    title = (
        "# VStarBench controlled latent-activation evaluation"
        if controlled
        else "# VStarBench forced-latent counterfactual evaluation"
    )
    lines = [
        title,
        "",
        "| X forced | Forced | Suppressed | Realized activation | Accuracy | "
        "Delta vs baseline | Force compliance | Suppression compliance |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall.itertuples():
        forced_text = str(int(row.forced)) if row.x_percent else "0"
        suppressed_text = str(int(row.suppressed)) if row.x_percent else "0"
        force_compliance = (
            _percent(row.force_compliance) if not pd.isna(row.force_compliance) else "n/a"
        )
        suppression_compliance = (
            _percent(row.suppression_compliance)
            if not pd.isna(row.suppression_compliance)
            else "n/a"
        )
        lines.append(
            f"| {row.x_percent:g}% | {forced_text}/{int(row.samples)} | "
            f"{suppressed_text}/{int(row.samples)} | "
            f"{int(row.realized_activated)}/{int(row.samples)} ({_percent(row.activation_rate)}) | "
            f"{_percent(row.accuracy)} | {row.accuracy_delta * 100:+.2f} pp | "
            f"{force_compliance} | {suppression_compliance} |"
        )

    lines.extend([
        "",
        "## Subset effects",
        "",
        "| X | Subset | N | Accuracy | Paired delta vs baseline | Corrected | Broken |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ])
    detail_scopes = ["forced", "forced_baseline_inactive", "forced_baseline_active"]
    if controlled:
        detail_scopes.extend([
            "suppressed",
            "suppressed_baseline_inactive",
            "suppressed_baseline_active",
        ])
    detail = summary[summary["scope"].isin(detail_scopes)]
    for row in detail.sort_values(["x_percent", "scope"]).itertuples():
        lines.append(
            f"| {row.x_percent:g}% | {row.scope} | {int(row.samples)} | "
            f"{_percent(row.accuracy)} | {row.accuracy_delta * 100:+.2f} pp | "
            f"{int(row.corrected)} | {int(row.broken)} |"
        )

    lines.extend(["", "## Reproducibility", ""])
    if not controlled:
        for row in overall[overall["x_percent"] > 0].itertuples():
            lines.append(
                f"- X={row.x_percent:g}: unforced prediction drift on "
                f"{int(row.prediction_drift)} samples."
            )
    if shared_forced_match is not None:
        shared_total = len(run_data[x_values[0]][1].query("forced"))
        lines.append(
            f"- Shared X={x_values[0]:g} forced outputs matching X={x_values[-1]:g}: "
            f"{shared_forced_match}/{shared_total}; generation drift "
            f"{shared_total - shared_forced_match}."
        )
    if shared_suppressed_match is not None:
        shared_suppressed_total = int(run_data[x_values[-1]][1]["suppressed"].sum())
        lines.append(
            f"- Shared suppressed outputs matching across X={x_values[0]:g} and "
            f"X={x_values[-1]:g}: {shared_suppressed_match}/{shared_suppressed_total}; "
            f"generation drift {shared_suppressed_total - shared_suppressed_match}."
        )
    if marginal is not None:
        lines.append(
            f"- Additional X={x_values[-1]:g} policy-switch group: {len(marginal)} samples, "
            f"{int(marginal['baseline_activated'].sum())} naturally active at baseline, "
            f"paired X={x_values[-1]:g} vs X={x_values[0]:g} accuracy delta "
            f"{marginal_delta * 100:+.2f} pp."
        )

    overall_acc = {float(row.x_percent): float(row.accuracy) for row in overall.itertuples()}
    monotonic_runs = all(
        overall_acc[x_values[index]] <= overall_acc[x_values[index + 1]]
        for index in range(len(x_values) - 1)
    )
    if controlled and monotonic_runs and marginal_delta is not None and marginal_delta > 0:
        interpretation = "Evidence favors increasing controlled latent activation for this assignment."
    elif controlled:
        interpretation = (
            "The controlled result is mixed or non-monotonic; one assignment seed is "
            "insufficient for a guidance change."
        )
    else:
        baseline_acc = overall_acc.get(0.0)
        inactive_rows = summary[summary["scope"] == "forced_baseline_inactive"].sort_values("x_percent")
        monotonic = monotonic_runs and all(overall_acc[x] >= baseline_acc for x in x_values)
        inactive_positive = not inactive_rows.empty and (inactive_rows["accuracy_delta"] > 0).all()
        if monotonic and inactive_positive:
            interpretation = "Evidence favors increasing forced latent activation for this assignment."
        else:
            interpretation = (
                "The result is mixed or non-monotonic; one assignment seed is "
                "insufficient for a guidance change."
            )
    lines.extend(["", "## Interpretation", "", interpretation, ""])
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return markdown_path, summary_path, samples_path


def _validated_rescue_baseline(path: str | Path):
    import pandas as pd

    baseline = _load_table(path).copy()
    required = {"index", "prediction", "hit", "category"}
    if not required.issubset(baseline.columns):
        raise ValueError(f"baseline is missing columns: {sorted(required - set(baseline.columns))}")
    baseline["_key"] = baseline["index"].map(canonical_index)
    if baseline["_key"].duplicated().any():
        raise ValueError("baseline contains duplicate indices")
    baseline["hit"] = pd.to_numeric(baseline["hit"], errors="raise")
    if not baseline["hit"].isin([0, 1]).all():
        raise ValueError("baseline hit column must contain only 0 or 1")
    baseline["baseline_blocks"] = baseline["prediction"].map(latent_block_count)
    if baseline["baseline_blocks"].isna().any():
        raise ValueError("baseline contains missing or ambiguous <ltnt:N> metadata")
    baseline["baseline_activated"] = baseline["baseline_blocks"].gt(0)
    return baseline


def prepare_rescue_policy(
    *,
    baseline_path: str | Path,
    dataset: str,
    targets_output: str | Path,
    manifest_output: str | Path,
    model_path: str | Path,
    latent_size: int,
    max_new_tokens: int,
    max_pixels: int,
    system_prompt: str,
    latent_start_id: int = 151666,
    latent_end_id: int = 151667,
) -> tuple[Path, Path]:
    """Select natural-inactive baseline errors and force latent thinking on all of them."""
    baseline = _validated_rescue_baseline(baseline_path)
    targets = baseline[(~baseline["baseline_activated"]) & baseline["hit"].eq(0)].copy()
    if targets.empty:
        raise ValueError("baseline has no natural-inactive incorrect samples to rescue")

    manifest = build_manifest(
        dataset=dataset,
        indices=targets["index"],
        x_percent=100.0,
        seed=0,
        model_path=model_path,
        latent_size=latent_size,
        max_new_tokens=max_new_tokens,
        max_pixels=max_pixels,
        system_prompt=system_prompt,
        latent_start_id=latent_start_id,
        latent_end_id=latent_end_id,
        suppress_unselected=False,
    )
    manifest["targeting"] = {
        "method": "natural_inactive_incorrect_v1",
        "baseline_sha256": file_sha256(baseline_path),
        "predicate": "latent_block_count == 0 and hit == 0",
        "baseline_total": len(baseline),
        "natural_inactive": int((~baseline["baseline_activated"]).sum()),
        "baseline_incorrect": int(baseline["hit"].eq(0).sum()),
        "target_count": len(targets),
    }

    manifest_path = Path(manifest_output)
    write_manifest(manifest, manifest_path)
    targets_path = Path(targets_output)
    targets_path.parent.mkdir(parents=True, exist_ok=True)
    targets[[
        "index", "category", "prediction", "hit", "baseline_blocks", "baseline_activated"
    ]].to_csv(targets_path, index=False)
    return targets_path, manifest_path


def analyze_rescue_run(
    *,
    baseline_path: str | Path,
    targets_path: str | Path,
    run_dir: str | Path,
    output_dir: str | Path,
    dataset: str,
) -> tuple[Path, Path, Path]:
    """Report how often forced latent thinking corrects targeted baseline errors."""
    import pandas as pd

    baseline = _validated_rescue_baseline(baseline_path)
    expected = baseline[(~baseline["baseline_activated"]) & baseline["hit"].eq(0)].copy()
    targets = _load_table(targets_path).copy()
    if "index" not in targets.columns:
        raise ValueError(f"targets file lacks an index column: {targets_path}")
    target_keys = targets["index"].map(canonical_index)
    if target_keys.duplicated().any():
        raise ValueError("targets file contains duplicate indices")
    if set(target_keys) != set(expected["_key"]):
        raise ValueError("targets do not match natural-inactive incorrect baseline samples")

    run_dir = Path(run_dir)
    manifest = LatentPolicyManifest.load(run_dir / "policy_manifest.json")
    manifest.validate_dataset_indices(dataset, targets["index"])
    if any(manifest.policy_for(dataset, index) != FORCE_FIRST_POLICY for index in targets["index"]):
        raise ValueError("rescue manifest must force every target sample")

    result_path = _find_result(run_dir, dataset)
    result = _load_table(result_path).copy()
    required = {"index", "prediction", "hit"}
    if not required.issubset(result.columns):
        raise ValueError(f"result is missing columns: {sorted(required - set(result.columns))}")
    result["_key"] = result["index"].map(canonical_index)
    if result["_key"].duplicated().any():
        raise ValueError("rescue result contains duplicate indices")
    result["hit"] = pd.to_numeric(result["hit"], errors="raise")
    if not result["hit"].isin([0, 1]).all():
        raise ValueError("rescue result hit column must contain only 0 or 1")
    result["forced_blocks"] = result["prediction"].map(latent_block_count)
    if result["forced_blocks"].isna().any():
        raise ValueError("rescue result contains missing or ambiguous <ltnt:N> metadata")
    if result["forced_blocks"].le(0).any():
        raise ValueError("rescue result contains a target without forced latent activation")

    samples = expected[[
        "_key", "index", "category", "prediction", "hit", "baseline_blocks"
    ]].merge(
        result[["_key", "prediction", "hit", "forced_blocks"]],
        on="_key",
        how="outer",
        validate="one_to_one",
        suffixes=("_baseline", "_forced"),
        indicator=True,
    )
    if not samples["_merge"].eq("both").all():
        raise ValueError(f"rescue result indices do not match targets: {result_path}")
    samples.drop(columns=["_merge"], inplace=True)
    samples["activated"] = samples["forced_blocks"].gt(0)
    samples["hit_delta"] = samples["hit_forced"] - samples["hit_baseline"]
    samples["outcome"] = samples["hit_forced"].map({1: "corrected", 0: "still_wrong"})
    samples["manifest_sha256"] = manifest.digest
    samples["result_path"] = str(result_path)

    summary_rows = []
    scopes = [("overall", samples)]
    scopes.extend(
        (f"category:{category}", subset)
        for category, subset in samples.groupby("category", dropna=False)
    )
    for scope, subset in scopes:
        summary_rows.append({
            "scope": scope,
            "samples": len(subset),
            "baseline_accuracy": float(subset["hit_baseline"].mean()),
            "forced_accuracy": float(subset["hit_forced"].mean()),
            "accuracy_delta": float(subset["hit_delta"].mean()),
            "corrected": int(subset["hit_forced"].eq(1).sum()),
            "still_wrong": int(subset["hit_forced"].eq(0).sum()),
            "activation_rate": float(subset["activated"].mean()),
            "force_compliance": float(subset["activated"].mean()),
        })
    summary = pd.DataFrame(summary_rows)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / "latent_rescue_summary.md"
    summary_path = output_dir / "latent_rescue_summary.csv"
    samples_path = output_dir / "latent_rescue_samples.csv"
    summary.to_csv(summary_path, index=False)
    samples.to_csv(samples_path, index=False)

    lines = [
        "# VStarBench targeted latent-rescue evaluation",
        "",
        "Target predicate: natural latent block count is zero and baseline hit is zero.",
        "",
        "| Scope | N | Baseline accuracy | Forced accuracy | Paired delta | Corrected | Still wrong | Force compliance |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples():
        lines.append(
            f"| {row.scope} | {int(row.samples)} | {_percent(row.baseline_accuracy)} | "
            f"{_percent(row.forced_accuracy)} | {row.accuracy_delta * 100:+.2f} pp | "
            f"{int(row.corrected)} | {int(row.still_wrong)} | {_percent(row.force_compliance)} |"
        )
    lines.extend([
        "",
        "This is an oracle-conditioned diagnostic: target selection uses judged baseline errors,",
        "so it does not represent a deployable routing policy.",
        "",
    ])
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return markdown_path, summary_path, samples_path


def _default_system_prompt() -> str:
    return (
        "You are a helpful multimodal assistant. You are required to answer the "
        "question based on the image provided. Put your final answer in \\boxed{}."
    )


def _create_command(args: argparse.Namespace) -> None:
    table = _load_table(args.indices_file)
    if "index" not in table.columns:
        raise ValueError(f"indices file lacks an index column: {args.indices_file}")
    manifest = build_manifest(
        dataset=args.dataset,
        indices=table["index"],
        x_percent=args.x_percent,
        seed=args.seed,
        model_path=args.model_path,
        latent_size=args.latent_size,
        max_new_tokens=args.max_new_tokens,
        max_pixels=args.max_pixels,
        system_prompt=args.system_prompt,
        latent_start_id=args.latent_start_id,
        latent_end_id=args.latent_end_id,
        suppress_unselected=args.suppress_unselected,
    )
    digest = write_manifest(manifest, args.output)
    policy = manifest["datasets"][args.dataset]
    print(
        f"[latent-policy] mode={manifest['mode']} X={args.x_percent:g} seed={args.seed} "
        f"forced={policy['forced_count']}/{policy['total']} sha256={digest}"
    )


def _analyze_command(args: argparse.Namespace) -> None:
    paths = analyze_runs(args.baseline, args.run_dir, args.output_dir, args.dataset)
    print("[latent-policy] wrote " + ", ".join(str(path) for path in paths))


def _create_rescue_command(args: argparse.Namespace) -> None:
    paths = prepare_rescue_policy(
        baseline_path=args.baseline,
        dataset=args.dataset,
        targets_output=args.targets_output,
        manifest_output=args.output,
        model_path=args.model_path,
        latent_size=args.latent_size,
        max_new_tokens=args.max_new_tokens,
        max_pixels=args.max_pixels,
        system_prompt=args.system_prompt,
        latent_start_id=args.latent_start_id,
        latent_end_id=args.latent_end_id,
    )
    manifest = LatentPolicyManifest.load(paths[1])
    policy = manifest.datasets[args.dataset]
    print(
        f"[latent-policy] rescue targets={policy.total} forced={len(policy.forced_indices)} "
        f"sha256={manifest.digest}"
    )


def _analyze_rescue_command(args: argparse.Namespace) -> None:
    paths = analyze_rescue_run(
        baseline_path=args.baseline,
        targets_path=args.targets,
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        dataset=args.dataset,
    )
    print("[latent-policy] wrote " + ", ".join(str(path) for path in paths))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="create an immutable policy manifest")
    create.add_argument("--dataset", required=True)
    create.add_argument("--indices-file", required=True)
    create.add_argument("--x-percent", type=float, required=True)
    create.add_argument("--seed", type=int, default=0)
    create.add_argument("--output", required=True)
    create.add_argument("--model-path", required=True)
    create.add_argument("--latent-size", type=int, required=True)
    create.add_argument("--max-new-tokens", type=int, default=2048)
    create.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    create.add_argument("--system-prompt", default=os.environ.get("MONET_SYSTEM_PROMPT", _default_system_prompt()))
    create.add_argument("--latent-start-id", type=int, default=151666)
    create.add_argument("--latent-end-id", type=int, default=151667)
    create.add_argument(
        "--suppress-unselected",
        action="store_true",
        help="prevent unselected samples from emitting the latent-start token",
    )
    create.set_defaults(func=_create_command)

    analyze = subparsers.add_parser("analyze", help="compare completed policy runs to baseline")
    analyze.add_argument("--baseline", required=True)
    analyze.add_argument("--run-dir", action="append", required=True)
    analyze.add_argument("--output-dir", required=True)
    analyze.add_argument("--dataset", default="VStarBench")
    analyze.set_defaults(func=_analyze_command)

    create_rescue = subparsers.add_parser(
        "create-rescue", help="force all natural-inactive incorrect baseline samples"
    )
    create_rescue.add_argument("--baseline", required=True)
    create_rescue.add_argument("--dataset", default="VStarBench")
    create_rescue.add_argument("--targets-output", required=True)
    create_rescue.add_argument("--output", required=True, help="policy manifest output")
    create_rescue.add_argument("--model-path", required=True)
    create_rescue.add_argument("--latent-size", type=int, required=True)
    create_rescue.add_argument("--max-new-tokens", type=int, default=2048)
    create_rescue.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    create_rescue.add_argument(
        "--system-prompt", default=os.environ.get("MONET_SYSTEM_PROMPT", _default_system_prompt())
    )
    create_rescue.add_argument("--latent-start-id", type=int, default=151666)
    create_rescue.add_argument("--latent-end-id", type=int, default=151667)
    create_rescue.set_defaults(func=_create_rescue_command)

    analyze_rescue = subparsers.add_parser(
        "analyze-rescue", help="analyze a completed targeted latent-rescue run"
    )
    analyze_rescue.add_argument("--baseline", required=True)
    analyze_rescue.add_argument("--targets", required=True)
    analyze_rescue.add_argument("--run-dir", required=True)
    analyze_rescue.add_argument("--output-dir", required=True)
    analyze_rescue.add_argument("--dataset", default="VStarBench")
    analyze_rescue.set_defaults(func=_analyze_rescue_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
