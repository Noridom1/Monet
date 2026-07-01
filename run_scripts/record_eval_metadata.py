#!/usr/bin/env python3
"""Record reproducibility metadata without exposing API credentials."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from vlmeval.smp import LMUDataRoot, load


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--latent-size", type=int, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    args = parser.parse_args()
    data_root = Path(LMUDataRoot())
    model_config = args.model_path / "config.json"
    metadata = {
        "datasets": {name: {"path": str(data_root / f"{name}.tsv"),
                             "rows": len(load(data_root / f"{name}.tsv")),
                             "sha256": sha256(data_root / f"{name}.tsv")} for name in args.datasets},
        "gpu_memory_utilization": os.environ.get("GPU_MEMORY_UTILIZATION"),
        "judge": {"base_url": os.environ.get("JUDGE_BASE_URL"), "concurrency": os.environ.get("JUDGE_CONCURRENCY"),
                  "model": os.environ.get("JUDGE_MODEL"), "rpm": os.environ.get("JUDGE_RPM"), "temperature": 0},
        "latent_size": args.latent_size,
        "max_new_tokens": int(os.environ.get("MONET_MAX_NEW_TOKENS", "2048")),
        "max_pixels": int(os.environ.get("MONET_MAX_PIXELS", str(1280 * 28 * 28))),
        "model_path": str(args.model_path),
        "model_config_sha256": sha256(model_config),
        "vlmevalkit_commit": git_head(args.eval_dir),
    }
    args.output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
