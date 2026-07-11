"""Download and normalize the official 300-item MMVP benchmark."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from inspection.donor_recipient.common import atomic_json_dump, build_question, normalize_label


DEFAULT_ROOT = Path(__file__).resolve().parent / "data" / "mmvp"


def prepare_from_source(source_dir: Path, output_path: Path) -> dict:
    csv_path = source_dir / "Questions.csv"
    image_dir = source_dir / "MMVP Images"
    if not csv_path.is_file() or not image_dir.is_dir():
        raise FileNotFoundError(f"expected Questions.csv and 'MMVP Images/' under {source_dir}")

    samples = []
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            missing = {"Index", "Question", "Options", "Correct Answer"} - row.keys()
            if missing:
                raise ValueError(f"Questions.csv is missing columns: {sorted(missing)}")
            index = int(row["Index"])
            image_path = image_dir / f"{index}.jpg"
            if not image_path.is_file():
                raise FileNotFoundError(f"missing MMVP image {image_path}")
            samples.append({
                "id": f"mmvp_{index:03d}",
                "index": index,
                "image": os.path.relpath(image_path, output_path.parent),
                "question": row["Question"],
                "options": row["Options"],
                "question_text": build_question(row["Question"], row["Options"]),
                "gold": normalize_label(row["Correct Answer"]),
            })

    samples.sort(key=lambda sample: sample["index"])
    indices = [sample["index"] for sample in samples]
    if indices != list(range(1, 301)):
        raise ValueError(f"expected MMVP indices 1..300, got {len(indices)} rows spanning {indices[:1]}..{indices[-1:]}")
    manifest = {
        "dataset": "MMVP",
        "source": "MMVP/MMVP",
        "prompt_protocol": "lvr_mmvp_image_first_v1",
        "samples": samples,
    }
    atomic_json_dump(manifest, output_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default=str(DEFAULT_ROOT), help="download and manifest directory")
    parser.add_argument("--repo_id", default="MMVP/MMVP", help="Hugging Face dataset repository")
    parser.add_argument("--force_download", action="store_true")
    args = parser.parse_args()

    root = Path(args.data_dir).resolve()
    source = root / "source"
    manifest_path = root / "manifest.json"
    if manifest_path.is_file() and not args.force_download:
        print(f"[MMVP] manifest already exists: {manifest_path}")
        return

    from huggingface_hub import snapshot_download

    root.mkdir(parents=True, exist_ok=True)
    print(f"[MMVP] downloading {args.repo_id} -> {source}")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=source,
        force_download=args.force_download,
    )
    manifest = prepare_from_source(source, manifest_path)
    print(f"[MMVP] prepared {len(manifest['samples'])} samples -> {manifest_path}")


if __name__ == "__main__":
    main()

