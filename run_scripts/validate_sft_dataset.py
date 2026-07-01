#!/usr/bin/env python
"""Validate the local Monet-SFT-125K layout and referenced image files."""

import argparse
import json
from pathlib import Path


SUBSETS = (
    "Visual_CoT",
    "CogCoM",
    "ReFocus",
    "Zebra_CoT_count",
    "Zebra_CoT_visual_search",
    "Zebra_CoT_geometry",
)


def referenced_images(sample):
    for message in sample.get("data", []):
        for content in message.get("content", []):
            if content.get("type") == "image" and content.get("image"):
                yield content["image"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument(
        "--skip-image-check",
        action="store_true",
        help="only validate JSON structure; do not check every referenced image",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.expanduser().resolve()
    errors = []
    total_samples = 0
    total_images = 0

    for subset in SUBSETS:
        json_path = dataset_dir / subset / "train.json"
        if not json_path.is_file():
            errors.append(f"missing {json_path}")
            continue

        try:
            with json_path.open("r", encoding="utf-8") as handle:
                samples = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"cannot read {json_path}: {exc}")
            continue

        if not isinstance(samples, list):
            errors.append(f"{json_path} must contain a JSON list")
            continue

        missing_images = []
        image_count = 0
        for sample_index, sample in enumerate(samples):
            if not isinstance(sample, dict) or not isinstance(sample.get("data"), list):
                errors.append(f"{json_path}: sample {sample_index} has no data list")
                continue
            for image_path in referenced_images(sample):
                image_count += 1
                if not args.skip_image_check and not (dataset_dir / image_path).is_file():
                    if len(missing_images) < 10:
                        missing_images.append(image_path)

        if missing_images:
            errors.append(
                f"{json_path}: missing referenced images, including: "
                + ", ".join(missing_images)
            )

        total_samples += len(samples)
        total_images += image_count
        print(f"[validate-sft] {subset}: {len(samples)} samples, {image_count} image references")

    if errors:
        for error in errors:
            print(f"[validate-sft] ERROR: {error}")
        raise SystemExit(1)

    print(
        f"[validate-sft] OK: {total_samples} samples and "
        f"{total_images} image references under {dataset_dir}"
    )


if __name__ == "__main__":
    main()
