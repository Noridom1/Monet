"""CPU-only tests for donor-recipient data, controls, and aggregation."""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from inspection.donor_recipient import PROTOCOL_VERSION
from inspection.donor_recipient.analyze_results import summarize_records
from inspection.donor_recipient.common import (
    build_question,
    cyclic_wrong_sample_ids,
    donor_artifact_error,
    non_identity_permutation,
    norm_matched_random,
    parse_option,
    solid_gray_image,
)
from inspection.donor_recipient.prepare_mmvp import prepare_from_source


class InterventionTest(unittest.TestCase):
    def test_non_identity_permutation(self):
        permutation = non_identity_permutation(10, seed=7)
        self.assertEqual(sorted(permutation.tolist()), list(range(10)))
        self.assertFalse(torch.equal(permutation, torch.arange(10)))

    def test_norm_matched_random(self):
        latents = torch.randn(10, 32, dtype=torch.bfloat16)
        random = norm_matched_random(latents, seed=11)
        self.assertEqual(random.dtype, latents.dtype)
        self.assertFalse(torch.equal(random, latents))
        torch.testing.assert_close(
            torch.linalg.vector_norm(random.float(), dim=-1),
            torch.linalg.vector_norm(latents.float(), dim=-1),
            rtol=0.01,
            atol=0.01,
        )

    def test_wrong_sample_mapping_has_no_fixed_points(self):
        ids = [f"s{i}" for i in range(10)]
        mapping = cyclic_wrong_sample_ids(ids, seed=0)
        self.assertEqual(set(mapping.values()), set(ids))
        self.assertTrue(all(source != target for source, target in mapping.items()))

    def test_mask_preserves_dimensions(self):
        image = Image.new("RGB", (321, 123), "red")
        masked = solid_gray_image(image)
        self.assertEqual(masked.size, image.size)
        self.assertEqual(masked.getpixel((0, 0)), (127, 127, 127))

    def test_lvr_prompt_and_scoring(self):
        prompt = build_question("Which?", "(a) left\n(b) right")
        self.assertIn("A. left", prompt)
        self.assertIn("B. right", prompt)
        self.assertEqual(parse_option("<answer>B</answer>"), "B")
        self.assertEqual(parse_option("A"), "A")
        self.assertIsNone(parse_option("The answer is A"))

    def test_artifact_validation(self):
        artifact = {
            "protocol_version": PROTOCOL_VERSION,
            "sample_id": "s1",
            "latents": torch.ones(10, 8, dtype=torch.bfloat16),
            "norms": torch.ones(10),
        }
        self.assertIsNone(donor_artifact_error(artifact, "s1", 10))
        artifact["latents"][0, 0] = float("nan")
        self.assertIn("non-finite", donor_artifact_error(artifact, "s1", 10))


class DatasetTest(unittest.TestCase):
    def test_prepare_official_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            images = source / "MMVP Images"
            images.mkdir(parents=True)
            with open(source / "Questions.csv", "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["Index", "Question", "Options", "Correct Answer"]
                )
                writer.writeheader()
                for index in range(1, 301):
                    writer.writerow({
                        "Index": index,
                        "Question": f"Question {index}?",
                        "Options": "(a) first\n(b) second",
                        "Correct Answer": "(a)" if index % 2 else "(b)",
                    })
                    (images / f"{index}.jpg").touch()
            manifest_path = root / "manifest.json"
            manifest = prepare_from_source(source, manifest_path)
            self.assertEqual(len(manifest["samples"]), 300)
            self.assertEqual(manifest["samples"][0]["gold"], "A")
            self.assertEqual(manifest["samples"][1]["gold"], "B")
            with open(manifest_path, encoding="utf-8") as handle:
                self.assertEqual(json.load(handle)["dataset"], "MMVP")


class AnalysisTest(unittest.TestCase):
    def test_paired_transitions_and_pair_accuracy(self):
        ids = ["s1", "s2"]
        indices = {"s1": 1, "s2": 2}
        records = {}
        for condition in (
            "vanilla_baseline", "same_sample", "order_shuffled", "norm_matched_random",
            "recipient_image_masked", "wrong_sample",
        ):
            records[(condition, 0)] = {
                "s1": {"correct": condition != "vanilla_baseline"},
                "s2": {"correct": True},
            }
        records[("vanilla_baseline", 0)]["s1"]["correct"] = False
        rows = summarize_records(records, ids, indices, [0], bootstrap_samples=100, bootstrap_seed=4)
        same = next(row for row in rows if row["condition"] == "same_sample")
        self.assertEqual(same["wrong_to_right"], 1)
        self.assertEqual(same["right_to_wrong"], 0)
        self.assertEqual(same["pair_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()

