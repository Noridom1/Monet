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
    HYBRID_SCORING_PROTOCOL,
    build_question,
    cyclic_wrong_sample_ids,
    donor_artifact_error,
    non_identity_permutation,
    norm_matched_random,
    parse_option,
    qwen_decode_position_ids,
    response_digest,
    solid_gray_image,
    stored_hybrid_score,
)
from inspection.donor_recipient.postprocess_llm import (
    estimate_request_tokens,
    make_batches,
    make_item,
    parse_llm_response,
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
        self.assertEqual(parse_option("The answer is A"), "A")
        self.assertEqual(
            parse_option("\n<|im_start|> reasoning. Therefore, the answer is:\n\nB<|im_end|>"),
            "B",
        )
        self.assertEqual(parse_option("final: \\boxed{A}"), "A")
        self.assertIsNone(parse_option("<|im_end|>"))

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

    def test_qwen_decode_positions_are_materialized(self):
        position_ids = qwen_decode_position_ids(100, -7, "cpu")
        self.assertEqual(position_ids.shape, (3, 1, 1))
        self.assertEqual(position_ids[:, 0, 0].tolist(), [93, 93, 93])
        self.assertTrue(position_ids.is_contiguous())

    def test_completed_hybrid_score_is_reused(self):
        result = {
            "parsed": "B",
            "scoring_protocol": HYBRID_SCORING_PROTOCOL,
            "response": "The intended answer is B.",
            "parsing": {
                "method": "llm_fallback",
                "response_sha256": response_digest("The intended answer is B."),
            },
        }
        self.assertEqual(stored_hybrid_score(result, "B"), ("B", True))


class LLMPostprocessTest(unittest.TestCase):
    def test_batches_respect_estimated_prompt_cap(self):
        items = [
            make_item(f"item-{index}", "Question? Options: A. yes B. no", "reason " * 100)
            for index in range(5)
        ]
        batches = make_batches(items, max_prompt_tokens=300)
        self.assertGreater(len(batches), 1)
        self.assertTrue(all(estimate_request_tokens(batch) <= 300 for batch in batches))
        self.assertEqual(
            [item["id"] for batch in batches for item in batch],
            [item["id"] for item in items],
        )

    def test_parse_llm_json_response(self):
        content = '```json\n{"results":[{"id":"one","choice":"b"},{"id":"two","choice":null}]}\n```'
        self.assertEqual(parse_llm_response(content, {"one", "two"}), {"one": "B", "two": None})


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
