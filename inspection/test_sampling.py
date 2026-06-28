"""Focused CPU tests for Phase A sampling and evaluation summarization."""
import json
import math
import os
import tempfile
import unittest

import torch

from inspection.generate_latents import (
    SamplingConfig,
    _apply_repetition_penalty,
    sample_next_token,
)
from inspection.summarize_eval import summarize


class SamplingTest(unittest.TestCase):
    def _sample(self, logits, config, tracked=0, seed=7, history=None):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return sample_next_token(
            torch.tensor(logits), history or [], config, generator, tracked)

    def test_greedy_matches_argmax(self):
        token, diag = self._sample([0.0, 3.0, 1.0], SamplingConfig(), tracked=1)
        self.assertEqual(token, 1)
        self.assertEqual(diag["rank"], 1)
        self.assertEqual(diag["sampling_probability"], 1.0)

    def test_top_k_probability_and_rank(self):
        config = SamplingConfig(temperature=1.0, top_k=2, top_p=1.0)
        _, diag = self._sample([2.0, 3.0, 0.0], config, tracked=0)
        self.assertEqual(diag["rank"], 2)
        self.assertTrue(diag["in_sampling_pool"])
        self.assertAlmostEqual(diag["sampling_probability"], math.exp(2) / (math.exp(2) + math.exp(3)))

    def test_top_p_retains_crossing_token(self):
        logits = [math.log(0.6), math.log(0.3), math.log(0.1)]
        config = SamplingConfig(temperature=1.0, top_k=0, top_p=0.7)
        _, second = self._sample(logits, config, tracked=1)
        _, third = self._sample(logits, config, tracked=2)
        self.assertTrue(second["in_sampling_pool"])
        self.assertFalse(third["in_sampling_pool"])

    def test_repetition_penalty_is_sign_aware(self):
        logits = torch.tensor([2.0, -2.0, 1.0])
        actual = _apply_repetition_penalty(logits, [0, 1], 2.0)
        torch.testing.assert_close(actual, torch.tensor([1.0, -4.0, 1.0]))

    def test_seeded_sampling_is_reproducible(self):
        config = SamplingConfig(temperature=1.0, top_k=0, top_p=1.0)
        first = self._sample([1.0, 1.0, 1.0], config, seed=123)[0]
        second = self._sample([1.0, 1.0, 1.0], config, seed=123)[0]
        self.assertEqual(first, second)


class SummaryTest(unittest.TestCase):
    def test_new_old_and_missing_traces(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = {
                "dataset": "test",
                "samples": [
                    {"id": "new", "index": 1, "gold": "A"},
                    {"id": "old", "index": 2, "gold": "B"},
                    {"id": "missing", "index": 3, "gold": "C"},
                ],
            }
            manifest_path = os.path.join(tmp, "manifest.json")
            with open(manifest_path, "w") as f:
                json.dump(manifest, f)

            sampling = {"temperature": 0.7, "top_k": 50, "top_p": 0.8,
                        "repetition_penalty": 1.0, "seed": 42}
            os.makedirs(os.path.join(tmp, "new"))
            torch.save({
                "generated_text": "answer \\boxed{A}",
                "latent_positions": [10],
                "latent_blocks": [[10]],
                "latent_start_candidates": [{"sampled": True}],
                "meta": {"sampling": sampling, "num_latent": 1, "num_latent_blocks": 1,
                         "latent_activated": True, "latent_start_pool_count": 1,
                         "latent_start_sampled_count": 1},
            }, os.path.join(tmp, "new", "trace.pt"))
            os.makedirs(os.path.join(tmp, "old"))
            torch.save({
                "generated_text": "answer \\boxed{B}",
                "latent_positions": [], "latent_blocks": [], "meta": {},
            }, os.path.join(tmp, "old", "trace.pt"))

            result = summarize(manifest_path, tmp)
            self.assertEqual(result["metadata"]["num_latent_activated"], 1)
            self.assertAlmostEqual(result["metadata"]["latent_activation_rate"], 1 / 3)
            self.assertEqual(result["metadata"]["sampling"], sampling)
            self.assertIsNone(result["results"][1]["latent_start_pool_count"])
            self.assertFalse(result["results"][2]["latent_activated"])


if __name__ == "__main__":
    unittest.main()
