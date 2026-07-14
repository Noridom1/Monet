from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

from inference.vllm.latent_policy_logits import force_token_for_row, suppress_token_for_row
from run_scripts.latent_policy import (
    FORCE_FIRST_POLICY,
    NATURAL_ELSEWHERE_MODE,
    SUPPRESS_ELSEWHERE_MODE,
    SUPPRESS_LATENT_START_POLICY,
    LatentPolicyManifest,
    analyze_runs,
    attach_policy_to_sampling_params,
    build_manifest,
    canonical_index,
    manifest_sha256,
    select_forced_indices,
    validate_policy_block_count,
    write_manifest,
)


class LatentPolicyTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.model_path = self.root / "model"
        self.model_path.mkdir()
        (self.model_path / "config.json").write_text('{"model_type": "test"}\n', encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def manifest(self, x_percent, indices=range(191), suppress_unselected=False):
        return build_manifest(
            dataset="VStarBench",
            indices=indices,
            x_percent=x_percent,
            seed=0,
            model_path=self.model_path,
            latent_size=16,
            max_new_tokens=2048,
            max_pixels=1003520,
            system_prompt="test prompt",
            suppress_unselected=suppress_unselected,
        )

    def test_selection_is_deterministic_exact_and_nested(self):
        x15 = select_forced_indices(range(191), "VStarBench", 15, 0)
        x25 = select_forced_indices(range(191), "VStarBench", 25, 0)
        self.assertEqual(len(x15), 29)
        self.assertEqual(len(x25), 48)
        self.assertEqual(x15, select_forced_indices(range(191), "VStarBench", 15, 0))
        self.assertTrue(set(x15).issubset(x25))

    def test_duplicate_indices_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            select_forced_indices([1, 1], "VStarBench", 15, 0)

    def test_manifest_validation_and_dataset_hash(self):
        manifest = self.manifest(15)
        path = self.root / "policy.json"
        digest = write_manifest(manifest, path)
        self.assertEqual(len(digest), 64)
        loaded = LatentPolicyManifest.load(path)
        loaded.validate_dataset_indices("VStarBench", range(191))
        with self.assertRaisesRegex(ValueError, "index hash"):
            loaded.validate_dataset_indices("VStarBench", list(range(190)) + [999])

        manifest["datasets"]["VStarBench"]["forced_indices"].append(
            manifest["datasets"]["VStarBench"]["forced_indices"][0]
        )
        manifest["datasets"]["VStarBench"]["forced_count"] += 1
        with self.assertRaisesRegex(ValueError, "duplicates"):
            LatentPolicyManifest(manifest)

    def test_existing_manifest_cannot_change(self):
        path = self.root / "policy.json"
        write_manifest(self.manifest(15), path)
        with self.assertRaisesRegex(ValueError, "differs"):
            write_manifest(self.manifest(25), path)

    def test_manifest_modes_hash_and_policy_resolution(self):
        natural = self.manifest(15)
        controlled = self.manifest(15, suppress_unselected=True)
        self.assertEqual(natural["mode"], NATURAL_ELSEWHERE_MODE)
        self.assertEqual(controlled["mode"], SUPPRESS_ELSEWHERE_MODE)
        self.assertNotEqual(manifest_sha256(natural), manifest_sha256(controlled))

        natural_policy = LatentPolicyManifest(natural)
        controlled_policy = LatentPolicyManifest(controlled)
        forced_index = controlled["datasets"]["VStarBench"]["forced_indices"][0]
        unselected_index = next(
            index
            for index in range(191)
            if index != forced_index
            and not controlled_policy.is_forced("VStarBench", index)
        )
        self.assertEqual(
            controlled_policy.policy_for("VStarBench", forced_index), FORCE_FIRST_POLICY
        )
        self.assertEqual(
            controlled_policy.policy_for("VStarBench", unselected_index),
            SUPPRESS_LATENT_START_POLICY,
        )
        self.assertIsNone(natural_policy.policy_for("VStarBench", unselected_index))

    def test_sampling_params_preserve_existing_metadata(self):
        params = SimpleNamespace(extra_args={"existing": 1})
        attach_policy_to_sampling_params(params, None)
        self.assertEqual(params.extra_args, {"existing": 1})
        attach_policy_to_sampling_params(params, FORCE_FIRST_POLICY)
        self.assertEqual(
            params.extra_args,
            {"existing": 1, "monet_latent_policy": "force_first"},
        )
        attach_policy_to_sampling_params(params, SUPPRESS_LATENT_START_POLICY)
        self.assertEqual(
            params.extra_args,
            {"existing": 1, "monet_latent_policy": "suppress_latent_start"},
        )
        with self.assertRaisesRegex(ValueError, "unsupported"):
            attach_policy_to_sampling_params(params, "invalid")

    def test_policy_block_count_compliance(self):
        validate_policy_block_count(FORCE_FIRST_POLICY, 1)
        validate_policy_block_count(SUPPRESS_LATENT_START_POLICY, 0)
        validate_policy_block_count(None, 2)
        with self.assertRaisesRegex(RuntimeError, "without a latent activation"):
            validate_policy_block_count(FORCE_FIRST_POLICY, 0)
        with self.assertRaisesRegex(RuntimeError, "completed with 1"):
            validate_policy_block_count(SUPPRESS_LATENT_START_POLICY, 1)

    def test_force_token_for_row(self):
        logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        original_second = logits[1].clone()
        force_token_for_row(logits, 0, 1)
        self.assertEqual(int(logits[0].argmax()), 1)
        self.assertTrue(torch.isneginf(logits[0, 0]))
        self.assertTrue(torch.equal(logits[1], original_second))
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            force_token_for_row(logits, 0, 3)

    def test_suppress_token_for_row_preserves_other_logits(self):
        logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        original = logits.clone()
        suppress_token_for_row(logits, 0, 1)
        self.assertTrue(torch.isneginf(logits[0, 1]))
        self.assertEqual(float(logits[0, 0]), float(original[0, 0]))
        self.assertEqual(float(logits[0, 2]), float(original[0, 2]))
        self.assertTrue(torch.equal(logits[1], original[1]))
        suppress_token_for_row(logits, 0, 1)
        self.assertTrue(torch.isneginf(logits[0, 1]))

    def test_analysis_outputs(self):
        indices = list(range(8))
        baseline = pd.DataFrame({
            "index": indices,
            "category": ["direct_attributes"] * 4 + ["relative_position"] * 4,
            "prediction": [f"baseline-{index}\n<ltnt:{index % 2}>" for index in indices],
            "hit": [1, 0, 1, 0, 1, 0, 1, 0],
        })
        baseline_path = self.root / "baseline.xlsx"
        baseline.to_excel(baseline_path, index=False)

        run_dirs = []
        for x_percent in (25, 50):
            run_dir = self.root / f"x{x_percent}"
            manifest = build_manifest(
                dataset="VStarBench",
                indices=indices,
                x_percent=x_percent,
                seed=0,
                model_path=self.model_path,
                latent_size=16,
                max_new_tokens=2048,
                max_pixels=1003520,
                system_prompt="test prompt",
            )
            write_manifest(manifest, run_dir / "policy_manifest.json")
            forced = {
                canonical_index(index)
                for index in manifest["datasets"]["VStarBench"]["forced_indices"]
            }
            result = baseline.copy()
            for row_index, index in enumerate(result["index"]):
                if canonical_index(index) in forced:
                    result.loc[row_index, "prediction"] = f"forced-{index}\n<ltnt:1>"
                    result.loc[row_index, "hit"] = 1
            result_dir = run_dir / "Monet" / "T20260713-000000"
            result_dir.mkdir(parents=True)
            result.to_excel(result_dir / "Monet_VStarBench_judge_result.xlsx", index=False)
            run_dirs.append(run_dir)

        markdown, summary, samples = analyze_runs(
            baseline_path, run_dirs, self.root / "analysis", "VStarBench"
        )
        self.assertTrue(markdown.is_file())
        self.assertTrue(summary.is_file())
        self.assertTrue(samples.is_file())
        summary_data = pd.read_csv(summary)
        self.assertEqual(set(summary_data.query("scope == 'overall'")["x_percent"]), {0, 25, 50})

    def test_controlled_analysis_reports_exact_activation_and_policy_switch(self):
        indices = list(range(8))
        baseline = pd.DataFrame({
            "index": indices,
            "category": ["direct_attributes"] * 4 + ["relative_position"] * 4,
            "prediction": [f"baseline-{index}\n<ltnt:{index % 2}>" for index in indices],
            "hit": [index % 2 for index in indices],
        })
        baseline_path = self.root / "controlled_baseline.xlsx"
        baseline.to_excel(baseline_path, index=False)

        run_dirs = []
        for x_percent in (25, 50):
            run_dir = self.root / f"controlled_x{x_percent}"
            manifest = build_manifest(
                dataset="VStarBench",
                indices=indices,
                x_percent=x_percent,
                seed=0,
                model_path=self.model_path,
                latent_size=16,
                max_new_tokens=2048,
                max_pixels=1003520,
                system_prompt="test prompt",
                suppress_unselected=True,
            )
            write_manifest(manifest, run_dir / "policy_manifest.json")
            policy = LatentPolicyManifest(manifest)
            result = baseline.copy()
            for row_index, index in enumerate(result["index"]):
                assigned = policy.policy_for("VStarBench", index)
                activated = assigned == FORCE_FIRST_POLICY
                result.loc[row_index, "prediction"] = (
                    f"{assigned}-{index}\n<ltnt:{int(activated)}>"
                )
                result.loc[row_index, "hit"] = int(activated)
            result_dir = run_dir / "Monet" / "T20260713-000000"
            result_dir.mkdir(parents=True)
            result.to_excel(result_dir / "Monet_VStarBench_judge_result.xlsx", index=False)
            run_dirs.append(run_dir)

        markdown, summary, samples = analyze_runs(
            baseline_path, run_dirs, self.root / "controlled_analysis", "VStarBench"
        )
        summary_data = pd.read_csv(summary)
        controlled_overall = summary_data.query(
            "scope == 'overall' and mode == @SUPPRESS_ELSEWHERE_MODE"
        ).sort_values("x_percent")
        self.assertEqual(list(controlled_overall["realized_activated"]), [2, 4])
        self.assertEqual(list(controlled_overall["suppressed"]), [6, 4])
        self.assertTrue((controlled_overall["force_compliance"] == 1.0).all())
        self.assertTrue((controlled_overall["suppression_compliance"] == 1.0).all())
        switch = summary_data.query("scope == 'policy_switch_x25_to_x50'").iloc[0]
        self.assertEqual(switch["samples"], 2)
        self.assertEqual(switch["accuracy_delta"], 1.0)
        sample_data = pd.read_csv(samples)
        self.assertEqual(
            set(sample_data["assigned_policy"]),
            {FORCE_FIRST_POLICY, SUPPRESS_LATENT_START_POLICY},
        )
        self.assertIn("Shared suppressed outputs matching", markdown.read_text())


if __name__ == "__main__":
    unittest.main()
