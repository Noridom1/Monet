#!/usr/bin/env python3
"""Focused synthetic tests for the Table 3 report pipeline."""

from __future__ import annotations

import argparse
import io
import json
import logging
import tempfile
import unittest
from pathlib import Path

import pandas as pd

try:
    from run_scripts.latent_activation import annotate_latent_response, latent_block_count
    from run_scripts.secret_redaction import install_log_secret_redaction, redact_cli_secret
    from run_scripts.summarize_table3 import DATASETS, write_reports
except ModuleNotFoundError:  # Support direct execution from run_scripts/.
    from latent_activation import annotate_latent_response, latent_block_count
    from secret_redaction import install_log_secret_redaction, redact_cli_secret
    from summarize_table3 import DATASETS, write_reports


class Table3SummaryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.work_dir = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_predictions(self, run_dir: Path, dataset: str, responses: list[str]) -> None:
        pd.DataFrame({"index": range(len(responses)), "prediction": responses}).to_csv(
            run_dir / f"Monet_{dataset}.csv", index=False
        )

    def write_config(self, run_dir: Path, datasets: tuple[str, ...], rows: int) -> None:
        config = {"datasets": {dataset: {"rows": rows} for dataset in datasets}}
        (run_dir / "run_config.json").write_text(json.dumps(config), encoding="utf-8")

    def prepare_complete_run(self, latent_size: int, vstar_score: float = 0.5) -> None:
        run_dir = self.work_dir / f"latent_{latent_size}"
        run_dir.mkdir()
        self.write_config(run_dir, DATASETS, 4)
        responses = ["answer\n<ltnt:1>", "answer\n<ltnt:0>", "answer\n<ltnt:2>", "answer\n<ltnt:0>"]
        for dataset in DATASETS:
            self.write_predictions(run_dir, dataset, responses)

        pd.DataFrame({"hit": [vstar_score] * 4}).to_csv(
            run_dir / "Monet_VStarBench_judge_result.csv", index=False
        )
        pd.DataFrame({"split": ["Attribute", "Spatial"], "Overall": [0.6, 0.4]}).to_csv(
            run_dir / "Monet_VStarBench_acc.csv", index=False
        )
        for dataset, score in (("HRBench4K", 0.71), ("HRBench8K", 0.68)):
            pd.DataFrame({
                "cycle": ["Cycle 1", "Average", "Average"],
                "type": ["all", "FSP", "all"],
                "accuracy": [0.1, 0.8, score],
            }).to_csv(run_dir / f"Monet_{dataset}_acc.csv", index=False)
        rating = {"Overall": 0.555, "Reasoning": {"Avg": 0.51}, "Perception": {"Avg": 0.58}}
        (run_dir / "Monet_MME-RealWorld-Lite_rating.json").write_text(json.dumps(rating), encoding="utf-8")

    def args(self, latent_sizes: list[int], datasets: tuple[str, ...] = DATASETS) -> argparse.Namespace:
        return argparse.Namespace(
            work_dir=self.work_dir,
            latent_sizes=latent_sizes,
            datasets=list(datasets),
            allow_missing_scores=False,
        )

    def test_complete_report_uses_official_overall_formats(self) -> None:
        self.prepare_complete_run(8)
        self.assertEqual(write_reports(self.args([8])), [])

        best = pd.read_csv(self.work_dir / "table3_best.csv").set_index("dataset")
        self.assertEqual(len(best), 4)
        self.assertAlmostEqual(best.loc["VStarBench", "overall"], 50.0)
        self.assertAlmostEqual(best.loc["HRBench4K", "overall"], 71.0)
        self.assertAlmostEqual(best.loc["HRBench8K", "overall"], 68.0)
        self.assertAlmostEqual(best.loc["MME-RealWorld-Lite", "overall"], 55.5)
        self.assertAlmostEqual(best.loc["VStarBench", "activation_ratio_all"], 0.5)
        self.assertAlmostEqual(best.loc["VStarBench", "activation_ratio_successful"], 0.5)

    def test_tied_scores_choose_smaller_latent_size(self) -> None:
        self.prepare_complete_run(8, vstar_score=0.75)
        self.prepare_complete_run(10, vstar_score=0.75)
        self.assertEqual(write_reports(self.args([10, 8])), [])
        best = pd.read_csv(self.work_dir / "table3_best.csv").set_index("dataset")
        self.assertTrue((best["latent_size"] == 8).all())

    def test_stale_prediction_and_missing_score_are_failures(self) -> None:
        run_dir = self.work_dir / "latent_8"
        run_dir.mkdir()
        self.write_config(run_dir, ("VStarBench",), 1)
        self.write_predictions(run_dir, "VStarBench", ["old output without activation metadata"])
        failures = write_reports(self.args([8], ("VStarBench",)))
        self.assertTrue(any("<ltnt:N>" in failure for failure in failures))
        self.assertTrue(any("missing VStar" in failure for failure in failures))

    def test_latent_annotation_is_evaluator_safe(self) -> None:
        annotated = annotate_latent_response(
            "<abs_vis_token>hidden</abs_vis_token><abs_vis_token>hidden</abs_vis_token> answer B"
        )
        self.assertNotIn("abs_vis_token", annotated)
        self.assertTrue(annotated.endswith("<ltnt:2>"))
        self.assertEqual(latent_block_count(annotated), 2)
        self.assertTrue(annotate_latent_response("answer B", 3).endswith("<ltnt:3>"))

    def test_judge_key_is_redacted_from_argv_and_logs(self) -> None:
        secret = "unit-test-secret"
        argv = redact_cli_secret(["run.py", "--judge-key", secret])
        self.assertNotIn(secret, argv)

        previous_factory = logging.getLogRecordFactory()
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        logger = logging.getLogger("table3-secret-test")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        try:
            install_log_secret_redaction(secret)
            logger.info("judge kwargs key=%s", secret)
        finally:
            logging.setLogRecordFactory(previous_factory)
            logger.removeHandler(handler)
        self.assertNotIn(secret, stream.getvalue())
        self.assertIn("[REDACTED]", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
