import unittest
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"


class TestSupervisorIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = OUTPUT_DIR / "supervisor_agent_results.parquet"
        if not cls.path.exists():
            raise FileNotFoundError(f"Supervisor results not found at {cls.path}")
        cls.df = pd.read_parquet(cls.path)

    def test_supervisor_schema(self):
        required_cols = [
            "user", "day", "ground_truth", "existing_agent_score",
            "isolation_forest_score", "random_forest_score", "lstm_score",
            "ml_model_score", "supervisor_risk_score",
            "supervisor_threat_level", "supervisor_prediction",
            "scoring_mode", "ml_models_available",
        ]
        for col in required_cols:
            self.assertIn(col, self.df.columns, f"Missing expected column: {col}")

    def test_supervisor_row_count(self):
        self.assertEqual(len(self.df), 342706, f"Expected 342,706 rows, got {len(self.df)}")

    def test_key_uniqueness(self):
        dups = self.df.duplicated(["user", "day"]).sum()
        self.assertEqual(dups, 0, f"Found {dups} duplicate (user, day) keys")

    def test_ground_truth_validity(self):
        unique_gt = set(self.df["ground_truth"].unique())
        self.assertTrue(unique_gt.issubset({0, 1}), f"Unexpected ground_truth labels: {unique_gt}")
        malicious_count = int((self.df["ground_truth"] == 1).sum())
        self.assertEqual(malicious_count, 32, f"Expected 32 malicious rows in ground truth, got {malicious_count}")

    def test_score_bounds(self):
        scores = self.df["supervisor_risk_score"]
        self.assertTrue(scores.notna().all(), "Found NaN in supervisor_risk_score")
        self.assertTrue((scores >= 0.0).all(), f"Found scores < 0: min={scores.min()}")
        self.assertTrue((scores <= 100.0).all(), f"Found scores > 100: max={scores.max()}")

    def test_threat_level_assignment(self):
        high_mask = self.df["supervisor_risk_score"] >= 60.0
        med_mask = (self.df["supervisor_risk_score"] >= 25.0) & (self.df["supervisor_risk_score"] < 60.0)
        low_mask = self.df["supervisor_risk_score"] < 25.0

        self.assertTrue((self.df.loc[high_mask, "supervisor_threat_level"] == "HIGH").all())
        self.assertTrue((self.df.loc[med_mask, "supervisor_threat_level"] == "MEDIUM").all())
        self.assertTrue((self.df.loc[low_mask, "supervisor_threat_level"] == "LOW").all())

    def test_scoring_modes_and_fallback(self):
        agent_only = self.df[self.df["scoring_mode"] == "AGENT_ONLY"]
        hybrid = self.df[self.df["scoring_mode"] == "HYBRID"]

        self.assertGreater(len(agent_only), 0, "Expected non-zero agent-only scoring rows")
        self.assertGreater(len(hybrid), 0, "Expected non-zero hybrid scoring rows")
        self.assertEqual(len(agent_only) + len(hybrid), len(self.df))

        # In agent-only mode, ml_model_score must be NaN and supervisor score must equal agent score
        self.assertTrue(agent_only["ml_model_score"].isna().all(), "ml_model_score must be NaN for AGENT_ONLY rows")
        np.testing.assert_allclose(
            agent_only["supervisor_risk_score"].to_numpy(),
            agent_only["existing_agent_score"].to_numpy(),
            rtol=1e-5,
        )

        # In hybrid mode, ml_model_score must be present
        self.assertTrue(hybrid["ml_model_score"].notna().all(), "ml_model_score must not be NaN for HYBRID rows")

    def test_malicious_detection_recall(self):
        # Verify that all 32 true malicious days are flagged as MEDIUM or HIGH
        positives = self.df[self.df["ground_truth"] == 1]
        missed = int((positives["supervisor_threat_level"] == "LOW").sum())
        self.assertEqual(missed, 0, f"{missed} malicious days were assigned LOW threat level (missed)")


if __name__ == "__main__":
    unittest.main()
