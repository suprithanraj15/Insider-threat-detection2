import unittest
from fastapi.testclient import TestClient
from src.api.main import app


class TestInsiderThreatAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("database", data)
        self.assertEqual(data["database"], "connected")
        self.assertIn("live_alerts_count", data)
        self.assertIn("live_assessments_count", data)
        self.assertIn("live_scoring_status", data)
        self.assertIn("live_assessments_table_available", data)
        self.assertIsInstance(data["live_assessments_count"], int)
        self.assertIn(
            data["live_scoring_status"],
            ("operational", "not_configured", "unknown"),
        )

    def test_summary_endpoint(self):
        response = self.client.get("/api/summary")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_records"], 342706)
        self.assertIn("risk_level_counts", data)
        self.assertIn("HIGH", data["risk_level_counts"])
        self.assertIn("MEDIUM", data["risk_level_counts"])

    def test_db_alerts_endpoint(self):
        response = self.client.get("/api/alerts/db?limit=10")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("alerts", data)
        self.assertIn("count", data)

    def test_alert_status_update(self):
        # Fetch first alert
        alerts_resp = self.client.get("/api/alerts/db?limit=1")
        alerts = alerts_resp.json().get("alerts", [])
        if alerts:
            alert_id = alerts[0]["alert_id"]
            # Test update to INVESTIGATING
            patch_resp = self.client.patch(
                f"/api/alerts/db/{alert_id}/status",
                json={"status": "INVESTIGATING"}
            )
            self.assertEqual(patch_resp.status_code, 200)
            self.assertEqual(patch_resp.json()["alert"]["status"], "INVESTIGATING")

            # Reset back to OPEN
            reset_resp = self.client.patch(
                f"/api/alerts/db/{alert_id}/status",
                json={"status": "OPEN"}
            )
            self.assertEqual(reset_resp.status_code, 200)
            self.assertEqual(reset_resp.json()["alert"]["status"], "OPEN")

    def test_model_analytics_endpoint(self):
        response = self.client.get("/api/analytics/models")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("dataset", data)
        self.assertIn("model_coverage", data)
        self.assertIn("supervisor_fusion", data)
        self.assertEqual(data["dataset"]["unique_users"], 1000)
        self.assertEqual(data["dataset"]["ground_truth_malicious_days"], 32)

    def test_user_intelligence_endpoint(self):
        response = self.client.get("/api/users/intelligence?limit=5")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("users", data)
        self.assertGreaterEqual(len(data["users"]), 1)
        top_user = data["users"][0]
        self.assertIn("user", top_user)
        self.assertIn("max_risk_score", top_user)
        self.assertGreaterEqual(top_user["max_risk_score"], 60.0)

    def test_user_detail_endpoint(self):
        response = self.client.get("/api/users/intelligence/HFC0492")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["user"], "HFC0492")
        self.assertGreater(data["elevated_risk_days"], 0)
        self.assertIn("timeline", data)


if __name__ == "__main__":
    unittest.main()
