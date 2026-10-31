import inspect
import unittest
from datetime import datetime, timezone, timedelta
from src.live_processor.live_risk_engine import (
    evaluate_path_suspiciousness,
    persist_alerts,
    persist_assessments,
    score_event,
)


class TestLiveRiskEngine(unittest.TestCase):
    def test_path_suspiciousness_standard_path(self):
        inc, reasons = evaluate_path_suspiciousness(r"C:\Program Files\McAfee\wps\mc-dad.exe")
        self.assertEqual(inc, 0.0)
        self.assertEqual(len(reasons), 0)

    def test_path_suspiciousness_temp_directory(self):
        inc, reasons = evaluate_path_suspiciousness(r"C:\Users\USER\AppData\Local\Temp\updater.exe")
        self.assertGreaterEqual(inc, 35.0)
        self.assertTrue(any("user/temporary" in r for r in reasons))

    def test_path_suspiciousness_powershell_runner(self):
        inc, reasons = evaluate_path_suspiciousness(r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -enc AAA==")
        self.assertGreaterEqual(inc, 40.0)
        self.assertTrue(any("scripting interpreter" in r for r in reasons))

    def test_score_event_not_assessed(self):
        event = {
            "event_id": 1,
            "event_code": "506",
            "event_category": "UNCLASSIFIED",
            "signal_status": "UNCLASSIFIED",
        }
        res = score_event(event, [], {})
        self.assertEqual(res["assessment_status"], "NOT_ASSESSED")
        self.assertIsNone(res["risk_score"])
        self.assertIsNone(res["risk_level"])

    def test_score_event_benign_service_single(self):
        event = {
            "event_id": 10,
            "event_code": "7045",
            "event_category": "SERVICE_INSTALLED",
            "signal_status": "REVIEW_SIGNAL",
        }
        evidence = {
            "service_name": "StandardVendorService",
            "service_file_name": r"C:\Program Files\Vendor\service.exe",
            "service_account": "LocalSystem",
            "service_start_type": "demand start",
        }
        res = score_event(event, [], evidence)
        self.assertEqual(res["risk_score"], 15.0)
        self.assertEqual(res["risk_level"], "LOW")
        self.assertEqual(res["priority"], "P3")

    def test_score_event_correlated_service(self):
        event = {
            "event_id": 20,
            "event_code": "7045",
            "event_category": "SERVICE_INSTALLED",
            "signal_status": "REVIEW_SIGNAL",
        }
        correlated = [{"event_id": 19, "event_code": "7040", "event_timestamp": datetime.now(timezone.utc)}]
        evidence = {
            "service_name": "VendorService",
            "service_file_name": r"C:\Program Files\Vendor\service.exe",
            "service_account": "LocalSystem",
            "service_start_type": "demand start",
        }
        res = score_event(event, correlated, evidence)
        self.assertEqual(res["risk_score"], 30.0)
        self.assertEqual(res["risk_level"], "MEDIUM")
        self.assertEqual(res["priority"], "P2")

    def test_score_event_malicious_suspicious_service(self):
        event = {
            "event_id": 30,
            "event_code": "7045",
            "event_category": "SERVICE_INSTALLED",
            "signal_status": "REVIEW_SIGNAL",
        }
        correlated = [{"event_id": 29, "event_code": "7040", "event_timestamp": datetime.now(timezone.utc)}]
        evidence = {
            "service_name": "SuspiciousPersistence",
            "service_file_name": r"C:\Users\USER\AppData\Local\Temp\backdoor.exe",
            "service_account": "USER",
            "service_start_type": "auto start",
        }
        res = score_event(event, correlated, evidence)
        # Base (15) + Correlated (15) + Path Temp (35) + User Account (15) + AutoStart (10) = 90
        self.assertGreaterEqual(res["risk_score"], 60.0)
        self.assertEqual(res["risk_level"], "HIGH")
        self.assertEqual(res["priority"], "P1")

    def test_assessments_commit_independently_of_alerts(self):
        assessment_source = inspect.getsource(persist_assessments)
        alert_source = inspect.getsource(persist_alerts)
        self.assertIn("conn.commit()", assessment_source)
        self.assertIn("conn.commit()", alert_source)


if __name__ == "__main__":
    unittest.main()
