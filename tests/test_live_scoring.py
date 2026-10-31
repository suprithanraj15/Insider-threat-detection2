import inspect
import unittest
from datetime import datetime, timezone, timedelta
from src.live_processor.live_risk_engine import (
    evaluate_path_suspiciousness,
    persist_alerts,
    persist_assessments,
    score_event,
    _rule_description,
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

    def test_attribution_and_rule_descriptions(self):
        desc = _rule_description("suspicious_service_path")
        self.assertIn("temporary directory", desc)
        unknown_desc = _rule_description("custom_test_rule")
        self.assertEqual(unknown_desc, "Custom Test Rule")

    def test_service_event_attribution(self):
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
        self.assertEqual(res["detection_source"], "Live Windows Rule-Based Risk Engine")
        self.assertTrue(isinstance(res["triggered_rules"], list))
        self.assertEqual(len(res["triggered_rules"]), 1)
        self.assertEqual(res["triggered_rules"][0]["rule"], "base_lifecycle_signal")

    def test_not_assessed_and_unscored_attribution(self):
        unassessed = score_event({"signal_status": "UNCLASSIFIED", "event_category": "UNCLASSIFIED"}, [], {})
        self.assertEqual(unassessed["triggered_rules"], [])
        self.assertEqual(unassessed["detection_source"], "Live Windows Rule-Based Risk Engine")

        unscored = score_event({"signal_status": "REVIEW_SIGNAL", "event_category": "UNKNOWN_SIGNAL"}, [], {})
        self.assertEqual(unscored["assessment_status"], "UNSCORED")
        self.assertEqual(unscored["triggered_rules"], [])
        self.assertEqual(unscored["detection_source"], "Live Windows Rule-Based Risk Engine")

    def test_classify_new_events(self):
        from src.live_processor.event_processor import classify_event

        # 4624 Logon Success
        e_logon = classify_event({
            "event_code": "4624",
            "event_source": "Microsoft-Windows-Security-Auditing",
            "host_name": "CORP-PC1",
            "event_timestamp": datetime.now(timezone.utc),
        })
        self.assertEqual(e_logon["event_category"], "LOGON_SUCCESS")
        self.assertEqual(e_logon["signal_status"], "REVIEW_SIGNAL")

        # 4625 Logon Failure
        e_fail = classify_event({
            "event_code": "4625",
            "event_source": "Microsoft-Windows-Security-Auditing",
            "host_name": "CORP-PC1",
            "event_timestamp": datetime.now(timezone.utc),
        })
        self.assertEqual(e_fail["event_category"], "LOGON_FAILURE")
        self.assertEqual(e_fail["signal_status"], "REVIEW_SIGNAL")

        # 2003 DriverFrameworks
        e_dev1 = classify_event({
            "event_code": "2003",
            "event_source": "Microsoft-Windows-DriverFrameworks-UserMode",
            "host_name": "CORP-PC1",
            "event_timestamp": datetime.now(timezone.utc),
        })
        self.assertEqual(e_dev1["event_category"], "DEVICE_ACTIVITY")
        self.assertEqual(e_dev1["signal_status"], "REVIEW_SIGNAL")

        # 6416 External Device
        e_dev2 = classify_event({
            "event_code": "6416",
            "event_source": "Microsoft-Windows-Security-Auditing",
            "host_name": "CORP-PC1",
            "event_timestamp": datetime.now(timezone.utc),
        })
        self.assertEqual(e_dev2["event_category"], "DEVICE_ACTIVITY")
        self.assertEqual(e_dev2["signal_status"], "REVIEW_SIGNAL")

    def test_score_event_logon_success_unusual_hours(self):
        late_dt = datetime(2026, 10, 31, 23, 15, 0, tzinfo=timezone.utc)
        event = {
            "event_id": 50,
            "event_code": "4624",
            "event_category": "LOGON_SUCCESS",
            "signal_status": "REVIEW_SIGNAL",
            "event_timestamp": late_dt,
        }
        res = score_event(event, [], {})
        self.assertEqual(res["assessment_status"], "UNUSUAL_LOGIN_TIME")
        self.assertEqual(res["risk_score"], 20.0)
        self.assertEqual(res["risk_level"], "LOW")
        self.assertEqual(res["priority"], "P3")
        self.assertEqual(res["detection_source"], "Live Windows Rule-Based Risk Engine")
        self.assertEqual(res["triggered_rules"][0]["rule"], "unusual_login_time")

    def test_score_event_logon_success_routine_hours(self):
        routine_dt = datetime(2026, 10, 31, 10, 0, 0, tzinfo=timezone.utc)
        event = {
            "event_id": 51,
            "event_code": "4624",
            "event_category": "LOGON_SUCCESS",
            "signal_status": "REVIEW_SIGNAL",
            "event_timestamp": routine_dt,
        }
        res = score_event(event, [], {})
        self.assertEqual(res["assessment_status"], "ROUTINE_LOGON")
        self.assertIsNone(res["risk_score"])
        self.assertIsNone(res["risk_level"])
        self.assertEqual(res["triggered_rules"], [])
        self.assertEqual(res["detection_source"], "Live Windows Rule-Based Risk Engine")

    def test_score_event_logon_failure(self):
        event = {
            "event_id": 60,
            "event_code": "4625",
            "event_category": "LOGON_FAILURE",
            "signal_status": "REVIEW_SIGNAL",
            "event_timestamp": datetime.now(timezone.utc),
        }
        res = score_event(event, [], {})
        self.assertEqual(res["assessment_status"], "FAILED_LOGIN_DETECTED")
        self.assertEqual(res["risk_score"], 15.0)
        self.assertEqual(res["risk_level"], "LOW")
        self.assertEqual(res["priority"], "P3")
        self.assertEqual(res["triggered_rules"][0]["rule"], "failed_login_burst")
        self.assertEqual(res["detection_source"], "Live Windows Rule-Based Risk Engine")

    def test_score_event_device_activity(self):
        event = {
            "event_id": 70,
            "event_code": "2003",
            "event_category": "DEVICE_ACTIVITY",
            "signal_status": "REVIEW_SIGNAL",
            "event_timestamp": datetime.now(timezone.utc),
        }
        res = score_event(event, [], {})
        self.assertEqual(res["assessment_status"], "DEVICE_ACTIVITY_DETECTED")
        self.assertEqual(res["risk_score"], 25.0)
        self.assertEqual(res["risk_level"], "MEDIUM")
        self.assertEqual(res["priority"], "P2")
        self.assertEqual(res["triggered_rules"][0]["rule"], "device_activity")
        self.assertEqual(res["detection_source"], "Live Windows Rule-Based Risk Engine")


if __name__ == "__main__":
    unittest.main()
