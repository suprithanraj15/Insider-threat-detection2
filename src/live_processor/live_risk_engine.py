from pathlib import Path
from datetime import timedelta
import os
import re
import argparse

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

try:
    from src.live_processor.event_evidence import extract_service_evidence
except ImportError:
    from event_evidence import extract_service_evidence


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

DB_CONFIG = {
    "host": os.environ["DB_HOST"],
    "port": int(os.environ["DB_PORT"]),
    "dbname": os.environ["DB_NAME"],
    "user": os.environ["DB_USER"],
    "password": os.environ["DB_PASSWORD"],
    "connect_timeout": 5,
}

BATCH_SIZE = 100
CORRELATION_WINDOW = timedelta(minutes=5)

MEDIUM_THRESHOLD = 25.0
HIGH_THRESHOLD = 60.0

# Suspicious locations commonly abused for service persistence.
SUSPICIOUS_PATH_PATTERNS = [
    r"[\\/](?:Users|AppData|Temp|Temporary Internet Files|PerfLogs)[\\/]",
    r"[\\/]ProgramData[\\/][^\\/]+\.exe$",
]

# Command and scripting interpreters used as service binaries.
SCRIPT_RUNNER_PATTERNS = [
    r"cmd(?:\.exe)?",
    r"powershell(?:\.exe)?",
    r"pwsh(?:\.exe)?",
    r"wscript(?:\.exe)?",
    r"cscript(?:\.exe)?",
    r"mshta(?:\.exe)?",
    r"rundll32(?:\.exe)?",
    r"certutil(?:\.exe)?",
    r"bitsadmin(?:\.exe)?",
]


# --------------------------------------------------
# FETCH PROCESSED EVENTS
# --------------------------------------------------

def fetch_processed_events(conn, limit=BATCH_SIZE):
    """Fetch the latest processed Windows events."""

    return conn.execute(
        """
        SELECT
            e.event_id,
            e.event_timestamp,
            e.host_name,
            e.user_name,
            e.event_source,
            e.event_code,
            p.event_category,
            p.signal_status,
            p.classification_reason,
            p.evidence
        FROM live_event_processing AS p
        JOIN security_events AS e
            ON e.event_id = p.event_id
        ORDER BY e.event_timestamp DESC, e.event_id DESC
        LIMIT %s
        """,
        (limit,),
    ).fetchall()


# --------------------------------------------------
# EXTRACT SERVICE EVIDENCE
# --------------------------------------------------

def get_service_evidence(event):
    """Extract service details from the stored Windows event message."""

    stored_evidence = event.get("evidence") or {}
    message = stored_evidence.get("message") or ""

    return extract_service_evidence(message)


# --------------------------------------------------
# CORRELATE EVENTS
# --------------------------------------------------

def correlate_events(events):
    """
    Correlate service installation (7045) and service start-type
    modification (7040) events on the same host and service within
    five minutes.

    Correlation is contextual evidence, not proof of maliciousness.
    """

    assessed = []

    for event in events:
        evidence = get_service_evidence(event)
        service_name = evidence.get("service_name")
        matches = []

        if (
            event["event_category"]
            in ("SERVICE_INSTALLED", "SERVICE_START_TYPE_CHANGED")
            and service_name
        ):
            for other in events:
                if other["event_id"] == event["event_id"]:
                    continue

                if (other.get("host_name") or "").casefold() != (
                    event.get("host_name") or ""
                ).casefold():
                    continue

                if other["event_category"] not in (
                    "SERVICE_INSTALLED",
                    "SERVICE_START_TYPE_CHANGED",
                ):
                    continue

                other_evidence = get_service_evidence(other)

                if (other_evidence.get("service_name") or "").casefold() != (
                    service_name
                ).casefold():
                    continue

                if (
                    event["event_timestamp"] is None
                    or other["event_timestamp"] is None
                ):
                    continue

                time_difference = abs(
                    event["event_timestamp"] - other["event_timestamp"]
                )

                event_codes = {
                    str(event["event_code"]),
                    str(other["event_code"]),
                }

                if (
                    timedelta(0) < time_difference <= CORRELATION_WINDOW
                    and event_codes == {"7040", "7045"}
                ):
                    matches.append(
                        {
                            "event_id": other["event_id"],
                            "event_code": other["event_code"],
                            "event_timestamp": other["event_timestamp"],
                        }
                    )

        assessed.append(
            {
                "event": event,
                "extracted_evidence": evidence,
                "correlated_events": matches,
            }
        )

    return assessed


# --------------------------------------------------
# LIVE EVENT SCORING ENGINE
# --------------------------------------------------

def evaluate_path_suspiciousness(file_path: str) -> tuple[float, list[str]]:
    """
    Evaluate a service binary path against persistence-related patterns.

    Returns:
        (score_increment, list_of_reasons)
    """

    if not file_path:
        return 0.0, []

    reasons = []
    increment = 0.0

    # Check for command or scripting interpreters.
    for pattern in SCRIPT_RUNNER_PATTERNS:
        if re.search(r"\b" + pattern + r"\b", file_path, re.IGNORECASE):
            increment += 40.0
            reasons.append(
                f"Service invokes scripting interpreter ({pattern})"
            )
            break

    # Check for user-writable or temporary locations.
    for pattern in SUSPICIOUS_PATH_PATTERNS:
        if re.search(pattern, file_path, re.IGNORECASE):
            increment += 35.0
            reasons.append(
                "Service binary is located in a user/temporary directory."
            )
            break

    return increment, reasons


def _rule_description(rule_key: str) -> str:
    """Human-readable description for each scoring rule component."""
    descriptions = {
        "base_lifecycle_signal": "Windows service lifecycle event detected (base signal).",
        "correlated_lifecycle_modification": "Correlated service install and start-type change on same host within 5 minutes.",
        "suspicious_service_path": "Service binary located in a user-writable or temporary directory.",
        "non_system_service_account": "Service configured under a non-standard user account (not LocalSystem/LocalService/NetworkService).",
        "auto_start_persistence": "Service configured for automatic start — possible persistence mechanism.",
        "unusual_login_time": "Login recorded outside configured working hours (08:00–22:00).",
        "failed_login_burst": "Multiple failed login attempts detected within a short time window.",
        "device_activity": "Removable device activity detected.",
    }
    return descriptions.get(rule_key, rule_key.replace("_", " ").title())


def score_event(event, correlated_events, extracted_evidence):
    """
    Score Windows System service lifecycle events.

    This rule-based live engine is scoped to Windows service events.
    Historical CERT HTTP, email, file, login and device agents are not
    automatically applicable to these raw Windows System log events.
    """

    if event["signal_status"] != "REVIEW_SIGNAL":
        return {
            "assessment_status": "NOT_ASSESSED",
            "risk_score": None,
            "risk_level": None,
            "priority": None,
            "scoring_reason": (
                "No configured review signal matched for this event code."
            ),
            "recommended_action": None,
            "scoring_breakdown": {},
            "triggered_rules": [],
            "detection_source": "Live Windows Rule-Based Risk Engine",
        }

    category = event["event_category"]

    if category not in (
        "SERVICE_INSTALLED",
        "SERVICE_START_TYPE_CHANGED",
        "LOGON_SUCCESS",
        "LOGON_FAILURE",
        "DEVICE_ACTIVITY",
    ):
        return {
            "assessment_status": "UNSCORED",
            "risk_score": None,
            "risk_level": None,
            "priority": None,
            "scoring_reason": (
                f"Event category {category} has no validated scoring rule."
            ),
            "recommended_action": None,
            "scoring_breakdown": {},
            "triggered_rules": [],
            "detection_source": "Live Windows Rule-Based Risk Engine",
        }

    # --- LOGON_SUCCESS: unusual time detection ---
    if category == "LOGON_SUCCESS":
        hour = (
            event["event_timestamp"].hour
            if hasattr(event.get("event_timestamp"), "hour")
            else None
        )
        if hour is None and isinstance(event.get("event_timestamp"), str):
            try:
                from datetime import datetime
                hour = datetime.fromisoformat(event["event_timestamp"]).hour
            except (ValueError, TypeError):
                hour = None

        if hour is not None and (hour >= 22 or hour < 6):
            base_score = 20.0
            breakdown = {"unusual_login_time": base_score}
            total_score = base_score
            triggered_rules = [{
                "rule": "unusual_login_time",
                "score": base_score,
                "description": _rule_description("unusual_login_time"),
            }]
            return {
                "assessment_status": "UNUSUAL_LOGIN_TIME",
                "risk_score": round(total_score, 2),
                "risk_level": "LOW",
                "priority": "P3",
                "scoring_reason": f"Login at {hour:02d}:xx is outside normal working hours (08:00–22:00). Requires review, not conclusive of malicious behaviour.",
                "recommended_action": "Verify with the user whether the late-hours login was authorized.",
                "scoring_breakdown": breakdown,
                "triggered_rules": triggered_rules,
                "detection_source": "Live Windows Rule-Based Risk Engine",
            }
        else:
            return {
                "assessment_status": "ROUTINE_LOGON",
                "risk_score": None,
                "risk_level": None,
                "priority": None,
                "scoring_reason": "Logon during normal working hours — no anomaly detected.",
                "recommended_action": None,
                "scoring_breakdown": {},
                "triggered_rules": [],
                "detection_source": "Live Windows Rule-Based Risk Engine",
            }

    # --- LOGON_FAILURE: failed login detection ---
    if category == "LOGON_FAILURE":
        base_score = 15.0
        breakdown = {"failed_login_burst": base_score}
        triggered_rules = [{
            "rule": "failed_login_burst",
            "score": base_score,
            "description": _rule_description("failed_login_burst"),
        }]
        return {
            "assessment_status": "FAILED_LOGIN_DETECTED",
            "risk_score": round(base_score, 2),
            "risk_level": "LOW",
            "priority": "P3",
            "scoring_reason": "Failed login attempt recorded. Single instance scored LOW. Multiple correlated failures would increase risk.",
            "recommended_action": "Review for brute force patterns. Verify account lockout policy is in place.",
            "scoring_breakdown": breakdown,
            "triggered_rules": triggered_rules,
            "detection_source": "Live Windows Rule-Based Risk Engine",
        }

    # --- DEVICE_ACTIVITY ---
    if category == "DEVICE_ACTIVITY":
        base_score = 25.0
        breakdown = {"device_activity": base_score}
        triggered_rules = [{
            "rule": "device_activity",
            "score": base_score,
            "description": _rule_description("device_activity"),
        }]
        return {
            "assessment_status": "DEVICE_ACTIVITY_DETECTED",
            "risk_score": round(base_score, 2),
            "risk_level": "MEDIUM",
            "priority": "P2",
            "scoring_reason": "Removable device activity detected. Context review required.",
            "recommended_action": "Verify the device is an authorized peripheral. Check user policy compliance.",
            "scoring_breakdown": breakdown,
            "triggered_rules": triggered_rules,
            "detection_source": "Live Windows Rule-Based Risk Engine",
        }

    # Base score for a service lifecycle event.
    base_score = 15.0
    breakdown = {"base_lifecycle_signal": base_score}
    reasons = [
        f"Windows {category.replace('_', ' ').lower()} event detected."
    ]

    # Correlation component.
    if correlated_events:
        breakdown["correlated_lifecycle_modification"] = 15.0
        reasons.append(
            f"Correlated with {len(correlated_events)} related service "
            "event(s) on the same host within five minutes."
        )

    # Service executable path inspection.
    service_file = extracted_evidence.get("service_file_name", "")
    path_increment, path_reasons = evaluate_path_suspiciousness(service_file)

    if path_increment > 0:
        breakdown["suspicious_service_path"] = path_increment
        reasons.extend(path_reasons)

    # Service account inspection.
    service_account = extracted_evidence.get("service_account", "")
    standard_accounts = {
        "localsystem",
        "localservice",
        "networkservice",
    }

    if service_account and service_account.casefold() not in standard_accounts:
        breakdown["non_system_service_account"] = 15.0
        reasons.append(
            f"Service configured under user account: {service_account}."
        )

    # Automatic start can provide persistence.
    start_type = (
        extracted_evidence.get("service_start_type")
        or extracted_evidence.get("new_start_type", "")
    )

    if "auto start" in str(start_type).casefold():
        breakdown["auto_start_persistence"] = 10.0
        reasons.append(
            "Service configured for automatic persistence (auto start)."
        )

    # Clamp score to [0, 100].
    total_score = min(100.0, max(0.0, sum(breakdown.values())))

    if total_score >= HIGH_THRESHOLD:
        risk_level = "HIGH"
        priority = "P1"
        status = "HIGH_RISK_PERSISTENCE"
        action = (
            "Immediately investigate the host, verify service binary "
            "integrity, and check process execution history. Isolate the "
            "host only after validating the incident and following "
            "the organization's response procedure."
        )
    elif total_score >= MEDIUM_THRESHOLD:
        risk_level = "MEDIUM"
        priority = "P2"
        status = (
            "CORRELATED_REVIEW"
            if correlated_events
            else "REQUIRES_REVIEW"
        )
        action = (
            "Review service installation details and confirm that the "
            "software vendor and service account are authorized."
        )
    else:
        risk_level = "LOW"
        priority = "P3"
        status = "ROUTINE_MONITORING"
        action = "Continue routine security monitoring."

    # Build triggered_rules list for attribution
    triggered_rules = [
        {"rule": k, "score": v, "description": _rule_description(k)}
        for k, v in breakdown.items()
    ]

    return {
        "assessment_status": status,
        "risk_score": round(total_score, 2),
        "risk_level": risk_level,
        "priority": priority,
        "scoring_reason": " ".join(reasons),
        "recommended_action": action,
        "scoring_breakdown": breakdown,
        "triggered_rules": triggered_rules,
        "detection_source": "Live Windows Rule-Based Risk Engine",
    }


# --------------------------------------------------
# ASSESSMENT PERSISTENCE
# --------------------------------------------------

def persist_assessments(conn, assessed_results):
    """
    Save the latest assessment for every examined event.

    The live_event_assessments table must exist, with a UNIQUE constraint
    on event_id. Reassessment updates the existing row for that event.
    """

    saved_count = 0

    for item in assessed_results:
        event = item["event"]
        evidence = item["extracted_evidence"]
        matches = item["correlated_events"]
        assessment = item["assessment"]

        correlated_events = [
            {
                "event_id": match["event_id"],
                "event_code": str(match["event_code"]),
                "event_timestamp": (
                    match["event_timestamp"].isoformat()
                    if hasattr(match["event_timestamp"], "isoformat")
                    else str(match["event_timestamp"])
                ),
            }
            for match in matches
        ]

        conn.execute(
            """
            INSERT INTO live_event_assessments (
                event_id,
                assessment_status,
                risk_score,
                risk_level,
                priority,
                scoring_reason,
                recommended_action,
                scoring_breakdown,
                extracted_evidence,
                correlated_events,
                scoring_mode,
                assessed_at
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, NOW()
            )
            ON CONFLICT (event_id)
            DO UPDATE SET
                assessment_status = EXCLUDED.assessment_status,
                risk_score = EXCLUDED.risk_score,
                risk_level = EXCLUDED.risk_level,
                priority = EXCLUDED.priority,
                scoring_reason = EXCLUDED.scoring_reason,
                recommended_action = EXCLUDED.recommended_action,
                scoring_breakdown = EXCLUDED.scoring_breakdown,
                extracted_evidence = EXCLUDED.extracted_evidence,
                correlated_events = EXCLUDED.correlated_events,
                scoring_mode = EXCLUDED.scoring_mode,
                assessed_at = NOW()
            """,
            (
                event["event_id"],
                assessment["assessment_status"],
                assessment.get("risk_score"),
                assessment.get("risk_level"),
                assessment.get("priority"),
                assessment.get("scoring_reason"),
                assessment.get("recommended_action"),
                Jsonb(assessment.get("scoring_breakdown", {})),
                Jsonb(evidence),
                Jsonb(correlated_events),
                "LIVE_WINDOWS_SERVICE_RULE",
            ),
        )

        saved_count += 1

    conn.commit()
    return saved_count


# --------------------------------------------------
# ALERT PERSISTENCE
# --------------------------------------------------

def persist_alerts(conn, assessed_results):
    """
    Persist MEDIUM and HIGH assessments as security alerts.

    Existing alerts for the same event_id are not duplicated.
    """

    inserted_count = 0
    duplicate_count = 0

    for item in assessed_results:
        event = item["event"]
        evidence = item["extracted_evidence"]
        matches = item["correlated_events"]
        assessment = item["assessment"]

        if assessment.get("risk_level") not in ("HIGH", "MEDIUM"):
            continue

        event_id = event["event_id"]

        existing = conn.execute(
            "SELECT alert_id FROM security_alerts WHERE event_id = %s",
            (event_id,),
        ).fetchone()

        if existing:
            duplicate_count += 1
            continue

        agent_evidence = {
            "source_event_code": str(event["event_code"]),
            "event_category": event["event_category"],
            "signal_status": event["signal_status"],
            "extracted_evidence": evidence,
            "correlated_events": [
                {
                    "event_id": match["event_id"],
                    "event_code": match["event_code"],
                    "event_timestamp": (
                        match["event_timestamp"].isoformat()
                        if hasattr(match["event_timestamp"], "isoformat")
                        else str(match["event_timestamp"])
                    ),
                }
                for match in matches
            ],
            "scoring_breakdown": assessment.get(
                "scoring_breakdown", {}
            ),
            "triggered_rules": assessment.get(
                "triggered_rules", []
            ),
            "detection_source": assessment.get(
                "detection_source", "Live Windows Rule-Based Risk Engine"
            ),
            "scoring_mode": "LIVE_WINDOWS_SERVICE_RULE",
            "multimodal_cert_agents": (
                "UNAVAILABLE_FOR_WINDOWS_SYSTEM_LOG"
            ),
        }

        conn.execute(
            """
            INSERT INTO security_alerts (
                event_id,
                user_name,
                detected_at,
                risk_score,
                risk_level,
                priority,
                explanation,
                recommended_action,
                agent_evidence,
                status
            )
            VALUES (%s, %s, NOW(), %s, %s, %s, %s, %s, %s, 'OPEN')
            """,
            (
                event_id,
                event.get("user_name") or "SYSTEM",
                assessment["risk_score"],
                assessment["risk_level"],
                assessment["priority"],
                assessment["scoring_reason"],
                assessment["recommended_action"],
                Jsonb(agent_evidence),
            ),
        )

        inserted_count += 1

    conn.commit()
    return inserted_count, duplicate_count


# --------------------------------------------------
# MAIN ASSESSMENT RUNNER
# --------------------------------------------------

def run_assessment(persist=False, limit=BATCH_SIZE):
    """
    Run live Windows event assessment.

    persist=False:
        Read-only inspection; no assessment or alert rows are written.

    persist=True:
        Save every assessment and create MEDIUM/HIGH alerts.
    """

    with psycopg.connect(
        **DB_CONFIG,
        row_factory=dict_row,
    ) as conn:
        events = fetch_processed_events(conn, limit=limit)
        results = correlate_events(events)

        scored_results = []

        high_count = 0
        med_count = 0
        low_count = 0
        not_assessed_count = 0
        correlation_count = 0

        for item in results:
            event = item["event"]
            matches = item["correlated_events"]
            evidence = item["extracted_evidence"]

            assessment = score_event(event, matches, evidence)
            item["assessment"] = assessment
            scored_results.append(item)

            level = assessment.get("risk_level")

            if level == "HIGH":
                high_count += 1
            elif level == "MEDIUM":
                med_count += 1
            elif level == "LOW":
                low_count += 1
            else:
                not_assessed_count += 1

            if matches:
                correlation_count += 1

        persisted = 0
        duplicates = 0
        assessments_saved = 0

        if persist:
            assessments_saved = persist_assessments(
                conn,
                scored_results,
            )

            persisted, duplicates = persist_alerts(
                conn,
                scored_results,
            )

    return {
        "events_examined": len(events),
        "high_risk_count": high_count,
        "medium_risk_count": med_count,
        "low_risk_count": low_count,
        "not_assessed_count": not_assessed_count,
        "events_with_correlation": correlation_count,
        "assessments_saved": assessments_saved,
        "alerts_persisted": persisted,
        "duplicates_skipped": duplicates,
        "scored_results": scored_results,
    }


# --------------------------------------------------
# COMMAND-LINE RUNNER
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Live Risk Engine for Windows System Events"
    )

    parser.add_argument(
        "--persist",
        action="store_true",
        help="Persist assessments and generated alerts to PostgreSQL",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=BATCH_SIZE,
        help="Number of processed events to inspect",
    )

    args = parser.parse_args()

    mode_label = (
        "PERSISTENCE ENABLED"
        if args.persist
        else "READ-ONLY INSPECTION"
    )

    print("=" * 60)
    print(f"LIVE RISK ASSESSMENT - {mode_label}")
    print("=" * 60)

    summary = run_assessment(
        persist=args.persist,
        limit=args.limit,
    )

    for item in summary["scored_results"][:15]:
        event = item["event"]
        assessment = item["assessment"]

        print(
            f"\nEvent {event['event_id']} | "
            f"Code {event['event_code']} | "
            f"{event['event_category']} | "
            f"Level: {assessment['risk_level']} | "
            f"Score: {assessment['risk_score']}"
        )

        print(f"Reason: {assessment['scoring_reason']}")

        if assessment.get("scoring_breakdown"):
            print(
                f"Breakdown: {assessment['scoring_breakdown']}"
            )

    print("\n" + "=" * 60)
    print("ASSESSMENT SUMMARY")
    print("=" * 60)
    print(f"Events examined: {summary['events_examined']}")
    print(f"HIGH risk events: {summary['high_risk_count']}")
    print(f"MEDIUM risk events: {summary['medium_risk_count']}")
    print(f"LOW risk events: {summary['low_risk_count']}")
    print(f"Not assessed: {summary['not_assessed_count']}")
    print(
        "Events with correlated evidence: "
        f"{summary['events_with_correlation']}"
    )
    print(f"Event assessments saved: {summary['assessments_saved']}")
    print(
        "Numeric risk scores assigned: "
        f"{summary['high_risk_count'] + summary['medium_risk_count'] + summary['low_risk_count']}"
    )
    print(f"Security alerts created: {summary['alerts_persisted']}")
    print(f"Duplicate alerts skipped: {summary['duplicates_skipped']}")


if __name__ == "__main__":
    main()
