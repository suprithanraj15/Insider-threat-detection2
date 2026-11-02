
from datetime import datetime, timezone, timedelta
from pathlib import Path
import os
import traceback

import psycopg
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from src.live_processor.event_processor import classify_event
from src.live_processor.live_risk_engine import (
    correlate_events,
    score_event,
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "127.0.0.1"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.getenv("DB_NAME", ""),
    "user": os.getenv("DB_USER", ""),
    "password": os.getenv("DB_PASSWORD", ""),
    "connect_timeout": 5,
}

router = APIRouter(
    prefix="/api/simulation",
    tags=["Attack Simulation"],
)


# --------------------------------------------------
# REQUEST MODEL
# --------------------------------------------------

class SimulationRequest(BaseModel):
    scenario_key: str = Field(
        ...,
        description="One of the supported simulation scenario keys.",
    )
    host_name: str = "SIMULATED-WINDOWS-HOST"
    user_name: str = "simulated.user"


# --------------------------------------------------
# SYNTHETIC SCENARIO DEFINITIONS
# --------------------------------------------------

def build_scenario_events(scenario_key, host_name, user_name):
    """
    Build synthetic Windows event payloads.

    These events are never inserted into security_events.
    """

    now = datetime.now(timezone.utc)

    scenarios = {

        "suspicious_service_install": {
            "name": "Suspicious Service Installation",
            "expected_level": "HIGH",
            "events": [
                {
                    "timestamp": now,
                    "source": "Service Control Manager",
                    "code": "7045",
                    "message": (
                        "Service Name: SimulatedUpdateService\n"
                        "Service File Name: "
                        "C:\\Users\\Public\\powershell.exe "
                        "-NoProfile\n"
                        "Service Type: user mode service\n"
                        "Service Start Type: auto start\n"
                        "Service Account: .\\simulated.user"
                    ),
                }
            ],
        },

        "correlated_service_persistence": {
            "name": "Correlated Service Persistence",
            "expected_level": "MEDIUM",
            "events": [
                {
                    "timestamp": now,
                    "source": "Service Control Manager",
                    "code": "7045",
                    "message": (
                        "Service Name: SimulatedAgentService\n"
                        "Service File Name: "
                        "C:\\Program Files\\Example\\agent.exe\n"
                        "Service Type: user mode service\n"
                        "Service Start Type: auto start\n"
                        "Service Account: LocalSystem"
                    ),
                },
                {
                    "timestamp": now + timedelta(seconds=10),
                    "source": "Service Control Manager",
                    "code": "7040",
                    "message": (
                        "The start type of the "
                        "SimulatedAgentService service was changed "
                        "from demand start to auto start."
                    ),
                },
            ],
        },

        "unusual_hours_login": {
            "name": "Unusual Hours Login",
            "expected_level": "LOW",
            "events": [
                {
                    "timestamp": now.replace(
                        hour=2,
                        minute=15,
                        second=0,
                        microsecond=0,
                    ),
                    "source": (
                        "Microsoft-Windows-Security-Auditing"
                    ),
                    "code": "4624",
                    "message": (
                        "Synthetic successful login at 02:15."
                    ),
                }
            ],
        },

        "removable_device": {
            "name": "Removable Device Recognition",
            "expected_level": "MEDIUM",
            "events": [
                {
                    "timestamp": now,
                    "source": (
                        "Microsoft-Windows-Security-Auditing"
                    ),
                    "code": "6416",
                    "message": (
                        "Synthetic external device recognition event."
                    ),
                }
            ],
        },

        "failed_login": {
            "name": "Failed Login Attempt",
            "expected_level": "LOW",
            "events": [
                {
                    "timestamp": now,
                    "source": (
                        "Microsoft-Windows-Security-Auditing"
                    ),
                    "code": "4625",
                    "message": (
                        "Synthetic failed login attempt."
                    ),
                }
            ],
        },
    }

    scenario = scenarios.get(scenario_key)

    if scenario is None:
        raise ValueError(
            f"Unsupported scenario: {scenario_key}"
        )

    events = []

    for index, raw in enumerate(scenario["events"], start=1):
        events.append({
            "event_id": -index,
            "event_timestamp": raw["timestamp"],
            "host_name": host_name,
            "user_name": user_name,
            "event_source": raw["source"],
            "event_type": "Windows Event",
            "event_code": raw["code"],
            "details": {
                "message": raw["message"],
                "simulation": True,
            },
            "scenario_key": scenario_key,
            "expected_risk_level": scenario["expected_level"],
        })

    return scenario, events


# --------------------------------------------------
# CLASSIFICATION AND SCORING
# --------------------------------------------------

def assess_simulated_events(events):
    """
    Reuse the existing event classifier, correlation logic,
    and scoring function without using live persistence.
    """

    classified_events = []

    for event in events:
        classification = classify_event(event)

        event_timestamp = event["event_timestamp"]

        # Ensure timestamps are datetime objects before correlation.
        if isinstance(event_timestamp, str):
            event_timestamp = datetime.fromisoformat(
                event_timestamp.replace("Z", "+00:00")
            )

        classified_event = {
            **event,
            **classification,
            "event_timestamp": event_timestamp,
            "evidence": {
                "message": event["details"].get("message", ""),
                "simulation": True,
            },
        }

        classified_events.append(classified_event)

    correlated_results = correlate_events(classified_events)

    assessed_results = []

    for item in correlated_results:
        assessment = score_event(
            item["event"],
            item["correlated_events"],
            item["extracted_evidence"],
        )

        item["assessment"] = assessment
        assessed_results.append(item)

    return assessed_results


# --------------------------------------------------
# DATABASE PERSISTENCE
# --------------------------------------------------

def save_simulation_results(
    conn,
    simulation_id,
    scenario_key,
    events,
    assessed_results,
):
    detected_count = 0
    high_count = 0
    medium_count = 0
    low_count = 0

    for item in assessed_results:
        event = item["event"]
        assessment = item["assessment"]

        risk_level = assessment.get("risk_level")
        risk_score = assessment.get("risk_score")

        detection_status = (
            "DETECTED"
            if risk_level is not None
            else "NOT_DETECTED"
        )

        if detection_status == "DETECTED":
            detected_count += 1

        if risk_level == "HIGH":
            high_count += 1
        elif risk_level == "MEDIUM":
            medium_count += 1
        elif risk_level == "LOW":
            low_count += 1

        detection_evidence = {
            "assessment_status": assessment.get(
                "assessment_status"
            ),
            "scoring_reason": assessment.get(
                "scoring_reason"
            ),
            "scoring_breakdown": assessment.get(
                "scoring_breakdown", {}
            ),
            "triggered_rules": assessment.get(
                "triggered_rules", []
            ),
            "extracted_evidence": item[
                "extracted_evidence"
            ],
            "correlated_events": [
                {
                    "event_id": match["event_id"],
                    "event_code": str(match["event_code"]),
                }
                for match in item["correlated_events"]
            ],
            "detection_source": (
                "SIMULATION_USING_LIVE_RULE_ENGINE"
            ),
        }

        conn.execute(
            """
            INSERT INTO sim_events (
                simulation_id,
                event_timestamp,
                host_name,
                user_name,
                event_source,
                event_type,
                event_code,
                scenario_key,
                details,
                expected_risk_level,
                detected_risk_score,
                detected_risk_level,
                detection_status,
                detection_evidence
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s
            )
            """,
            (
                simulation_id,
                event["event_timestamp"],
                event["host_name"],
                event["user_name"],
                event["event_source"],
                event["event_type"],
                str(event["event_code"]),
                scenario_key,
                Jsonb(event["details"]),
                event["expected_risk_level"],
                risk_score,
                risk_level,
                detection_status,
                Jsonb(detection_evidence),
            ),
        )

    return {
        "events_generated": len(events),
        "events_detected": detected_count,
        "high_risk_count": high_count,
        "medium_risk_count": medium_count,
        "low_risk_count": low_count,
    }


# --------------------------------------------------
# RUN SIMULATION API
# --------------------------------------------------

@router.post("/run")
def run_simulation(request: SimulationRequest):

    try:
        scenario, events = build_scenario_events(
            request.scenario_key,
            request.host_name,
            request.user_name,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    conn = None
    simulation_id = None

    try:
        conn = psycopg.connect(
            **DB_CONFIG,
            row_factory=dict_row,
        )

        # Create simulation run first.
        run = conn.execute(
            """
            INSERT INTO simulation_runs (
                scenario_key,
                scenario_name,
                host_name,
                user_name,
                status,
                expected_result,
                started_at
            )
            VALUES (%s, %s, %s, %s, 'RUNNING', %s, NOW())
            RETURNING simulation_id
            """,
            (
                request.scenario_key,
                scenario["name"],
                request.host_name,
                request.user_name,
                Jsonb({
                    "expected_risk_level": (
                        scenario["expected_level"]
                    ),
                    "expected_event_count": len(events),
                }),
            ),
        ).fetchone()

        simulation_id = run["simulation_id"]
        conn.commit()

        # Run classifier, correlation and scoring.
        assessed_results = assess_simulated_events(events)

        # Save only to simulation tables.
        actual_result = save_simulation_results(
            conn,
            simulation_id,
            request.scenario_key,
            events,
            assessed_results,
        )

        expected_level = scenario["expected_level"]

        actual_levels = [
            item["assessment"].get("risk_level")
            for item in assessed_results
        ]

        expected_match = (
            len(actual_levels) == len(events)
            and all(
                level == expected_level
                for level in actual_levels
            )
        )

        actual_result["expected_risk_level"] = expected_level
        actual_result["actual_risk_levels"] = actual_levels
        actual_result["expected_result_matched"] = expected_match

        conn.execute(
            """
            UPDATE simulation_runs
            SET
                status = 'COMPLETED',
                actual_result = %s,
                completed_at = NOW()
            WHERE simulation_id = %s
            """,
            (
                Jsonb(actual_result),
                simulation_id,
            ),
        )

        conn.commit()

        return {
            "simulation_id": simulation_id,
            "scenario_key": request.scenario_key,
            "scenario_name": scenario["name"],
            "status": "COMPLETED",
            "expected_result": expected_level,
            "actual_result": actual_result,
        }

    except Exception as exc:
        if conn is not None:
            conn.rollback()

            if simulation_id is not None:
                conn.execute(
                    """
                    UPDATE simulation_runs
                    SET
                        status = 'FAILED',
                        error_message = %s,
                        completed_at = NOW()
                    WHERE simulation_id = %s
                    """,
                    (str(exc), simulation_id),
                )
                conn.commit()

        # Print full traceback in the backend terminal.
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Simulation failed: {str(exc)}",
        )

    finally:
        if conn is not None:
            conn.close()


# --------------------------------------------------
# LIST SIMULATION RUNS
# --------------------------------------------------

@router.get("/runs")
def get_simulation_runs(limit: int = 20):

    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=400,
            detail="Limit must be between 1 and 100.",
        )

    try:
        with psycopg.connect(
            **DB_CONFIG,
            row_factory=dict_row,
        ) as conn:

            rows = conn.execute(
                """
                SELECT *
                FROM simulation_runs
                ORDER BY started_at DESC
                LIMIT %s
                """,
                (limit,),
            ).fetchall()

        return {
            "count": len(rows),
            "runs": rows,
        }

    except psycopg.Error as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not retrieve simulation runs: {exc}",
        )


# --------------------------------------------------
# GET SIMULATION DETAILS
# --------------------------------------------------

@router.get("/runs/{simulation_id}")
def get_simulation_details(simulation_id: int):

    try:
        with psycopg.connect(
            **DB_CONFIG,
            row_factory=dict_row,
        ) as conn:

            run = conn.execute(
                """
                SELECT *
                FROM simulation_runs
                WHERE simulation_id = %s
                """,
                (simulation_id,),
            ).fetchone()

            if run is None:
                raise HTTPException(
                    status_code=404,
                    detail="Simulation run not found.",
                )

            events = conn.execute(
                """
                SELECT *
                FROM sim_events
                WHERE simulation_id = %s
                ORDER BY event_timestamp, sim_event_id
                """,
                (simulation_id,),
            ).fetchall()

        return {
            "simulation": run,
            "events": events,
        }

    except psycopg.Error as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not retrieve simulation: {exc}",
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not retrieve simulation: {exc}",
        )
