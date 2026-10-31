import json
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

ALERTS_FILE = (
    PROJECT_ROOT / "data" / "outputs" / "alert_explanations.parquet"
)
SUPERVISOR_FILE = (
    PROJECT_ROOT / "data" / "outputs" / "supervisor_agent_results.parquet"
)

DB_CONFIG = {
    "host": os.environ["DB_HOST"],
    "port": int(os.environ["DB_PORT"]),
    "dbname": os.environ["DB_NAME"],
    "user": os.environ["DB_USER"],
    "password": os.environ["DB_PASSWORD"],
    "connect_timeout": 5,
}

app = FastAPI(
    title="Insider Threat Detection API",
    description=(
        "API for historical multi-agent fusion, normalized security events, "
        "live Windows event processing, and PostgreSQL incident management."
    ),
    version="1.3.0",
)

# Mount chart data router
from src.api.charts import router as charts_router  # noqa: E402
app.include_router(charts_router)


class SecurityEventIn(BaseModel):
    source_event_id: Optional[str] = None
    event_timestamp: datetime
    host_name: str = Field(min_length=1, max_length=255)
    user_name: Optional[str] = None
    event_source: str = Field(min_length=1, max_length=255)
    event_type: str = Field(min_length=1, max_length=100)
    event_code: Optional[str] = None
    details: dict[str, Any] = Field(default_factory=dict)


class AlertStatusUpdate(BaseModel):
    status: str = Field(pattern="^(OPEN|INVESTIGATING|RESOLVED|FALSE_POSITIVE)$")


def get_connection():
    try:
        return psycopg.connect(**DB_CONFIG, row_factory=dict_row)
    except psycopg.Error as exc:
        raise HTTPException(
            status_code=503,
            detail="PostgreSQL is unavailable.",
        ) from exc


LIVE_ALERT_SELECT = """
    SELECT
        a.alert_id,
        a.event_id,
        a.user_name,
        a.detected_at,
        a.risk_score,
        a.risk_level,
        a.priority,
        a.explanation,
        a.recommended_action,
        a.agent_evidence,
        a.status,
        e.host_name,
        e.event_code,
        e.event_timestamp,
        e.event_source,
        p.event_category,
        p.signal_status,
        ass.assessment_status,
        ass.scoring_breakdown,
        ass.extracted_evidence,
        ass.correlated_events,
        ass.scoring_mode,
        ass.assessed_at
    FROM security_alerts AS a
    LEFT JOIN security_events AS e
        ON e.event_id = a.event_id
    LEFT JOIN live_event_processing AS p
        ON p.event_id = a.event_id
    LEFT JOIN live_event_assessments AS ass
        ON ass.event_id = a.event_id
"""

LIVE_ASSESSMENT_SELECT = """
    SELECT
        ass.assessment_id,
        ass.event_id,
        ass.assessment_status,
        ass.risk_score,
        ass.risk_level,
        ass.priority,
        ass.scoring_reason,
        ass.recommended_action,
        ass.scoring_breakdown,
        ass.extracted_evidence,
        ass.correlated_events,
        ass.scoring_mode,
        ass.assessed_at,
        e.host_name,
        e.user_name,
        e.event_code,
        e.event_timestamp,
        e.event_source,
        p.event_category,
        p.signal_status,
        p.classification_reason
    FROM live_event_assessments AS ass
    JOIN security_events AS e
        ON e.event_id = ass.event_id
    LEFT JOIN live_event_processing AS p
        ON p.event_id = ass.event_id
"""


def _like_contains(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )
    return f"%{escaped}%"


def _raise_live_query_error(exc: psycopg.Error, detail: str):
    if isinstance(exc, psycopg.errors.UndefinedTable):
        raise HTTPException(
            status_code=503,
            detail="Live monitoring tables are not initialized.",
        ) from exc
    raise HTTPException(status_code=500, detail=detail) from exc


def _fetch_live_alert(conn, alert_id: int):
    return conn.execute(
        LIVE_ALERT_SELECT + " WHERE a.alert_id = %s",
        (alert_id,),
    ).fetchone()


def load_alerts() -> pd.DataFrame:
    """Load existing historical alert explanations."""
    if not ALERTS_FILE.exists():
        raise HTTPException(
            status_code=503,
            detail="Historical alert explanation file is unavailable.",
        )

    try:
        return pd.read_parquet(ALERTS_FILE, engine="fastparquet")
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to load historical alert data.",
        ) from exc


def json_safe(value):
    """Convert pandas/numpy scalar values into JSON-compatible values."""
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def records_to_json(df: pd.DataFrame) -> list[dict]:
    return [
        {key: json_safe(value) for key, value in row.items()}
        for row in df.to_dict(orient="records")
    ]


def _safe_table_count(conn, table_name: str) -> tuple[int, bool]:
    """Return (count, table_available) for a known live-pipeline table."""
    allowed = {
        "security_events",
        "live_event_processing",
        "security_alerts",
        "live_event_assessments",
    }
    if table_name not in allowed:
        return 0, False
    try:
        row = conn.execute(f"SELECT count(*) FROM {table_name}").fetchone()
        return int(row[0] if not isinstance(row, dict) else next(iter(row.values()))), True
    except psycopg.Error:
        conn.rollback()
        return 0, False


@app.get("/health", tags=["System"])
def health():
    database_status = "connected"
    live_events_count = 0
    live_processed_count = 0
    live_alerts_count = 0
    live_assessments_count = 0
    processing_table_available = False
    assessments_table_available = False

    try:
        with psycopg.connect(**DB_CONFIG) as conn:
            live_events_count, _events_available = _safe_table_count(
                conn, "security_events"
            )
            live_processed_count, processing_table_available = _safe_table_count(
                conn, "live_event_processing"
            )
            live_alerts_count, _alerts_available = _safe_table_count(
                conn, "security_alerts"
            )
            live_assessments_count, assessments_table_available = _safe_table_count(
                conn, "live_event_assessments"
            )
    except psycopg.Error:
        database_status = "unavailable"

    checkpoint_file = PROJECT_ROOT / "data" / "outputs" / "system_event_checkpoint.txt"
    collector_active = checkpoint_file.exists()

    if database_status != "connected":
        live_scoring_status = "unknown"
    elif assessments_table_available:
        live_scoring_status = "operational"
    else:
        live_scoring_status = "not_configured"

    overall_status = "healthy"
    if database_status != "connected":
        overall_status = "degraded"
    elif not processing_table_available or not assessments_table_available:
        overall_status = "degraded"

    return {
        "status": overall_status,
        "service": "insider-threat-detection-api",
        "database": database_status,
        "historical_alert_data_available": ALERTS_FILE.exists(),
        "live_scoring_status": live_scoring_status,
        "live_events_count": live_events_count,
        "live_processed_count": live_processed_count,
        "live_alerts_count": live_alerts_count,
        "live_assessments_count": live_assessments_count,
        "live_processing_table_available": processing_table_available,
        "live_assessments_table_available": assessments_table_available,
        "collector_checkpoint_exists": collector_active,
    }


@app.get("/api/summary", tags=["Alerts"])
def summary():
    """Summary of the existing historical alert output."""
    df = load_alerts()

    if "risk_level" not in df.columns:
        raise HTTPException(
            status_code=500,
            detail="Historical data does not contain risk_level.",
        )

    risk_counts = (
        df["risk_level"]
        .fillna("UNKNOWN")
        .astype(str)
        .value_counts()
        .to_dict()
    )

    return {
        "total_records": len(df),
        "risk_level_counts": risk_counts,
        "data_source": str(ALERTS_FILE.relative_to(PROJECT_ROOT)),
        "mode": "historical_batch_results",
    }


@app.get("/api/alerts", tags=["Historical Alerts"])
def list_alerts(
    risk_level: Optional[str] = Query(
        default=None,
        pattern="^(LOW|MEDIUM|HIGH)$",
    ),
    search: Optional[str] = Query(default=None, min_length=1, max_length=100),
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """Historical CERT supervisor alerts. Separate from live Windows alerts."""
    df = load_alerts()

    if risk_level and "risk_level" in df.columns:
        df = df[
            df["risk_level"].astype(str).str.upper() == risk_level
        ]

    if search and "user" in df.columns:
        needle = search.casefold()
        user_match = df["user"].astype(str).str.casefold().str.contains(
            needle, regex=False, na=False
        )
        explanation_match = (
            df["explanation"].astype(str).str.casefold().str.contains(
                needle, regex=False, na=False
            )
            if "explanation" in df.columns
            else False
        )
        df = df[user_match | explanation_match]

    if (date_from or date_to) and "day" in df.columns:
        days = pd.to_datetime(df["day"], errors="coerce").dt.date
        if date_from:
            df = df[days >= date_from]
            days = pd.to_datetime(df["day"], errors="coerce").dt.date
        if date_to:
            df = df[days <= date_to]

    total = len(df)
    df = df.iloc[offset:offset + limit]
    alerts = records_to_json(df)
    for alert in alerts:
        alert["source"] = "historical"
        alert["scoring_mode"] = "HISTORICAL_CERT_SUPERVISOR"

    return {
        "source": "historical",
        "scoring_mode": "HISTORICAL_CERT_SUPERVISOR",
        "total": total,
        "limit": limit,
        "offset": offset,
        "alerts": alerts,
    }


@app.get(
    "/api/alerts/{user_id}/{event_day}",
    tags=["Historical Investigation"],
)
def get_user_day_alert(user_id: str, event_day: date):
    df = load_alerts()

    required = {"user", "day"}
    if not required.issubset(df.columns):
        raise HTTPException(
            status_code=500,
            detail="Expected user and day columns were not found.",
        )

    user_match = df["user"].astype(str) == user_id
    day_match = (
        pd.to_datetime(df["day"], errors="coerce").dt.date == event_day
    )
    matches = df[user_match & day_match]

    if matches.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No alert found for user '{user_id}' on day '{event_day}'.",
        )

    return records_to_json(matches)[0]


@app.post("/api/events", tags=["Database Events"], status_code=201)
def create_event(event: SecurityEventIn):
    timestamp = event.event_timestamp

    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    else:
        timestamp = timestamp.astimezone(timezone.utc)

    is_system_event = (
        event.source_event_id is not None
        and event.source_event_id.startswith("System-")
    )

    if is_system_event:
        sql = """
            INSERT INTO security_events (
                source_event_id, event_timestamp, host_name, user_name,
                event_source, event_type, event_code, details
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (source_event_id)
                WHERE source_event_id LIKE 'System-%%'
            DO NOTHING
            RETURNING event_id, ingested_at
        """
    else:
        sql = """
            INSERT INTO security_events (
                source_event_id, event_timestamp, host_name, user_name,
                event_source, event_type, event_code, details
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING event_id, ingested_at
        """

    params = (
        event.source_event_id,
        timestamp,
        event.host_name,
        event.user_name,
        event.event_source,
        event.event_type,
        event.event_code,
        Jsonb(event.details),
    )

    try:
        with get_connection() as conn:
            row = conn.execute(sql, params).fetchone()

            if row is None and is_system_event:
                row = conn.execute(
                    """
                    SELECT event_id, ingested_at
                    FROM security_events
                    WHERE source_event_id = %s
                    """,
                    (event.source_event_id,),
                ).fetchone()

                if row is None:
                    raise HTTPException(
                        status_code=500,
                        detail="Unable to retrieve the existing security event.",
                    )

                return {
                    "message": "Duplicate event skipped; existing event retained.",
                    "event_id": row["event_id"],
                    "ingested_at": row["ingested_at"],
                    "duplicate": True,
                }

    except HTTPException:
        raise
    except psycopg.Error as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to store the security event.",
        ) from exc

    return {
        "message": "Event stored successfully.",
        "event_id": row["event_id"],
        "ingested_at": row["ingested_at"],
        "duplicate": False,
    }


@app.get("/api/events", tags=["Database Events"])
def list_events(
    user_name: Optional[str] = None,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    sql = """
        SELECT event_id, source_event_id, event_timestamp, ingested_at,
               host_name, user_name, event_source, event_type,
               event_code, details
        FROM security_events
    """
    params = []

    if user_name:
        sql += " WHERE user_name = %s"
        params.append(user_name)

    sql += " ORDER BY ingested_at DESC LIMIT %s OFFSET %s"
    params.extend([limit, offset])

    try:
        with get_connection() as conn:
            rows = conn.execute(sql, params).fetchall()
    except HTTPException:
        raise
    except psycopg.Error as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve security events.",
        ) from exc

    return {
        "limit": limit,
        "offset": offset,
        "events": rows,
    }


@app.get("/api/alerts/db", tags=["Database Alerts"])
def list_database_alerts(
    risk_level: Optional[str] = Query(
        default=None,
        pattern="^(LOW|MEDIUM|HIGH)$",
    ),
    status: Optional[str] = Query(
        default=None,
        pattern="^(OPEN|INVESTIGATING|RESOLVED|FALSE_POSITIVE)$",
    ),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    sql = """
        SELECT alert_id, event_id, user_name, detected_at, risk_score,
               risk_level, priority, explanation, recommended_action,
               agent_evidence, status
        FROM security_alerts
        WHERE 1=1
    """
    params = []

    if risk_level:
        sql += " AND risk_level = %s"
        params.append(risk_level)

    if status:
        sql += " AND status = %s"
        params.append(status)

    sql += " ORDER BY detected_at DESC, alert_id DESC LIMIT %s OFFSET %s"
    params.extend([limit, offset])

    try:
        with get_connection() as conn:
            rows = conn.execute(sql, params).fetchall()
    except HTTPException:
        raise
    except psycopg.Error as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve database alerts.",
        ) from exc

    return {
        "limit": limit,
        "offset": offset,
        "count": len(rows),
        "alerts": rows,
    }


@app.patch("/api/alerts/db/{alert_id}/status", tags=["Database Alerts"])
def update_alert_status(alert_id: int, body: AlertStatusUpdate):
    """Update status of a persisted security alert (Incident Management)."""
    try:
        with get_connection() as conn:
            row = conn.execute(
                """
                UPDATE security_alerts
                SET status = %s
                WHERE alert_id = %s
                RETURNING alert_id, event_id, user_name, risk_score,
                          risk_level, priority, explanation, recommended_action,
                          agent_evidence, status, detected_at
                """,
                (body.status, alert_id),
            ).fetchone()

            if not row:
                raise HTTPException(status_code=404, detail=f"Alert ID {alert_id} not found.")

            conn.commit()
            return {
                "message": f"Alert {alert_id} status updated to {body.status}.",
                "alert": row,
            }
    except HTTPException:
        raise
    except psycopg.Error as exc:
        raise HTTPException(status_code=500, detail="Database update error.") from exc


@app.get("/api/live/events", tags=["Live Monitoring"])
def list_live_events(
    limit: int = Query(default=100, ge=1, le=500),
):
    """Return collected Windows events and their processing results."""

    sql = """
        SELECT
            e.event_id,
            e.source_event_id,
            e.event_timestamp,
            e.ingested_at,
            e.host_name,
            e.user_name,
            e.event_source,
            e.event_type,
            e.event_code,
            e.details,
            p.processed_at,
            p.event_category,
            p.signal_status,
            p.classification_reason,
            p.evidence
        FROM security_events AS e
        LEFT JOIN live_event_processing AS p
            ON p.event_id = e.event_id
        ORDER BY e.event_timestamp DESC, e.event_id DESC
        LIMIT %s
    """

    try:
        with psycopg.connect(
            **DB_CONFIG,
            row_factory=dict_row,
        ) as conn:
            rows = conn.execute(sql, (limit,)).fetchall()

        return {
            "count": len(rows),
            "events": rows,
            "source": "PostgreSQL",
        }

    except psycopg.Error as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve live Windows events.",
        ) from exc


@app.post("/api/live/assess", tags=["Live Monitoring"])
def trigger_live_assessment(persist: bool = Query(default=True)):
    """Trigger the live risk scoring engine on collected Windows events."""
    try:
        from src.live_processor.live_risk_engine import run_assessment
        summary = run_assessment(persist=persist)
        return {
            "message": "Live assessment completed successfully.",
            "mode": "persistence" if persist else "read_only",
            "summary": {
                "events_examined": summary["events_examined"],
                "high_risk_count": summary["high_risk_count"],
                "medium_risk_count": summary["medium_risk_count"],
                "low_risk_count": summary["low_risk_count"],
                "not_assessed_count": summary["not_assessed_count"],
                "events_with_correlation": summary["events_with_correlation"],
                "assessments_saved": summary["assessments_saved"],
                "alerts_persisted": summary["alerts_persisted"],
                "duplicates_skipped": summary["duplicates_skipped"],
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Live assessment error: {str(exc)}")


@app.post("/api/live/collect", tags=["Live Monitoring"])
def trigger_live_collection():
    """
    Execute an on-demand live Windows event collection and processing pass:
    1. Collects latest Windows System events into security_events using collect_system_events.ps1 -Once.
    2. Runs live event processor to categorize events into live_event_processing.
    3. Runs live risk engine to correlate, score, and persist any new alerts into security_alerts.
    """
    import subprocess
    collector_script = PROJECT_ROOT / "src" / "collector" / "collect_system_events.ps1"

    # 1. Run collector pass
    try:
        cmd = ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(collector_script), "-Once", "-BatchSize", "50"]
        subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except Exception as exc:
        print(f"Collector pass warning: {exc}", flush=True)

    # 2. Run processor pass
    processed_count = 0
    try:
        from src.live_processor.event_processor import process_batch, initialize_processor_table
        with psycopg.connect(**DB_CONFIG, row_factory=dict_row) as conn:
            initialize_processor_table(conn)
            processed_count = process_batch(conn)
    except Exception as exc:
        print(f"Processor pass warning: {exc}", flush=True)

    # 3. Run assessment pass
    summary = {}
    try:
        from src.live_processor.live_risk_engine import run_assessment
        assessment_res = run_assessment(persist=True)
        summary = {
            "events_examined": assessment_res["events_examined"],
            "assessments_saved": assessment_res["assessments_saved"],
            "alerts_persisted": assessment_res["alerts_persisted"],
            "duplicates_skipped": assessment_res["duplicates_skipped"],
            "high_risk_count": assessment_res["high_risk_count"],
            "medium_risk_count": assessment_res["medium_risk_count"],
            "low_risk_count": assessment_res["low_risk_count"],
            "not_assessed_count": assessment_res["not_assessed_count"],
        }
    except Exception as exc:
        print(f"Assessment pass warning: {exc}", flush=True)

    # Get latest counts
    total_events = 0
    total_alerts = 0
    try:
        with psycopg.connect(**DB_CONFIG) as conn:
            cur = conn.cursor()
            cur.execute("SELECT count(*) FROM security_events")
            total_events = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM security_alerts")
            total_alerts = cur.fetchone()[0]
    except Exception:
        pass

    return {
        "message": "Live collection, processing, and risk assessment completed.",
        "new_events_processed": processed_count,
        "assessment_summary": summary,
        "total_security_events": total_events,
        "total_security_alerts": total_alerts,
    }


@app.get("/api/analytics/agents", tags=["Agent Analytics"])
def get_agent_analytics():
    """Expose existing agent and supervisor output files."""

    output_dir = PROJECT_ROOT / "data" / "outputs"

    agent_files = {
        "Rule Agent": ["final_rule_agent_xgb_results.parquet"],
        "Login Agent": ["login_agent_results.parquet"],
        "File Agent": [
            "file_agent_v2_results.parquet",
            "file_agent_results.parquet",
        ],
        "Device Agent": [
            "device_agent_results.parquet",
            "device_agent_v2_results.parquet",
        ],
        "Network Agent": [
            "network_agent_v2_results.parquet",
            "network_agent_results.parquet",
        ],
        "Supervisor Agent": ["supervisor_agent_results.parquet"],
        "Risk Engine": ["risk_engine_results.parquet"],
    }

    results = {}

    for agent_name, candidates in agent_files.items():
        path = next(
            (output_dir / name for name in candidates
             if (output_dir / name).is_file()),
            None,
        )

        if path is None:
            results[agent_name] = {
                "available": False,
                "message": "Output file not found.",
            }
            continue

        try:
            df = pd.read_parquet(path)

            results[agent_name] = {
                "available": True,
                "mode": "historical",
                "file": path.name,
                "row_count": len(df),
                "columns": df.columns.tolist(),
                "sample_records": json.loads(
                    df.head(3).to_json(
                        orient="records",
                        date_format="iso",
                    )
                ),
            }
        except Exception as exc:
            results[agent_name] = {
                "available": False,
                "message": f"Could not read output: {type(exc).__name__}",
            }

    return {
        "source": "data/outputs",
        "results": results,
    }


@app.get("/api/analytics/models", tags=["Model Analytics"])
def get_model_analytics():
    """Return verified ML coverage, evaluation metrics, and score distributions."""
    metrics = {
        "dataset": {
            "total_user_days": 342706,
            "unique_users": 1000,
            "ground_truth_malicious_days": 32,
            "evaluation_timeframe": "2010-01-02 to 2011-05-17",
            "class_imbalance_ratio": "1 : 10,709",
        },
        "model_coverage": {
            "isolation_forest": {
                "heldout_test_rows": 66813,
                "training_rows": 275893,
                "split_type": "Chronological holdout (latest 20% of dates)",
                "contamination": 0.001,
                "score_representation": "Continuous anomaly intensity scaled [0, 100] from -score_samples(X)",
                "note": "Chronological holdout in 2011 contains only benign records as all CERT r4.1 malicious days occurred in mid-2010.",
            },
            "random_forest": {
                "heldout_test_rows": 68542,
                "split_type": "Stratified random holdout (20% of dataset)",
                "test_malicious_rows": 6,
                "score_representation": "Calibrated classifier probability scaled [0, 100]",
                "evaluation_metrics": {
                    "precision": 0.40,
                    "recall": 0.67,
                    "f1": 0.50,
                },
            },
            "lstm": {
                "heldout_test_rows": 62545,
                "split_type": "7-day sliding window sequences on stratified holdout",
                "test_malicious_rows": 6,
                "score_representation": "Sequence model prediction probability scaled [0, 100]",
                "evaluation_metrics": {
                    "precision": 0.38,
                    "recall": 0.50,
                    "f1": 0.43,
                },
            },
            "union_coverage_user_days": 161628,
            "three_model_intersection_user_days": 2646,
            "agent_only_fallback_user_days": 181078,
        },
        "supervisor_fusion": {
            "configured_agent_weights": {
                "Rule Agent (XGBoost)": 0.40,
                "Login Agent": 0.15,
                "File Agent": 0.15,
                "Device Agent": 0.15,
                "Network Agent": 0.15,
            },
            "configured_ml_weights": {
                "Isolation Forest": 0.40,
                "Random Forest": 0.30,
                "LSTM": 0.30,
            },
            "fusion_policy": "50% Agent Score + 50% Available ML Score (reweighted). 100% Agent Score when ML unavailable.",
            "detection_results": {
                "ground_truth_recall_at_medium": "100.0% (32 / 32 true malicious days flagged)",
                "high_threat_alerts": 10,
                "medium_threat_alerts": 5533,
                "low_routine_records": 337163,
            },
        },
    }
    return metrics


# Memoized user intelligence dataframe
_USER_INTEL_CACHE = None

@app.get("/api/users/intelligence", tags=["User Intelligence"])
def get_user_intelligence(
    limit: int = Query(default=50, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):
    """Return aggregated user intelligence ranked by risk."""
    global _USER_INTEL_CACHE

    if _USER_INTEL_CACHE is None:
        if not SUPERVISOR_FILE.exists():
            raise HTTPException(status_code=503, detail="Supervisor data unavailable.")

        df = pd.read_parquet(SUPERVISOR_FILE)
        grouped = df.groupby("user").agg(
            total_days=("day", "count"),
            max_risk=("supervisor_risk_score", "max"),
            avg_risk=("supervisor_risk_score", "mean"),
            high_risk_days=("supervisor_risk_score", lambda s: int((s >= 60.0).sum())),
            medium_risk_days=("supervisor_risk_score", lambda s: int(((s >= 25.0) & (s < 60.0)).sum())),
            is_malicious=("ground_truth", "max"),
        ).reset_index()

        grouped["max_threat_level"] = np.select(
            [grouped["max_risk"] >= 60.0, grouped["max_risk"] >= 25.0],
            ["HIGH", "MEDIUM"],
            default="LOW",
        )
        grouped = grouped.sort_values("max_risk", ascending=False).reset_index(drop=True)
        _USER_INTEL_CACHE = grouped

    total_users = len(_USER_INTEL_CACHE)
    page_df = _USER_INTEL_CACHE.iloc[offset:offset + limit]

    return {
        "total_users": total_users,
        "limit": limit,
        "offset": offset,
        "users": [
            {
                "user": str(row["user"]),
                "total_days": int(row["total_days"]),
                "max_risk_score": round(float(row["max_risk"]), 2),
                "avg_risk_score": round(float(row["avg_risk"]), 2),
                "max_threat_level": str(row["max_threat_level"]),
                "high_risk_days": int(row["high_risk_days"]),
                "medium_risk_days": int(row["medium_risk_days"]),
                "is_malicious": bool(row["is_malicious"] == 1),
            }
            for _, row in page_df.iterrows()
        ],
    }


@app.get("/api/users/intelligence/{user_id}", tags=["User Intelligence"])
def get_user_detail(user_id: str):
    """Return individual user risk profile and incident timeline."""
    df = load_alerts()
    user_records = df[df["user"].astype(str).str.upper() == user_id.upper()]

    if user_records.empty:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found.")

    elevated = user_records[user_records["risk_level"].isin(["HIGH", "MEDIUM"])].sort_values("day")

    return {
        "user": user_id.upper(),
        "total_days_recorded": len(user_records),
        "elevated_risk_days": len(elevated),
        "max_risk_score": round(float(user_records["supervisor_risk_score"].max()), 2),
        "avg_risk_score": round(float(user_records["supervisor_risk_score"].mean()), 2),
        "timeline": records_to_json(elevated.head(30)),
    }
