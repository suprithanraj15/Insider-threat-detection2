"""
Chart Data API — aggregated queries for dashboard visualizations.
All endpoints return lightweight JSON payloads suitable for pure CSS/SVG charts.
"""

import os
from datetime import datetime, timezone
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from psycopg.rows import dict_row

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

router = APIRouter(prefix="/api/charts", tags=["Charts"])


def get_connection():
    try:
        return psycopg.connect(**DB_CONFIG, row_factory=dict_row)
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail="PostgreSQL is unavailable.") from exc


@router.get("/alert-distribution")
def alert_distribution():
    """Return alert counts by risk_level and status for donut/bar charts."""
    try:
        with get_connection() as conn:
            by_level = conn.execute(
                """
                SELECT risk_level, count(*) AS count
                FROM security_alerts
                GROUP BY risk_level
                ORDER BY
                    CASE risk_level WHEN 'HIGH' THEN 1 WHEN 'MEDIUM' THEN 2 ELSE 3 END
                """
            ).fetchall()

            by_status = conn.execute(
                """
                SELECT status, count(*) AS count
                FROM security_alerts
                GROUP BY status
                ORDER BY count DESC
                """
            ).fetchall()

            total = conn.execute(
                "SELECT count(*) AS n FROM security_alerts"
            ).fetchone()["n"]

        return {
            "total": total,
            "by_level": [{"label": r["risk_level"], "count": r["count"]} for r in by_level],
            "by_status": [{"label": r["status"], "count": r["count"]} for r in by_status],
        }
    except HTTPException:
        raise
    except psycopg.Error as exc:
        raise HTTPException(status_code=500, detail="Query error.") from exc


@router.get("/event-timeline")
def event_timeline():
    """Return event counts grouped by hour over the last 48 hours."""
    try:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT
                    date_trunc('hour', event_timestamp) AS hour,
                    count(*) AS total,
                    count(*) FILTER (
                        WHERE event_code IN ('7040', '7045')
                    ) AS service_events,
                    count(*) FILTER (
                        WHERE event_code NOT IN ('7040', '7045')
                    ) AS other_events
                FROM security_events
                WHERE event_timestamp >= NOW() - INTERVAL '48 hours'
                GROUP BY hour
                ORDER BY hour ASC
                """
            ).fetchall()

            # Fill in any missing hours with 0 for a continuous timeline
            result = [
                {
                    "hour": r["hour"].isoformat() if r["hour"] else None,
                    "total": r["total"],
                    "service_events": r["service_events"],
                    "other_events": r["other_events"],
                }
                for r in rows
            ]

        return {"timeline": result, "count": len(result)}
    except HTTPException:
        raise
    except psycopg.Error as exc:
        raise HTTPException(status_code=500, detail="Query error.") from exc


@router.get("/alert-timeline")
def alert_timeline():
    """Return alert scores over time for the sparkline risk chart."""
    try:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT
                    detected_at,
                    risk_score,
                    risk_level,
                    user_name,
                    priority
                FROM security_alerts
                ORDER BY detected_at DESC
                LIMIT 100
                """
            ).fetchall()

        result = [
            {
                "detected_at": r["detected_at"].isoformat() if r["detected_at"] else None,
                "risk_score": round(float(r["risk_score"]), 1),
                "risk_level": r["risk_level"],
                "user_name": r["user_name"],
                "priority": r["priority"],
            }
            for r in reversed(rows)  # chronological order
        ]

        return {"alerts": result, "count": len(result)}
    except HTTPException:
        raise
    except psycopg.Error as exc:
        raise HTTPException(status_code=500, detail="Query error.") from exc


@router.get("/event-categories")
def event_categories():
    """Return event category breakdown from the live_event_processing table."""
    try:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT
                    COALESCE(event_category, 'UNCLASSIFIED') AS category,
                    count(*) AS count
                FROM live_event_processing
                GROUP BY event_category
                ORDER BY count DESC
                """
            ).fetchall()

            total = conn.execute(
                "SELECT count(*) AS n FROM live_event_processing"
            ).fetchone()["n"]

            # Top event codes from security_events
            codes = conn.execute(
                """
                SELECT event_code, count(*) AS count
                FROM security_events
                WHERE event_code IS NOT NULL
                GROUP BY event_code
                ORDER BY count DESC
                LIMIT 8
                """
            ).fetchall()

        return {
            "total": total,
            "categories": [{"label": r["category"], "count": r["count"]} for r in rows],
            "top_codes": [{"code": r["event_code"], "count": r["count"]} for r in codes],
        }
    except HTTPException:
        raise
    except psycopg.Error as exc:
        raise HTTPException(status_code=500, detail="Query error.") from exc


@router.get("/risk-score-distribution")
def risk_score_distribution():
    """Return histogram buckets for risk score distribution."""
    try:
        with get_connection() as conn:
            # Build buckets: 0-10, 10-20, ..., 90-100
            rows = conn.execute(
                """
                SELECT
                    floor(risk_score / 10) * 10 AS bucket_start,
                    count(*) AS count
                FROM security_alerts
                GROUP BY bucket_start
                ORDER BY bucket_start
                """
            ).fetchall()

        buckets = []
        bucket_map = {int(r["bucket_start"]): r["count"] for r in rows}
        for i in range(0, 100, 10):
            buckets.append({
                "label": f"{i}–{i+10}",
                "start": i,
                "end": i + 10,
                "count": bucket_map.get(i, 0),
            })

        return {"buckets": buckets}
    except HTTPException:
        raise
    except psycopg.Error as exc:
        raise HTTPException(status_code=500, detail="Query error.") from exc


@router.get("/summary-stats")
def summary_stats():
    """Return a comprehensive summary for the dashboard header cards."""
    try:
        with get_connection() as conn:
            stats = conn.execute(
                """
                SELECT
                    (SELECT count(*) FROM security_events) AS total_events,
                    (SELECT count(*) FROM security_alerts) AS total_alerts,
                    (SELECT count(*) FROM security_alerts WHERE status = 'OPEN') AS open_alerts,
                    (SELECT count(*) FROM security_alerts WHERE risk_level = 'HIGH') AS high_risk,
                    (SELECT count(*) FROM security_alerts WHERE risk_level = 'MEDIUM') AS medium_risk,
                    (SELECT count(*) FROM live_event_processing WHERE signal_status = 'REVIEW_SIGNAL') AS review_signals,
                    (SELECT count(*) FROM live_event_processing WHERE signal_status = 'UNCLASSIFIED') AS unclassified,
                    (SELECT count(*) FROM live_event_processing) AS processed_events,
                    (SELECT count(*) FROM live_event_assessments) AS live_assessments,
                    (SELECT count(*) FROM live_event_assessments WHERE risk_level = 'HIGH') AS assessed_high,
                    (SELECT count(*) FROM live_event_assessments WHERE risk_level = 'MEDIUM') AS assessed_medium,
                    (SELECT count(*) FROM live_event_assessments WHERE risk_level = 'LOW') AS assessed_low,
                    (SELECT count(DISTINCT host_name) FROM security_events) AS unique_hosts,
                    (SELECT count(DISTINCT user_name) FROM security_alerts WHERE user_name IS NOT NULL) AS unique_users_alerted,
                    (SELECT avg(risk_score) FROM security_alerts) AS avg_risk_score,
                    (SELECT max(risk_score) FROM security_alerts) AS max_risk_score
                """
            ).fetchone()

        return {
            "total_events": stats["total_events"],
            "total_alerts": stats["total_alerts"],
            "open_alerts": stats["open_alerts"],
            "high_risk": stats["high_risk"],
            "medium_risk": stats["medium_risk"],
            "review_signals": stats["review_signals"],
            "unclassified": stats["unclassified"],
            "processed_events": stats["processed_events"],
            "live_assessments": stats["live_assessments"],
            "assessed_high": stats["assessed_high"],
            "assessed_medium": stats["assessed_medium"],
            "assessed_low": stats["assessed_low"],
            "unique_hosts": stats["unique_hosts"],
            "unique_users_alerted": stats["unique_users_alerted"],
            "avg_risk_score": round(float(stats["avg_risk_score"] or 0), 1),
            "max_risk_score": round(float(stats["max_risk_score"] or 0), 1),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    except HTTPException:
        raise
    except psycopg.Error as exc:
        raise HTTPException(status_code=500, detail="Query error.") from exc


@router.get("/live-stats")
def live_stats():
    """Comprehensive live Windows event statistics from actual DB records.
    Used exclusively by the System Status page.
    No historical CERT user-day data is included here."""
    try:
        with get_connection() as conn:
            row = conn.execute("""
                SELECT
                    (SELECT count(*) FROM security_events) AS total_events,
                    (SELECT count(*) FROM live_event_processing) AS processed,
                    (SELECT count(*) FROM security_events) -
                        (SELECT count(*) FROM live_event_processing) AS awaiting_processing,
                    (SELECT count(*) FROM live_event_processing
                        WHERE signal_status = 'REVIEW_SIGNAL') AS classified,
                    (SELECT count(*) FROM live_event_processing
                        WHERE signal_status = 'UNCLASSIFIED') AS unclassified,
                    (SELECT count(*) FROM live_event_processing
                        WHERE signal_status = 'REVIEW_SIGNAL') AS review_signals,
                    (SELECT count(*) FROM live_event_assessments) AS assessed,
                    (SELECT count(*) FROM live_event_assessments
                        WHERE correlated_events::text != '[]') AS correlated,
                    (SELECT count(*) FROM live_event_assessments
                        WHERE risk_level = 'HIGH') AS assessed_high,
                    (SELECT count(*) FROM live_event_assessments
                        WHERE risk_level = 'MEDIUM') AS assessed_medium,
                    (SELECT count(*) FROM live_event_assessments
                        WHERE risk_level = 'LOW') AS assessed_low,
                    (SELECT count(*) FROM security_alerts
                        WHERE status = 'OPEN') AS open_alerts,
                    (SELECT count(*) FROM security_alerts
                        WHERE status = 'INVESTIGATING') AS investigating_alerts,
                    (SELECT count(DISTINCT host_name) FROM security_events) AS unique_hosts,
                    (SELECT max(event_timestamp) FROM security_events) AS last_event_time,
                    (SELECT max(assessed_at) FROM live_event_assessments) AS last_assessment_time
            """).fetchone()

            total = row["total_events"] or 0
            processed = row["processed"] or 0
            awaiting = max(0, total - processed)

        from datetime import datetime, timezone
        return {
            "total_events": total,
            "processed": processed,
            "awaiting_processing": awaiting,
            "classified": row["classified"] or 0,
            "unclassified": row["unclassified"] or 0,
            "review_signals": row["review_signals"] or 0,
            "assessed": row["assessed"] or 0,
            "correlated": row["correlated"] or 0,
            "assessed_high": row["assessed_high"] or 0,
            "assessed_medium": row["assessed_medium"] or 0,
            "assessed_low": row["assessed_low"] or 0,
            "open_alerts": row["open_alerts"] or 0,
            "investigating_alerts": row["investigating_alerts"] or 0,
            "unique_hosts": row["unique_hosts"] or 0,
            "last_event_time": row["last_event_time"].isoformat() if row["last_event_time"] else None,
            "last_assessment_time": row["last_assessment_time"].isoformat() if row["last_assessment_time"] else None,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data_source": "PostgreSQL live tables only — no historical CERT data",
        }
    except HTTPException:
        raise
    except psycopg.Error as exc:
        raise HTTPException(status_code=500, detail="Query error.") from exc


@router.get("/live-timeline")
def live_event_timeline(
    limit: int = 100,
    event_code: str = None,
    host_name: str = None,
    user_name: str = None,
    category: str = None,
    risk_level: str = None,
):
    """Paginated live event feed with processing and assessment data.
    Supports filters by event_code, host_name, user_name, category, risk_level."""
    from fastapi import Query as FQuery
    conditions = ["1=1"]
    params = []

    if event_code:
        conditions.append("e.event_code = %s")
        params.append(event_code)
    if host_name:
        conditions.append("e.host_name ILIKE %s")
        params.append(f"%{host_name}%")
    if user_name:
        conditions.append("e.user_name ILIKE %s")
        params.append(f"%{user_name}%")
    if category:
        conditions.append("p.event_category = %s")
        params.append(category)
    if risk_level:
        conditions.append("ass.risk_level = %s")
        params.append(risk_level)

    where = " AND ".join(conditions)
    params.append(min(limit, 500))

    sql = f"""
        SELECT
            e.event_id,
            e.event_timestamp,
            e.host_name,
            e.user_name,
            e.event_code,
            e.event_source,
            e.event_type,
            p.event_category,
            p.signal_status,
            p.classification_reason,
            ass.assessment_status,
            ass.risk_score,
            ass.risk_level,
            ass.priority,
            ass.scoring_reason,
            ass.scoring_breakdown,
            ass.extracted_evidence,
            ass.correlated_events,
            ass.scoring_mode,
            ass.assessed_at
        FROM security_events e
        LEFT JOIN live_event_processing p ON p.event_id = e.event_id
        LEFT JOIN live_event_assessments ass ON ass.event_id = e.event_id
        WHERE {where}
        ORDER BY e.event_timestamp DESC, e.event_id DESC
        LIMIT %s
    """

    try:
        with get_connection() as conn:
            rows = conn.execute(sql, params).fetchall()

        def fmt(r):
            return {
                "event_id": r["event_id"],
                "event_timestamp": r["event_timestamp"].isoformat() if r["event_timestamp"] else None,
                "host_name": r["host_name"],
                "user_name": r["user_name"],
                "event_code": r["event_code"],
                "event_source": r["event_source"],
                "event_type": r["event_type"],
                "event_category": r["event_category"],
                "signal_status": r["signal_status"],
                "classification_reason": r["classification_reason"],
                "assessment_status": r["assessment_status"],
                "risk_score": round(float(r["risk_score"]), 1) if r["risk_score"] is not None else None,
                "risk_level": r["risk_level"],
                "priority": r["priority"],
                "scoring_reason": r["scoring_reason"],
                "scoring_breakdown": r["scoring_breakdown"],
                "extracted_evidence": r["extracted_evidence"],
                "correlated_events": r["correlated_events"],
                "scoring_mode": r["scoring_mode"],
                "assessed_at": r["assessed_at"].isoformat() if r["assessed_at"] else None,
            }

        return {
            "count": len(rows),
            "events": [fmt(r) for r in rows],
            "data_source": "PostgreSQL live tables",
        }
    except HTTPException:
        raise
    except psycopg.Error as exc:
        raise HTTPException(status_code=500, detail="Query error.") from exc


@router.get("/live-assessments")
def live_assessments(limit: int = 50, risk_level: str = None):
    """All live event assessments with detection attribution fields."""
    conditions = ["1=1"]
    params = []
    if risk_level:
        conditions.append("ass.risk_level = %s")
        params.append(risk_level)
    params.append(min(limit, 500))
    where = " AND ".join(conditions)

    sql = f"""
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
            p.signal_status
        FROM live_event_assessments ass
        JOIN security_events e ON e.event_id = ass.event_id
        LEFT JOIN live_event_processing p ON p.event_id = ass.event_id
        WHERE {where}
        ORDER BY ass.assessed_at DESC, ass.assessment_id DESC
        LIMIT %s
    """
    try:
        with get_connection() as conn:
            rows = conn.execute(sql, params).fetchall()

        def fmt(r):
            bd = r["scoring_breakdown"] or {}
            triggered = [
                {"rule": k, "score": v, "description": _rule_description(k)}
                for k, v in bd.items()
            ]
            return {
                "assessment_id": r["assessment_id"],
                "event_id": r["event_id"],
                "assessment_status": r["assessment_status"],
                "risk_score": round(float(r["risk_score"]), 1) if r["risk_score"] is not None else None,
                "risk_level": r["risk_level"],
                "priority": r["priority"],
                "scoring_reason": r["scoring_reason"],
                "recommended_action": r["recommended_action"],
                "detection_source": "Live Windows Rule-Based Risk Engine",
                "scoring_mode": r["scoring_mode"] or "LIVE_WINDOWS_SERVICE_RULE",
                "triggered_rules": triggered,
                "scoring_breakdown": bd,
                "extracted_evidence": r["extracted_evidence"],
                "correlated_events": r["correlated_events"],
                "assessed_at": r["assessed_at"].isoformat() if r["assessed_at"] else None,
                "host_name": r["host_name"],
                "user_name": r["user_name"],
                "event_code": r["event_code"],
                "event_timestamp": r["event_timestamp"].isoformat() if r["event_timestamp"] else None,
                "event_source": r["event_source"],
                "event_category": r["event_category"],
                "signal_status": r["signal_status"],
            }

        return {"count": len(rows), "assessments": [fmt(r) for r in rows]}
    except HTTPException:
        raise
    except psycopg.Error as exc:
        raise HTTPException(status_code=500, detail="Query error.") from exc


def _rule_description(rule_key: str) -> str:
    descriptions = {
        "base_lifecycle_signal": "Windows service lifecycle event detected (base signal).",
        "correlated_lifecycle_modification": "Correlated service install and start-type change on same host within 5 minutes.",
        "suspicious_service_path": "Service binary located in a user-writable or temporary directory.",
        "non_system_service_account": "Service configured under a non-standard user account.",
        "auto_start_persistence": "Service configured for automatic start — possible persistence mechanism.",
        "unusual_login_time": "Login recorded outside configured working hours (08:00–22:00).",
        "failed_login_burst": "Multiple failed login attempts detected within a short time window.",
        "device_activity": "Removable device activity detected.",
    }
    return descriptions.get(rule_key, rule_key.replace("_", " ").title())

