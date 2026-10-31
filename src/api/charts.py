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
