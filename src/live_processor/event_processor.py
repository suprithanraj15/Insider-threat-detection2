from pathlib import Path
import os

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


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


# --------------------------------------------------
# EVENT CLASSIFICATION
# --------------------------------------------------

def classify_event(event):
    """Classify configured Windows System events as review signals."""

    code = str(event.get("event_code") or "").strip()
    source = str(event.get("event_source") or "").strip()
    details = event.get("details") or {}
    message = str(details.get("message") or "")

    if code == "7045" and source == "Service Control Manager":
        category = "SERVICE_INSTALLED"
        reason = "A Windows service installation was recorded."

    elif code == "7040" and source == "Service Control Manager":
        category = "SERVICE_START_TYPE_CHANGED"
        reason = "A Windows service start type was changed."

    elif code == "4624" and source == "Microsoft-Windows-Security-Auditing":
        # Successful logon
        category = "LOGON_SUCCESS"
        reason = "A successful Windows logon was recorded."

    elif code == "4625" and source == "Microsoft-Windows-Security-Auditing":
        # Failed logon
        category = "LOGON_FAILURE"
        reason = "A failed Windows logon attempt was recorded."

    elif code in ("2003", "2010", "2100") and "DriverFrameworks" in source:
        # USB / removable device events from Windows Kernel-PnP
        category = "DEVICE_ACTIVITY"
        reason = "Removable device activity detected via Windows DriverFrameworks."

    elif code == "6416" and source == "Microsoft-Windows-Security-Auditing":
        # External device recognized (requires Object Access auditing)
        category = "DEVICE_ACTIVITY"
        reason = "External device recognized event detected (Event 6416)."

    else:
        category = "UNCLASSIFIED"
        reason = "No configured event classification matched."

    signal_status = (
        "REVIEW_SIGNAL"
        if category in (
            "SERVICE_INSTALLED",
            "SERVICE_START_TYPE_CHANGED",
            "LOGON_FAILURE",
            "DEVICE_ACTIVITY",
            "LOGON_SUCCESS",  # will be further filtered by risk engine based on time
        )
        else "UNCLASSIFIED"
    )

    event_ts = event.get("event_timestamp")
    ts_str = (
        event_ts.isoformat()
        if hasattr(event_ts, "isoformat")
        else str(event_ts or "")
    )

    return {
        "event_category": category,
        "classification_reason": reason,
        "event_code": code,
        "event_source": source,
        "host_name": event["host_name"],
        "user_name": event.get("user_name"),
        "event_timestamp": ts_str,
        "message": message,
        "signal_status": signal_status,
    }


# --------------------------------------------------
# PROCESSING TABLE
# --------------------------------------------------

def initialize_processor_table(conn):
    """Create a separate table for processing results."""

    conn.execute("""
        CREATE TABLE IF NOT EXISTS live_event_processing (
            event_id BIGINT PRIMARY KEY
                REFERENCES security_events(event_id)
                ON DELETE CASCADE,
            processed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            event_category TEXT NOT NULL,
            signal_status TEXT NOT NULL,
            classification_reason TEXT NOT NULL,
            evidence JSONB NOT NULL DEFAULT '{}'::jsonb
        )
    """)


# --------------------------------------------------
# PROCESS A BATCH
# --------------------------------------------------

def process_batch(conn):
    """Process events that do not yet have a processing record."""

    events = conn.execute("""
        SELECT
            e.event_id,
            e.source_event_id,
            e.event_timestamp,
            e.host_name,
            e.user_name,
            e.event_source,
            e.event_type,
            e.event_code,
            e.details
        FROM security_events AS e
        LEFT JOIN live_event_processing AS p
            ON p.event_id = e.event_id
        WHERE p.event_id IS NULL
        ORDER BY e.event_id
        LIMIT %s
        FOR UPDATE OF e SKIP LOCKED
    """, (BATCH_SIZE,)).fetchall()

    for event in events:
        result = classify_event(event)

        conn.execute("""
            INSERT INTO live_event_processing (
                event_id,
                event_category,
                signal_status,
                classification_reason,
                evidence
            )
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (event_id) DO NOTHING
        """, (
            event["event_id"],
            result["event_category"],
            result["signal_status"],
            result["classification_reason"],
            Jsonb(result),
        ))

        print(
            f"Event {event['event_id']} | "
            f"Code {result['event_code']} | "
            f"{result['event_category']} | "
            f"{result['signal_status']}"
        )

    return len(events)


# --------------------------------------------------
# MAIN
# --------------------------------------------------


def main():
    import time
    import argparse

    parser = argparse.ArgumentParser(description="Live Event Processor")
    parser.add_argument("--once", action="store_true", help="Process current pending events and exit")
    parser.add_argument("--poll", type=int, default=5, help="Polling interval in seconds")
    args = parser.parse_args()

    poll_seconds = args.poll

    print("=" * 60)
    print("LIVE EVENT PROCESSOR" + (" (SINGLE PASS)" if args.once else " (CONTINUOUS)"))
    print(f"Polling interval: {poll_seconds} seconds")
    print("=" * 60)

    while True:
        try:
            with psycopg.connect(
                **DB_CONFIG,
                row_factory=dict_row,
            ) as conn:
                initialize_processor_table(conn)
                processed = process_batch(conn)

            if processed:
                print(f"Processed {processed} event(s).")
            else:
                print("No new events. Waiting for the next poll...")

        except (psycopg.Error, OSError) as exc:
            print(f"Processor error: {exc}")

        if args.once:
            print("Single-pass processing complete. Exiting.")
            break

        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
