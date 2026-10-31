
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

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

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS security_events (
    event_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_event_id TEXT,
    event_timestamp TIMESTAMPTZ NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    host_name TEXT NOT NULL,
    user_name TEXT,
    event_source TEXT NOT NULL,
    event_type TEXT NOT NULL,
    event_code TEXT,
    details JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_security_events_timestamp
    ON security_events (event_timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_security_events_user_time
    ON security_events (user_name, event_timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_security_events_code
    ON security_events (event_code);

-- Required by live ingest ON CONFLICT for Windows System events.
CREATE UNIQUE INDEX IF NOT EXISTS idx_security_events_system_source
    ON security_events (source_event_id)
    WHERE source_event_id LIKE 'System-%';


CREATE TABLE IF NOT EXISTS live_event_processing (
    event_id BIGINT PRIMARY KEY
        REFERENCES security_events(event_id)
        ON DELETE CASCADE,
    processed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_category TEXT NOT NULL,
    signal_status TEXT NOT NULL,
    classification_reason TEXT NOT NULL,
    evidence JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_live_processing_category
    ON live_event_processing (event_category);

CREATE INDEX IF NOT EXISTS idx_live_processing_status
    ON live_event_processing (signal_status);

CREATE INDEX IF NOT EXISTS idx_live_processing_time
    ON live_event_processing (processed_at DESC);


CREATE TABLE IF NOT EXISTS security_alerts (
    alert_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    event_id BIGINT REFERENCES security_events(event_id)
        ON DELETE SET NULL,
    user_name TEXT,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    risk_score DOUBLE PRECISION NOT NULL
        CHECK (risk_score >= 0 AND risk_score <= 100),
    risk_level TEXT NOT NULL
        CHECK (risk_level IN ('LOW', 'MEDIUM', 'HIGH')),
    priority TEXT NOT NULL
        CHECK (priority IN ('P1', 'P2', 'P3')),
    explanation TEXT NOT NULL,
    recommended_action TEXT,
    agent_evidence JSONB NOT NULL DEFAULT '{}'::jsonb,
    status TEXT NOT NULL DEFAULT 'OPEN'
        CHECK (
            status IN (
                'OPEN',
                'INVESTIGATING',
                'RESOLVED',
                'FALSE_POSITIVE'
            )
        )
);

CREATE INDEX IF NOT EXISTS idx_security_alerts_detected
    ON security_alerts (detected_at DESC);

CREATE INDEX IF NOT EXISTS idx_security_alerts_level
    ON security_alerts (risk_level, detected_at DESC);

CREATE INDEX IF NOT EXISTS idx_security_alerts_user
    ON security_alerts (user_name, detected_at DESC);


-- Stores the complete assessment for each processed event.
CREATE TABLE IF NOT EXISTS live_event_assessments (
    assessment_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    event_id BIGINT NOT NULL UNIQUE
        REFERENCES security_events(event_id)
        ON DELETE CASCADE,

    assessment_status TEXT NOT NULL,

    risk_score DOUBLE PRECISION
        CHECK (
            risk_score IS NULL
            OR (risk_score >= 0 AND risk_score <= 100)
        ),

    risk_level TEXT
        CHECK (
            risk_level IS NULL
            OR risk_level IN ('LOW', 'MEDIUM', 'HIGH')
        ),

    priority TEXT
        CHECK (
            priority IS NULL
            OR priority IN ('P1', 'P2', 'P3')
        ),

    scoring_reason TEXT,
    recommended_action TEXT,

    scoring_breakdown JSONB NOT NULL DEFAULT '{}'::jsonb,
    extracted_evidence JSONB NOT NULL DEFAULT '{}'::jsonb,
    correlated_events JSONB NOT NULL DEFAULT '[]'::jsonb,

    scoring_mode TEXT NOT NULL DEFAULT 'LIVE_WINDOWS_SERVICE_RULE',

    assessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_live_assessments_status
    ON live_event_assessments (assessment_status);

CREATE INDEX IF NOT EXISTS idx_live_assessments_risk
    ON live_event_assessments (risk_level, assessed_at DESC);

CREATE INDEX IF NOT EXISTS idx_live_assessments_time
    ON live_event_assessments (assessed_at DESC);
"""


def main():
    with psycopg.connect(**DB_CONFIG) as conn:
        conn.execute(SCHEMA_SQL)

    print("Database schema initialized successfully.")
    print(
        "Created or verified tables: "
        "security_events, live_event_processing, "
        "security_alerts, live_event_assessments"
    )
    print("Indexes created or verified.")


if __name__ == "__main__":
    main()

