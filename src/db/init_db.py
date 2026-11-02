
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


-- ============================================================
-- ATTACK SIMULATION TABLES
-- These tables are strictly separate from real Windows events.
-- ============================================================

CREATE TABLE IF NOT EXISTS simulation_runs (
    simulation_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    scenario_key TEXT NOT NULL,
    scenario_name TEXT NOT NULL,

    host_name TEXT NOT NULL,
    user_name TEXT,

    status TEXT NOT NULL DEFAULT 'RUNNING'
        CHECK (
            status IN ('RUNNING', 'COMPLETED', 'FAILED')
        ),

    expected_result JSONB NOT NULL DEFAULT '{}'::jsonb,
    actual_result JSONB NOT NULL DEFAULT '{}'::jsonb,

    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,

    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_simulation_runs_created
    ON simulation_runs (started_at DESC);

CREATE INDEX IF NOT EXISTS idx_simulation_runs_scenario
    ON simulation_runs (scenario_key);

CREATE INDEX IF NOT EXISTS idx_simulation_runs_status
    ON simulation_runs (status);


CREATE TABLE IF NOT EXISTS sim_events (
    sim_event_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    simulation_id BIGINT NOT NULL
        REFERENCES simulation_runs(simulation_id)
        ON DELETE CASCADE,

    event_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    host_name TEXT NOT NULL,
    user_name TEXT,

    event_source TEXT NOT NULL,
    event_type TEXT NOT NULL,
    event_code TEXT,

    scenario_key TEXT NOT NULL,

    details JSONB NOT NULL DEFAULT '{}'::jsonb,

    expected_risk_level TEXT
        CHECK (
            expected_risk_level IS NULL
            OR expected_risk_level IN ('LOW', 'MEDIUM', 'HIGH')
        ),

    detected_risk_score DOUBLE PRECISION
        CHECK (
            detected_risk_score IS NULL
            OR (
                detected_risk_score >= 0
                AND detected_risk_score <= 100
            )
        ),

    detected_risk_level TEXT
        CHECK (
            detected_risk_level IS NULL
            OR detected_risk_level IN ('LOW', 'MEDIUM', 'HIGH')
        ),

    detection_status TEXT NOT NULL DEFAULT 'PENDING'
        CHECK (
            detection_status IN (
                'PENDING',
                'DETECTED',
                'NOT_DETECTED',
                'ERROR'
            )
        ),

    detection_evidence JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sim_events_simulation
    ON sim_events (simulation_id);

CREATE INDEX IF NOT EXISTS idx_sim_events_timestamp
    ON sim_events (event_timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_sim_events_scenario
    ON sim_events (scenario_key);

CREATE INDEX IF NOT EXISTS idx_sim_events_detection
    ON sim_events (detection_status, detected_risk_level);
"""


def main():
    with psycopg.connect(**DB_CONFIG) as conn:
        conn.execute(SCHEMA_SQL)

    print("Database schema initialized successfully.")
    print(
        "Created or verified tables: "
        "security_events, live_event_processing, "
        "security_alerts, live_event_assessments, "
        "simulation_runs, sim_events"
    )
    print("Indexes created or verified.")


if __name__ == "__main__":
    main()
