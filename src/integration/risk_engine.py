from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data" / "outputs" / "supervisor_agent_results.parquet"
OUTPUT_DIR = ROOT / "data" / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "risk_engine_results.parquet"

MEDIUM_THRESHOLD = 25
HIGH_THRESHOLD = 60


def assign_risk_level(score):
    if pd.isna(score):
        return "UNKNOWN"
    if score >= HIGH_THRESHOLD:
        return "HIGH"
    if score >= MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def assign_priority(level):
    return {
        "HIGH": "P1",
        "MEDIUM": "P2",
        "LOW": "P3",
        "UNKNOWN": "UNASSESSED",
    }.get(level, "UNASSESSED")


def generate_explanation(row):
    level = row["risk_level"]
    score = row["supervisor_risk_score"]

    if level == "UNKNOWN":
        return "Risk score is unavailable; manual assessment is required."
    if level == "HIGH":
        return f"High risk score ({score:.2f}/100); investigate this user-day promptly."
    if level == "MEDIUM":
        return f"Moderate risk score ({score:.2f}/100); review the user's activity."
    return f"Low risk score ({score:.2f}/100); continue routine monitoring."


def recommended_action(level):
    return {
        "HIGH": "Investigate activity and validate against supporting evidence.",
        "MEDIUM": "Review activity patterns and check for additional indicators.",
        "LOW": "Continue routine monitoring.",
        "UNKNOWN": "Verify available data and assess manually.",
    }.get(level, "Assess manually.")


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Supervisor output not found: {INPUT_FILE}")

    df = pd.read_parquet(INPUT_FILE)

    if "supervisor_risk_score" not in df.columns:
        raise ValueError(
            "Expected column 'supervisor_risk_score' was not found. "
            f"Available columns: {list(df.columns)}"
        )

    df["supervisor_risk_score"] = pd.to_numeric(
        df["supervisor_risk_score"], errors="coerce"
    )

    # Risk scores outside 0–100 are invalid; don't silently clamp them.
    invalid = df["supervisor_risk_score"].notna() & (
        (df["supervisor_risk_score"] < 0)
        | (df["supervisor_risk_score"] > 100)
    )
    if invalid.any():
        raise ValueError(
            f"Found {int(invalid.sum())} risk scores outside the 0–100 range."
        )

    df["risk_level"] = df["supervisor_risk_score"].apply(assign_risk_level)
    df["alert_priority"] = df["risk_level"].apply(assign_priority)
    df["alert_explanation"] = df.apply(generate_explanation, axis=1)
    df["recommended_action"] = df["risk_level"].apply(recommended_action)

    df["alert_status"] = np.where(
        df["risk_level"].isin(["HIGH", "MEDIUM"]),
        "OPEN",
        "MONITOR",
    )
    df["risk_engine_version"] = "1.0"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False)

    print("=" * 60)
    print("RISK ENGINE COMPLETED")
    print("=" * 60)
    print(f"Input rows : {len(pd.read_parquet(INPUT_FILE)):,}")
    print(f"Output rows: {len(df):,}")
    print(f"\nRisk levels:\n{df['risk_level'].value_counts(dropna=False)}")
    print(f"\nAlert priorities:\n{df['alert_priority'].value_counts(dropna=False)}")
    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
