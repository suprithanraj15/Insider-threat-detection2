from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "outputs" / "final_behavioral_features.parquet"
OUTPUT_FILE = PROJECT_ROOT / "data" / "outputs" / "rule_agent_results.parquet"


print("=" * 60)
print("RULE-BASED INSIDER THREAT AGENT")
print("=" * 60)

# ------------------------------------------------------------
# 1. Load final behavioral features
# ------------------------------------------------------------

print("\n1. Loading final behavioral features...")

df = pd.read_parquet(INPUT_FILE)

print(f"Rows: {len(df):,}")
print(f"Users: {df['user'].nunique():,}")


# ------------------------------------------------------------
# 2. Initialize rule results
# ------------------------------------------------------------

df["rule_score"] = 0
df["rule_alert_count"] = 0

df["rule_high_activity"] = 0
df["rule_unusual_logon"] = 0
df["rule_unusual_device"] = 0
df["rule_unusual_file"] = 0
df["rule_unusual_email"] = 0
df["rule_unusual_web"] = 0
df["rule_behavior_deviation"] = 0


# ------------------------------------------------------------
# 3. Rule thresholds
# ------------------------------------------------------------

# Z-score threshold
Z_THRESHOLD = 3.0

# Overall behavioral deviation threshold
BEHAVIOR_THRESHOLD = 3.0


# ------------------------------------------------------------
# 4. Rule 1 — High overall activity
# ------------------------------------------------------------

df["rule_high_activity"] = (
    df["total_activity_clean_zscore"] >= Z_THRESHOLD
).astype(int)


# ------------------------------------------------------------
# 5. Rule 2 — Unusual login behavior
# ------------------------------------------------------------

df["rule_unusual_logon"] = (
    df["logon_events_clean_zscore"] >= Z_THRESHOLD
).astype(int)


# ------------------------------------------------------------
# 6. Rule 3 — Unusual device activity
# ------------------------------------------------------------

df["rule_unusual_device"] = (
    df["device_events_clean_zscore"] >= Z_THRESHOLD
).astype(int)


# ------------------------------------------------------------
# 7. Rule 4 — Unusual file activity
# ------------------------------------------------------------

df["rule_unusual_file"] = (
    df["file_events_clean_zscore"] >= Z_THRESHOLD
).astype(int)


# ------------------------------------------------------------
# 8. Rule 5 — Unusual email activity
# ------------------------------------------------------------

df["rule_unusual_email"] = (
    df["emails_sent_clean_zscore"] >= Z_THRESHOLD
).astype(int)


# ------------------------------------------------------------
# 9. Rule 6 — Unusual web activity
# ------------------------------------------------------------

df["rule_unusual_web"] = (
    df["web_visits_clean_zscore"] >= Z_THRESHOLD
).astype(int)


# ------------------------------------------------------------
# 10. Rule 7 — Overall behavioral deviation
# ------------------------------------------------------------

df["rule_behavior_deviation"] = (
    df["clean_behavior_deviation"] >= BEHAVIOR_THRESHOLD
).astype(int)


# ------------------------------------------------------------
# 11. Calculate total rule score
# ------------------------------------------------------------

rule_columns = [
    "rule_high_activity",
    "rule_unusual_logon",
    "rule_unusual_device",
    "rule_unusual_file",
    "rule_unusual_email",
    "rule_unusual_web",
    "rule_behavior_deviation"
]

df["rule_alert_count"] = df[rule_columns].sum(axis=1)

df["rule_score"] = (
    df["rule_alert_count"] / len(rule_columns)
) * 100


# ------------------------------------------------------------
# 12. Rule-based threat level
# ------------------------------------------------------------

def classify_rule_threat(score):

    if score >= 70:
        return "HIGH"

    elif score >= 40:
        return "MEDIUM"

    elif score > 0:
        return "LOW"

    return "NORMAL"


df["rule_threat_level"] = df["rule_score"].apply(
    classify_rule_threat
)


# ------------------------------------------------------------
# 13. Rule triggered flag
# ------------------------------------------------------------

df["rule_triggered"] = (
    df["rule_alert_count"] > 0
).astype(int)


# ------------------------------------------------------------
# 14. Save results
# ------------------------------------------------------------

df.to_parquet(
    OUTPUT_FILE,
    index=False
)


# ------------------------------------------------------------
# 15. Summary
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("RULE AGENT COMPLETED")
print("=" * 60)

print(f"Rows processed: {len(df):,}")

print(
    f"User-days triggering rules: "
    f"{df['rule_triggered'].sum():,}"
)

print("\nThreat-level distribution:")

print(
    df["rule_threat_level"]
    .value_counts()
    .to_string()
)

print("\nRule trigger counts:")

for column in rule_columns:
    print(
        f"  {column}: "
        f"{df[column].sum():,}"
    )

print("\nSaved:")
print(OUTPUT_FILE)