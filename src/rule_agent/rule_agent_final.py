from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BEHAVIOR_FILE = PROJECT_ROOT / "data" / "outputs" / "final_behavioral_features.parquet"
CONTEXT_FILE = PROJECT_ROOT / "data" / "outputs" / "rule_context_features.parquet"

OUTPUT_FILE = PROJECT_ROOT / "data" / "outputs" / "final_rule_agent_results.parquet"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FINAL RULE AGENT - REVISED VERSION")
print("=" * 70)

print("\nLoading behavioral features...")
behavior_df = pd.read_parquet(BEHAVIOR_FILE)

print("Loading rule context features...")
context_df = pd.read_parquet(CONTEXT_FILE)

print(f"Behavioral rows : {len(behavior_df):,}")
print(f"Context rows    : {len(context_df):,}")


# ============================================================
# SELECT ONLY VERIFIED CONTEXT FEATURES
# ============================================================

context_columns = [
    "user",
    "day",
    "after_hours_logons",
    "connect_events",
    "connect_disconnect_ratio",
    "unique_logon_pcs",
    "unique_device_pcs",
    "unique_file_extensions",
]

context_df = context_df[context_columns].copy()


# ============================================================
# MERGE
# ============================================================

df = behavior_df.merge(
    context_df,
    on=["user", "day"],
    how="left",
    validate="one_to_one"
)

print(f"Merged rows    : {len(df):,}")


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

numeric_context_columns = [
    "after_hours_logons",
    "connect_events",
    "connect_disconnect_ratio",
    "unique_logon_pcs",
    "unique_device_pcs",
    "unique_file_extensions",
]

for col in numeric_context_columns:
    df[col] = df[col].fillna(0)


# ============================================================
# PRIMARY BEHAVIORAL RULES
#
# These are based on the clean behavioral baseline that was
# already created from NORMAL user-days.
# ============================================================

df["rule_high_activity"] = (
    df["total_activity_clean_zscore"] >= 3
)

df["rule_unusual_logon"] = (
    df["logon_events_clean_zscore"] >= 3
)

df["rule_unusual_device"] = (
    df["device_events_clean_zscore"] >= 3
)

df["rule_unusual_file"] = (
    df["file_events_clean_zscore"] >= 3
)

df["rule_unusual_email"] = (
    df["emails_sent_clean_zscore"] >= 3
)

df["rule_unusual_web"] = (
    df["web_visits_clean_zscore"] >= 3
)

df["rule_behavior_deviation"] = (
    df["clean_behavior_deviation"] >= 3
)


primary_rules = [
    "rule_high_activity",
    "rule_unusual_logon",
    "rule_unusual_device",
    "rule_unusual_file",
    "rule_unusual_email",
    "rule_unusual_web",
    "rule_behavior_deviation",
]


# ============================================================
# SUPPORTING CONTEXT RULES
#
# IMPORTANT:
# These rules DO NOT independently create an alert.
# They only strengthen a behavioral anomaly.
#
# These conditions are based on the actual raw CERT data
# inspected earlier:
#   Logon activities: Logon / Logoff
#   Device activities: Connect / Disconnect
# ============================================================

df["rule_after_hours_login"] = (
    df["after_hours_logons"] > 0
)

df["rule_device_connection_pattern"] = (
    (df["connect_events"] > 0)
    &
    (df["connect_disconnect_ratio"] >= 0.5)
)

df["rule_multiple_logon_pcs"] = (
    df["unique_logon_pcs"] >= 2
)

df["rule_multiple_device_pcs"] = (
    df["unique_device_pcs"] >= 2
)

df["rule_file_type_activity"] = (
    df["unique_file_extensions"] > 0
)


supporting_rules = [
    "rule_after_hours_login",
    "rule_device_connection_pattern",
    "rule_multiple_logon_pcs",
    "rule_multiple_device_pcs",
    "rule_file_type_activity",
]


# ============================================================
# COUNT RULES
# ============================================================

df["primary_rule_count"] = df[primary_rules].sum(axis=1)

df["supporting_rule_count"] = df[supporting_rules].sum(axis=1)

df["rule_alert_count"] = (
    df["primary_rule_count"]
    + df["supporting_rule_count"]
)


# ============================================================
# RULE SCORE
#
# Score is based ONLY on primary behavioral anomalies.
# Context rules are supporting evidence and therefore do not
# artificially inflate the behavioral anomaly score.
# ============================================================

df["rule_score"] = (
    df["primary_rule_count"] / len(primary_rules) * 100
)


# ============================================================
# FINAL ALERT LOGIC
#
# A context rule alone is NOT enough.
#
# Alert when:
#
# 1. At least TWO primary behavioral anomalies
#
# OR
#
# 2. At least ONE primary behavioral anomaly AND
#    at least TWO supporting context indicators
# ============================================================

df["rule_triggered"] = (
    (df["primary_rule_count"] >= 2)
    |
    (
        (df["primary_rule_count"] >= 1)
        &
        (df["supporting_rule_count"] >= 2)
    )
)


# ============================================================
# THREAT LEVEL
# ============================================================

df["threat_level"] = "NORMAL"

# One primary behavioral anomaly
df.loc[
    df["primary_rule_count"] == 1,
    "threat_level"
] = "LOW"

# Medium:
# Two or more primary rules
# OR one primary + two supporting indicators
medium_condition = (
    (df["primary_rule_count"] >= 2)
    |
    (
        (df["primary_rule_count"] >= 1)
        &
        (df["supporting_rule_count"] >= 2)
    )
)

df.loc[
    medium_condition,
    "threat_level"
] = "MEDIUM"


# High:
# Three or more primary behavioral anomalies
# OR two primary + two supporting indicators

high_condition = (
    (df["primary_rule_count"] >= 3)
    |
    (
        (df["primary_rule_count"] >= 2)
        &
        (df["supporting_rule_count"] >= 2)
    )
)

df.loc[
    high_condition,
    "threat_level"
] = "HIGH"


# ============================================================
# EXPLANATIONS
# ============================================================

primary_rule_names = {
    "rule_high_activity": "High overall activity",
    "rule_unusual_logon": "Unusual logon activity",
    "rule_unusual_device": "Unusual device activity",
    "rule_unusual_file": "Unusual file activity",
    "rule_unusual_email": "Unusual email activity",
    "rule_unusual_web": "Unusual web activity",
    "rule_behavior_deviation": "Overall behavioral deviation",
}

supporting_rule_names = {
    "rule_after_hours_login": "After-hours login",
    "rule_device_connection_pattern": "Device connection pattern",
    "rule_multiple_logon_pcs": "Multiple logon PCs",
    "rule_multiple_device_pcs": "Multiple device PCs",
    "rule_file_type_activity": "File type activity",
}


def build_explanation(row):

    explanations = []

    for rule, name in primary_rule_names.items():
        if bool(row[rule]):
            explanations.append(f"Primary: {name}")

    for rule, name in supporting_rule_names.items():
        if bool(row[rule]):
            explanations.append(f"Supporting: {name}")

    return "; ".join(explanations)


df["triggered_rules"] = df.apply(
    build_explanation,
    axis=1
)


# ============================================================
# SAVE
# ============================================================

df.to_parquet(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("REVISED RULE AGENT SUMMARY")
print("=" * 70)

print(f"\nRows processed       : {len(df):,}")

print(
    f"Rule-triggered days  : "
    f"{df['rule_triggered'].sum():,}"
)

print("\nThreat levels:")
print(
    df["threat_level"]
    .value_counts()
    .sort_index()
)

print("\nPrimary rule counts:")
print(
    df["primary_rule_count"]
    .value_counts()
    .sort_index()
)

print("\nSupporting rule counts:")
print(
    df["supporting_rule_count"]
    .value_counts()
    .sort_index()
)

print("\nPrimary rule contributions:")

for rule in primary_rules:
    print(
        f"{rule:<35} "
        f"{df[rule].sum():,}"
    )

print("\nSupporting context contributions:")

for rule in supporting_rules:
    print(
        f"{rule:<35} "
        f"{df[rule].sum():,}"
    )

print("\nOutput saved to:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("RULE AGENT COMPLETED")
print("=" * 70)