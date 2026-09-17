from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

# This file is inside:
# src/rule_agent/
#
# Therefore:
# parents[0] = rule_agent
# parents[1] = src
# parents[2] = project root

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RULE_RESULTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "final_rule_agent_results.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "missed_malicious_days.csv"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("INSPECTION OF MISSED MALICIOUS USER-DAYS")
print("=" * 70)


# ============================================================
# LOAD FINAL RULE AGENT RESULTS
# ============================================================

print("\nLoading final Rule Agent results...")

df = pd.read_parquet(RULE_RESULTS_FILE)

print(f"Rows  : {len(df):,}")
print(f"Users : {df['user'].nunique():,}")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "user",
    "day",
    "ground_truth",
    "rule_triggered",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    print("\nERROR: Required columns are missing:")
    print(missing_columns)
    print("\nAvailable columns:")
    print(df.columns.tolist())
    raise SystemExit(1)


# ============================================================
# FIND MISSED MALICIOUS DAYS
# ============================================================

missed = df[
    (df["ground_truth"] == 1)
    &
    (df["rule_triggered"] == False)
].copy()


print("\n" + "=" * 70)
print("GROUND-TRUTH CHECK")
print("=" * 70)

print(
    f"\nTotal malicious user-days : "
    f"{int(df['ground_truth'].sum()):,}"
)

print(
    f"Missed malicious days     : "
    f"{len(missed):,}"
)


# ============================================================
# DISPLAY AVAILABLE COLUMNS
# ============================================================

print("\nColumns available for analysis:")

for col in missed.columns:
    print(f"  - {col}")


# ============================================================
# IMPORTANT FEATURE GROUPS
# ============================================================

activity_columns = [
    "total_activity",
    "active_channels",
    "logon_events",
    "device_events",
    "file_events",
    "emails_sent",
    "web_visits",
]

behavior_columns = [
    "total_activity_clean_zscore",
    "logon_events_clean_zscore",
    "device_events_clean_zscore",
    "file_events_clean_zscore",
    "emails_sent_clean_zscore",
    "web_visits_clean_zscore",
    "clean_behavior_deviation",
]

temporal_columns = [
    "total_activity_change_7d",
    "logon_events_change_7d",
    "device_events_change_7d",
    "file_events_change_7d",
    "emails_sent_change_7d",
    "web_visits_change_7d",
    "temporal_change_count",
    "temporal_change_score",
    "persistent_behavior_flag",
]

context_columns = [
    "after_hours_logons",
    "weekend_logons",
    "unique_logon_pcs",
    "total_logon_events",
    "connect_events",
    "disconnect_events",
    "unique_device_pcs",
    "unique_file_extensions",
    "executable_files",
    "archive_files",
    "connect_disconnect_ratio",
    "after_hours_logon_ratio",
]

rule_columns = [
    "rule_high_activity",
    "rule_unusual_logon",
    "rule_unusual_device",
    "rule_unusual_file",
    "rule_unusual_email",
    "rule_unusual_web",
    "rule_behavior_deviation",
    "rule_after_hours_login",
    "rule_device_connection_pattern",
    "rule_multiple_logon_pcs",
    "rule_multiple_device_pcs",
    "rule_file_type_activity",
]


# ============================================================
# ONLY USE COLUMNS THAT ACTUALLY EXIST
# ============================================================

activity_columns = [
    col for col in activity_columns
    if col in missed.columns
]

behavior_columns = [
    col for col in behavior_columns
    if col in missed.columns
]

temporal_columns = [
    col for col in temporal_columns
    if col in missed.columns
]

context_columns = [
    col for col in context_columns
    if col in missed.columns
]

rule_columns = [
    col for col in rule_columns
    if col in missed.columns
]


# ============================================================
# DISPLAY MISSED MALICIOUS DAYS
# ============================================================

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 100)


print("\n" + "=" * 70)
print("MISSED MALICIOUS DAYS - DETAILED ANALYSIS")
print("=" * 70)


for index, (_, row) in enumerate(missed.iterrows(), start=1):

    print("\n")
    print("-" * 70)
    print(f"MISSED MALICIOUS DAY {index} / {len(missed)}")
    print("-" * 70)

    # --------------------------------------------------------
    # Identity
    # --------------------------------------------------------

    print("\n[IDENTITY]")

    if "user" in row.index:
        print(f"User       : {row['user']}")

    if "day" in row.index:
        print(f"Day        : {row['day']}")

    if "scenario" in row.index:
        print(f"Scenario   : {row['scenario']}")

    print(f"Ground truth : {row['ground_truth']}")
    print(f"Rule triggered : {row['rule_triggered']}")


    # --------------------------------------------------------
    # Rule Agent information
    # --------------------------------------------------------

    print("\n[RULE AGENT]")

    for col in [
        "threat_level",
        "rule_score",
        "primary_rule_count",
        "supporting_rule_count",
        "rule_alert_count",
        "triggered_rules",
    ]:
        if col in row.index:
            print(f"{col:<25}: {row[col]}")


    # --------------------------------------------------------
    # Activity
    # --------------------------------------------------------

    print("\n[ACTIVITY]")

    for col in activity_columns:
        print(f"{col:<35}: {row[col]}")


    # --------------------------------------------------------
    # Behavioral anomaly features
    # --------------------------------------------------------

    print("\n[BEHAVIORAL DEVIATION]")

    for col in behavior_columns:
        print(f"{col:<35}: {row[col]}")


    # --------------------------------------------------------
    # Temporal behavior
    # --------------------------------------------------------

    print("\n[TEMPORAL BEHAVIOR]")

    for col in temporal_columns:
        print(f"{col:<35}: {row[col]}")


    # --------------------------------------------------------
    # Context
    # --------------------------------------------------------

    print("\n[CONTEXT]")

    for col in context_columns:
        print(f"{col:<35}: {row[col]}")


    # --------------------------------------------------------
    # Individual rule states
    # --------------------------------------------------------

    print("\n[RULE STATES]")

    for col in rule_columns:
        print(f"{col:<35}: {row[col]}")


# ============================================================
# SAVE MISSED DAYS
# ============================================================

missed.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SCENARIO SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MISSED DAYS BY SCENARIO")
print("=" * 70)

if "scenario" in missed.columns:

    scenario_counts = (
        missed["scenario"]
        .value_counts()
        .sort_index()
    )

    for scenario, count in scenario_counts.items():
        print(
            f"Scenario {scenario}: "
            f"{count} missed malicious day(s)"
        )


# ============================================================
# USER SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MISSED DAYS BY USER")
print("=" * 70)

user_counts = (
    missed["user"]
    .value_counts()
)

for user, count in user_counts.items():
    print(
        f"{user}: {count} missed malicious day(s)"
    )


# ============================================================
# OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("INSPECTION COMPLETED")
print("=" * 70)

print("\nMissed malicious days saved to:")
print(OUTPUT_FILE)

print("\nWe will use these actual missed days to improve the")
print("Rule Agent instead of blindly changing thresholds.")

print("=" * 70)