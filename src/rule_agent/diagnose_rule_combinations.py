from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONTEXT_FILE = PROJECT_ROOT / "data" / "outputs" / "rule_context_features.parquet"
GROUND_TRUTH_FILE = PROJECT_ROOT / "data" / "outputs" / "final_behavioral_features.parquet"

print("=" * 70)
print("RULE AGENT - RULE COMBINATION DIAGNOSTIC")
print("=" * 70)

# Load data
context_df = pd.read_parquet(CONTEXT_FILE)
gt_df = pd.read_parquet(GROUND_TRUTH_FILE)

df = context_df.merge(
    gt_df[["user", "day", "ground_truth", "scenario"]],
    on=["user", "day"],
    how="inner"
)

print(f"\nJoined rows: {len(df):,}")
print(f"Malicious user-days: {(df['ground_truth'] == 1).sum():,}")
print(f"Normal user-days: {(df['ground_truth'] == 0).sum():,}")

# ------------------------------------------------------------
# Define only features already verified in rule_context_features
# ------------------------------------------------------------

df["device_activity"] = (
    (df["connect_events"] > 0) |
    (df["disconnect_events"] > 0)
)

df["logon_activity"] = (
    (df["after_hours_logons"] > 0) |
    (df["weekend_logons"] > 0)
)

df["file_activity"] = (
    (df["unique_file_extensions"] > 0) |
    (df["executable_files"] > 0) |
    (df["archive_files"] > 0)
)

# ------------------------------------------------------------
# Rule combinations
# ------------------------------------------------------------

df["device_plus_logon"] = (
    df["device_activity"] & df["logon_activity"]
)

df["device_plus_file"] = (
    df["device_activity"] & df["file_activity"]
)

df["logon_plus_file"] = (
    df["logon_activity"] & df["file_activity"]
)

df["device_plus_high_ratio"] = (
    df["device_activity"] &
    (df["connect_disconnect_ratio"] >= 0.5)
)

df["multiple_device_pcs"] = (
    df["unique_device_pcs"] >= 2
)

df["multiple_logon_pcs"] = (
    df["unique_logon_pcs"] >= 2
)

df["multiple_logons_and_devices"] = (
    (df["unique_logon_pcs"] >= 2) &
    (df["unique_device_pcs"] >= 2)
)

combinations = [
    "device_plus_logon",
    "device_plus_file",
    "logon_plus_file",
    "device_plus_high_ratio",
    "multiple_device_pcs",
    "multiple_logon_pcs",
    "multiple_logons_and_devices",
]

# ------------------------------------------------------------
# Diagnostic
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("COMBINATION OCCURRENCE")
print("=" * 70)

for rule in combinations:

    malicious = df[df["ground_truth"] == 1]
    normal = df[df["ground_truth"] == 0]

    mal_count = int(malicious[rule].sum())
    normal_count = int(normal[rule].sum())

    print(
        f"{rule:30s} : "
        f"Malicious {mal_count:2d}/32 | "
        f"Normal {normal_count:,}/342674"
    )

# ------------------------------------------------------------
# Scenario-wise analysis
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SCENARIO-WISE COMBINATION ANALYSIS")
print("=" * 70)

for scenario in sorted(df.loc[df["ground_truth"] == 1, "scenario"].dropna().unique()):

    scenario_df = df[
        (df["ground_truth"] == 1) &
        (df["scenario"] == scenario)
    ]

    print(f"\nScenario {int(scenario)} ({len(scenario_df)} malicious days)")

    for rule in combinations:
        count = int(scenario_df[rule].sum())
        print(f"  {rule:30s}: {count}/{len(scenario_df)}")

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETED")
print("=" * 70)