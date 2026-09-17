from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "final_rule_agent_results.parquet"
)

df = pd.read_parquet(INPUT_FILE)

normal = df[df["ground_truth"] == 0].copy()
malicious = df[df["ground_truth"] == 1].copy()

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

print("=" * 70)
print("FALSE POSITIVE BREAKDOWN - FINAL RULE AGENT")
print("=" * 70)

print(f"\nNormal user-days : {len(normal):,}")
print(f"Malicious days   : {len(malicious):,}")

print("\nFalse-positive contribution by rule:")
print("-" * 70)

for rule in rule_columns:

    normal_count = int(normal[rule].sum())
    malicious_count = int(malicious[rule].sum())

    print(
        f"{rule:32s} "
        f"Normal FP: {normal_count:7,} | "
        f"Malicious: {malicious_count:2d}/32"
    )

print("\n" + "=" * 70)
print("NORMAL-DAY RULE COMBINATIONS")
print("=" * 70)

normal["rules_triggered_count"] = normal[rule_columns].sum(axis=1)

print(
    normal["rules_triggered_count"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\n" + "=" * 70)
print("FALSE POSITIVE RULE PAIRS")
print("=" * 70)

pairs = [
    ("rule_high_activity", "rule_unusual_logon"),
    ("rule_high_activity", "rule_unusual_device"),
    ("rule_high_activity", "rule_unusual_file"),
    ("rule_unusual_logon", "rule_unusual_device"),
    ("rule_unusual_logon", "rule_unusual_file"),
    ("rule_unusual_device", "rule_unusual_file"),
    ("rule_after_hours_login", "rule_device_connection_pattern"),
    ("rule_device_connection_pattern", "rule_file_type_activity"),
]

for rule1, rule2 in pairs:

    count = int(
        (normal[rule1] & normal[rule2]).sum()
    )

    print(
        f"{rule1} + {rule2}: {count:,}"
    )

print("\n" + "=" * 70)
print("FALSE POSITIVE DIAGNOSTIC COMPLETED")
print("=" * 70)