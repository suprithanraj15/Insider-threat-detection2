from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "rule_agent_results.parquet"
)


print("=" * 70)
print("RULE AGENT DIAGNOSTIC ANALYSIS")
print("=" * 70)

df = pd.read_parquet(INPUT_FILE)


# ------------------------------------------------------------
# Rule columns
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


# ------------------------------------------------------------
# 1. Analyze malicious user-days
# ------------------------------------------------------------

malicious = df[df["ground_truth"] == 1].copy()

print("\n1. MALICIOUS USER-DAYS")
print("-" * 50)

print(f"Total malicious user-days: {len(malicious)}")

for column in rule_columns:
    print(
        f"{column:30s}: "
        f"{malicious[column].sum():3d} "
        f"({malicious[column].mean() * 100:6.2f}%)"
    )


# ------------------------------------------------------------
# 2. Analyze normal user-days
# ------------------------------------------------------------

normal = df[df["ground_truth"] == 0].copy()

print("\n2. NORMAL USER-DAYS")
print("-" * 50)

print(f"Total normal user-days: {len(normal):,}")

for column in rule_columns:
    print(
        f"{column:30s}: "
        f"{normal[column].sum():5d} "
        f"({normal[column].mean() * 100:6.2f}%)"
    )


# ------------------------------------------------------------
# 3. Scenario-wise rule analysis
# ------------------------------------------------------------

print("\n3. SCENARIO-WISE RULE DETECTION")
print("-" * 50)

for scenario in sorted(malicious["scenario"].dropna().unique()):

    scenario_df = malicious[
        malicious["scenario"] == scenario
    ]

    print(f"\nScenario {int(scenario)}")
    print(
        f"Malicious user-days: "
        f"{len(scenario_df)}"
    )

    for column in rule_columns:

        count = scenario_df[column].sum()

        print(
            f"  {column:28s}: "
            f"{count}/{len(scenario_df)}"
        )


# ------------------------------------------------------------
# 4. Show actual malicious days
# ------------------------------------------------------------

print("\n4. MALICIOUS USER-DAY DETAILS")
print("-" * 70)

display_columns = [
    "user",
    "day",
    "scenario",
    "rule_score",
    "rule_alert_count"
] + rule_columns

print(
    malicious[display_columns]
    .sort_values(["scenario", "user", "day"])
    .to_string(index=False)
)


# ------------------------------------------------------------
# 5. False-positive rule combinations
# ------------------------------------------------------------

print("\n5. FALSE-POSITIVE ANALYSIS")
print("-" * 50)

false_positives = df[
    (df["ground_truth"] == 0) &
    (df["rule_triggered"] == 1)
].copy()

print(
    f"Total false-positive user-days: "
    f"{len(false_positives):,}"
)

for count in range(1, 8):

    number = (
        false_positives["rule_alert_count"] == count
    ).sum()

    print(
        f"False positives with "
        f"{count} rule(s): {number:,}"
    )


# ------------------------------------------------------------
# 6. Malicious rule combinations
# ------------------------------------------------------------

print("\n6. MALICIOUS RULE COMBINATIONS")
print("-" * 50)

for count in range(1, 8):

    number = (
        malicious["rule_alert_count"] == count
    ).sum()

    print(
        f"Malicious days with "
        f"{count} rule(s): {number}"
    )


print("\n" + "=" * 70)
print("DIAGNOSTIC ANALYSIS COMPLETED")
print("=" * 70)