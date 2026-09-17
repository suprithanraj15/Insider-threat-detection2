from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONTEXT_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "rule_context_features.parquet"
)

GROUND_TRUTH_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "final_behavioral_features.parquet"
)


print("=" * 70)
print("RULE CONTEXT FEATURE DIAGNOSTIC")
print("=" * 70)


# ------------------------------------------------------------
# 1. Load files
# ------------------------------------------------------------

print("\n1. Loading context features...")

context = pd.read_parquet(CONTEXT_FILE)

print(f"Context rows: {len(context):,}")

print("\n2. Loading ground truth...")

truth = pd.read_parquet(GROUND_TRUTH_FILE)

print(f"Ground-truth rows: {len(truth):,}")


# ------------------------------------------------------------
# 2. Keep only ground truth columns
# ------------------------------------------------------------

truth = truth[
    ["user", "day", "ground_truth", "scenario"]
].copy()

truth["day"] = pd.to_datetime(truth["day"])
context["day"] = pd.to_datetime(context["day"])


# ------------------------------------------------------------
# 3. Join
# ------------------------------------------------------------

df = context.merge(
    truth,
    on=["user", "day"],
    how="inner"
)

print(f"\nJoined rows: {len(df):,}")


# ------------------------------------------------------------
# 4. Split
# ------------------------------------------------------------

malicious = df[df["ground_truth"] == 1].copy()
normal = df[df["ground_truth"] == 0].copy()

print("\nGround truth:")
print(f"  Malicious user-days: {len(malicious):,}")
print(f"  Normal user-days:    {len(normal):,}")


# ------------------------------------------------------------
# 5. Context features
# ------------------------------------------------------------

features = [
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


# ------------------------------------------------------------
# 6. Compare malicious vs normal
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("1. MALICIOUS VS NORMAL FEATURE COMPARISON")
print("=" * 70)

comparison = pd.DataFrame({
    "malicious_mean": malicious[features].mean(),
    "normal_mean": normal[features].mean(),
    "malicious_max": malicious[features].max(),
    "normal_max": normal[features].max(),
})

print(comparison.to_string())


# ------------------------------------------------------------
# 7. Check how many malicious days contain each feature
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("2. MALICIOUS-DAY FEATURE OCCURRENCE")
print("=" * 70)

for feature in features:

    count = (
        malicious[feature] > 0
    ).sum()

    print(
        f"{feature:30s}: "
        f"{count:2d}/{len(malicious)}"
    )


# ------------------------------------------------------------
# 8. Check normal-day occurrence
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("3. NORMAL-DAY FEATURE OCCURRENCE")
print("=" * 70)

for feature in features:

    count = (
        normal[feature] > 0
    ).sum()

    print(
        f"{feature:30s}: "
        f"{count:6,}/{len(normal):,}"
    )


# ------------------------------------------------------------
# 9. Scenario-wise analysis
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("4. SCENARIO-WISE CONTEXT ANALYSIS")
print("=" * 70)

for scenario in sorted(
    malicious["scenario"].dropna().unique()
):

    scenario_df = malicious[
        malicious["scenario"] == scenario
    ]

    print(
        f"\nScenario {int(scenario)} "
        f"({len(scenario_df)} malicious days)"
    )

    for feature in features:

        count = (
            scenario_df[feature] > 0
        ).sum()

        if count > 0:

            print(
                f"  {feature:28s}: "
                f"{count}/{len(scenario_df)}"
            )


# ------------------------------------------------------------
# 10. Show malicious days
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("5. MALICIOUS USER-DAY CONTEXT")
print("=" * 70)

display_columns = [
    "user",
    "day",
    "scenario",
] + features

print(
    malicious[
        display_columns
    ]
    .sort_values(["scenario", "user", "day"])
    .to_string(index=False)
)


print("\n" + "=" * 70)
print("CONTEXT DIAGNOSTIC COMPLETED")
print("=" * 70)