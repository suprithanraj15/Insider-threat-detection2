from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "temporal_behavioral_features.parquet"
)


print("=" * 70)
print("TEMPORAL FEATURE DIAGNOSTIC")
print("=" * 70)

df = pd.read_parquet(INPUT_FILE)

print(f"\nRows: {len(df):,}")
print(f"Users: {df['user'].nunique():,}")


# ------------------------------------------------------------
# Separate ground-truth groups
# ------------------------------------------------------------

malicious = df[df["ground_truth"] == 1].copy()
normal = df[df["ground_truth"] == 0].copy()

print("\nGround truth:")
print(f"  Malicious user-days: {len(malicious):,}")
print(f"  Normal user-days:    {len(normal):,}")


# ------------------------------------------------------------
# Temporal features
# ------------------------------------------------------------

change_columns = [
    "total_activity_change_7d",
    "logon_events_change_7d",
    "device_events_change_7d",
    "file_events_change_7d",
    "emails_sent_change_7d",
    "web_visits_change_7d",
]


# ------------------------------------------------------------
# 1. Compare average temporal changes
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("1. TEMPORAL CHANGE COMPARISON")
print("=" * 70)

comparison = pd.DataFrame({
    "malicious_mean": malicious[change_columns].abs().mean(),
    "normal_mean": normal[change_columns].abs().mean(),
})

print(comparison.to_string())


# ------------------------------------------------------------
# 2. Threshold analysis
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("2. TEMPORAL THRESHOLD ANALYSIS")
print("=" * 70)

thresholds = [0.5, 1.0, 1.5, 2.0, 3.0]

for threshold in thresholds:

    malicious_flag = (
        malicious[change_columns]
        .abs()
        .max(axis=1)
        >= threshold
    )

    normal_flag = (
        normal[change_columns]
        .abs()
        .max(axis=1)
        >= threshold
    )

    print(
        f"\nThreshold {threshold}:"
        f"\n  Malicious detected: "
        f"{malicious_flag.sum()}/{len(malicious)}"
        f"\n  Normal flagged:     "
        f"{normal_flag.sum():,}/{len(normal):,}"
    )


# ------------------------------------------------------------
# 3. Temporal change count
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("3. TEMPORAL CHANGE COUNT")
print("=" * 70)

print("\nMalicious:")
print(
    malicious["temporal_change_count"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nNormal:")
print(
    normal["temporal_change_count"]
    .value_counts()
    .sort_index()
    .head(15)
    .to_string()
)


# ------------------------------------------------------------
# 4. Temporal change score
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("4. TEMPORAL CHANGE SCORE")
print("=" * 70)

print("\nMalicious:")
print(
    malicious["temporal_change_score"]
    .describe()
    .to_string()
)

print("\nNormal:")
print(
    normal["temporal_change_score"]
    .describe()
    .to_string()
)


# ------------------------------------------------------------
# 5. Persistence
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("5. PERSISTENT BEHAVIOR")
print("=" * 70)

print(
    "Malicious days with persistence:",
    int(malicious["persistent_behavior_flag"].sum())
)

print(
    "Normal days with persistence:",
    int(normal["persistent_behavior_flag"].sum())
)


# ------------------------------------------------------------
# 6. Show malicious temporal behavior
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("6. MALICIOUS USER-DAY TEMPORAL DETAILS")
print("=" * 70)

display_columns = [
    "user",
    "day",
    "scenario",
    "temporal_change_count",
    "temporal_change_score",
    "persistent_behavior_flag",
] + change_columns

print(
    malicious[
        display_columns
    ]
    .sort_values(["scenario", "user", "day"])
    .to_string(index=False)
)


print("\n" + "=" * 70)
print("TEMPORAL DIAGNOSTIC COMPLETED")
print("=" * 70)