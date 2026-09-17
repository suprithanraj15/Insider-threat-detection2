from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "final_behavioral_features.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "temporal_behavioral_features.parquet"
)


print("=" * 70)
print("BUILDING TEMPORAL BEHAVIORAL FEATURES")
print("=" * 70)


# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------

print("\n1. Loading final behavioral features...")

df = pd.read_parquet(INPUT_FILE)

print(f"Rows: {len(df):,}")
print(f"Users: {df['user'].nunique():,}")


# ------------------------------------------------------------
# 2. Prepare dates and sorting
# ------------------------------------------------------------

print("\n2. Preparing chronological user behavior...")

df["day"] = pd.to_datetime(df["day"])

df = df.sort_values(
    ["user", "day"]
).reset_index(drop=True)


# ------------------------------------------------------------
# 3. Activity columns
# ------------------------------------------------------------

activity_columns = [
    "total_activity",
    "logon_events",
    "device_events",
    "file_events",
    "emails_sent",
    "web_visits"
]


# ------------------------------------------------------------
# 4. Calculate previous 7-day average
# ------------------------------------------------------------

print("\n3. Calculating previous 7-day behavioral averages...")

for col in activity_columns:

    rolling_mean = (
        df.groupby("user")[col]
        .transform(
            lambda x: x.shift(1).rolling(
                window=7,
                min_periods=3
            ).mean()
        )
    )

    df[f"{col}_prev7_mean"] = rolling_mean


# ------------------------------------------------------------
# 5. Calculate change from previous 7-day behavior
# ------------------------------------------------------------

print("\n4. Calculating recent behavioral changes...")

for col in activity_columns:

    baseline = df[f"{col}_prev7_mean"]

    change = (
        (df[col] - baseline)
        / (baseline.abs() + 1)
    )

    df[f"{col}_change_7d"] = change.replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)


# ------------------------------------------------------------
# 6. Count unusually large behavioral changes
# ------------------------------------------------------------

print("\n5. Creating temporal anomaly indicators...")

change_columns = [
    f"{col}_change_7d"
    for col in activity_columns
]

df["temporal_change_count"] = (
    df[change_columns].abs() >= 1.0
).sum(axis=1)


# ------------------------------------------------------------
# 7. Overall temporal change score
# ------------------------------------------------------------

df["temporal_change_score"] = (
    df[change_columns]
    .abs()
    .mean(axis=1)
)


# ------------------------------------------------------------
# 8. Persistence of unusual behavior
# ------------------------------------------------------------

print("\n6. Calculating behavioral persistence...")

df["previous_day_activity"] = (
    df.groupby("user")["total_activity"]
    .shift(1)
)

df["previous_day_behavior_deviation"] = (
    df.groupby("user")["clean_behavior_deviation"]
    .shift(1)
)


# Current day and previous day both show strong deviation
df["persistent_behavior_flag"] = (
    (df["clean_behavior_deviation"] >= 3.0)
    &
    (df["previous_day_behavior_deviation"] >= 3.0)
).astype(int)


# ------------------------------------------------------------
# 9. Clean temporary columns
# ------------------------------------------------------------

df = df.drop(
    columns=[
        "previous_day_activity",
        "previous_day_behavior_deviation"
    ]
)


# ------------------------------------------------------------
# 10. Replace invalid values
# ------------------------------------------------------------

numeric_columns = df.select_dtypes(
    include=[np.number]
).columns

df[numeric_columns] = (
    df[numeric_columns]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)


# ------------------------------------------------------------
# 11. Save
# ------------------------------------------------------------

df.to_parquet(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 70)
print("TEMPORAL FEATURES CREATED")
print("=" * 70)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns):,}")
print(f"Users: {df['user'].nunique():,}")

print("\nNew temporal features:")

for col in df.columns:
    if (
        "_prev7_mean" in col
        or "_change_7d" in col
        or col in [
            "temporal_change_count",
            "temporal_change_score",
            "persistent_behavior_flag"
        ]
    ):
        print(f"  - {col}")

print("\nSaved:")
print(OUTPUT_FILE)