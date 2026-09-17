from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "behavioral_features.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "final_behavioral_features.parquet"
)


# ============================================================
# LOAD
# ============================================================

print("=" * 60)
print("BUILDING CLEAN USER BEHAVIOR BASELINES")
print("=" * 60)

print("\n1. Loading behavioral features...")

df = pd.read_parquet(INPUT_FILE)

print(f"Rows: {len(df):,}")
print(f"Users: {df['user'].nunique():,}")


# ============================================================
# NORMAL BASELINE
# ============================================================

print("\n2. Creating baselines from normal behavior only...")

normal_df = df[df["ground_truth"] == 0].copy()

print(f"Normal user-days used for baseline: {len(normal_df):,}")


baseline_columns = [
    "total_activity",
    "logon_events",
    "device_events",
    "file_events",
    "emails_sent",
    "web_visits",
]


# ============================================================
# USER BASELINE STATISTICS
# ============================================================

print("\n3. Calculating per-user mean and standard deviation...")

for col in baseline_columns:

    stats = (
        normal_df
        .groupby("user")[col]
        .agg(["mean", "std"])
        .rename(
            columns={
                "mean": f"{col}_baseline_mean",
                "std": f"{col}_baseline_std",
            }
        )
    )

    df = df.join(stats, on="user")

    df[f"{col}_baseline_std"] = (
        df[f"{col}_baseline_std"]
        .replace(0, np.nan)
        .fillna(1)
    )


# ============================================================
# CLEAN BASELINE Z-SCORES
# ============================================================

print("\n4. Creating clean behavioral deviation scores...")

for col in baseline_columns:

    df[f"{col}_clean_zscore"] = (
        (
            df[col]
            - df[f"{col}_baseline_mean"]
        )
        /
        df[f"{col}_baseline_std"]
    )


# ============================================================
# OVERALL DEVIATION
# ============================================================

print("\n5. Creating overall behavioral deviation score...")

clean_zscores = [
    f"{col}_clean_zscore"
    for col in baseline_columns
]

df["clean_behavior_deviation"] = (
    df[clean_zscores]
    .abs()
    .mean(axis=1)
)


# ============================================================
# CLEAN NUMERIC VALUES
# ============================================================

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

numeric_columns = df.select_dtypes(
    include=[np.number]
).columns

df[numeric_columns] = (
    df[numeric_columns]
    .fillna(0)
)


# ============================================================
# SORT
# ============================================================

df = df.sort_values(
    ["user", "day"]
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

print("\n" + "=" * 60)
print("CLEAN BASELINE FEATURES CREATED")
print("=" * 60)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Users: {df['user'].nunique():,}")

print(
    f"Ground-truth malicious days: "
    f"{df['ground_truth'].sum():,}"
)

print(
    f"Normal days used for baseline: "
    f"{len(normal_df):,}"
)

print("\nNew key feature:")
print("  - clean_behavior_deviation")

print("\nSaved:")
print(OUTPUT_FILE)