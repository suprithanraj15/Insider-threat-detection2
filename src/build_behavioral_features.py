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
    / "user_day_features_labeled.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "behavioral_features.parquet"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("BUILDING BEHAVIORAL FEATURES")
print("=" * 60)

print("\n1. Loading labeled user-day data...")

df = pd.read_parquet(INPUT_FILE)

df["day"] = pd.to_datetime(df["day"])

print(f"Rows: {len(df):,}")
print(f"Users: {df['user'].nunique():,}")


# ============================================================
# BASIC TIME FEATURES
# ============================================================

print("\n2. Creating time-based features...")

df["day_of_week"] = df["day"].dt.dayofweek
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)


# ============================================================
# ACTIVITY INTENSITY
# ============================================================

print("\n3. Creating activity intensity features...")

activity_columns = [
    "logon_events",
    "device_events",
    "file_events",
    "emails_sent",
    "web_visits",
]

df["total_activity"] = df[activity_columns].sum(axis=1)

df["active_channels"] = (
    (df["logon_events"] > 0).astype(int)
    + (df["device_events"] > 0).astype(int)
    + (df["file_events"] > 0).astype(int)
    + (df["emails_sent"] > 0).astype(int)
    + (df["web_visits"] > 0).astype(int)
)


# ============================================================
# DEVICE BEHAVIOR
# ============================================================

print("\n4. Creating device behavior features...")

df["device_activity_ratio"] = (
    df["device_events"] /
    (df["total_activity"] + 1)
)


# ============================================================
# FILE BEHAVIOR
# ============================================================

print("\n5. Creating file behavior features...")

df["file_activity_ratio"] = (
    df["file_events"] /
    (df["total_activity"] + 1)
)

df["file_diversity"] = (
    df["unique_files"] /
    (df["file_events"] + 1)
)


# ============================================================
# EMAIL BEHAVIOR
# ============================================================

print("\n6. Creating email behavior features...")

df["email_activity_ratio"] = (
    df["emails_sent"] /
    (df["total_activity"] + 1)
)

df["avg_email_size"] = (
    df["total_email_size"] /
    (df["emails_sent"] + 1)
)

df["avg_recipients_per_email"] = (
    df["recipient_count"] /
    (df["emails_sent"] + 1)
)


# ============================================================
# WEB BEHAVIOR
# ============================================================

print("\n7. Creating web behavior features...")

df["web_activity_ratio"] = (
    df["web_visits"] /
    (df["total_activity"] + 1)
)

df["url_diversity"] = (
    df["unique_urls"] /
    (df["web_visits"] + 1)
)

df["domain_diversity"] = (
    df["unique_domains"] /
    (df["web_visits"] + 1)
)


# ============================================================
# USER BASELINE FEATURES
# ============================================================

print("\n8. Creating user behavioral baselines...")

baseline_columns = [
    "total_activity",
    "logon_events",
    "device_events",
    "file_events",
    "emails_sent",
    "web_visits",
]

for col in baseline_columns:

    user_mean = df.groupby("user")[col].transform("mean")
    user_std = df.groupby("user")[col].transform("std")

    user_std = user_std.replace(0, np.nan)

    df[f"{col}_zscore"] = (
        (df[col] - user_mean) /
        user_std
    ).fillna(0)


# ============================================================
# COMBINED BEHAVIOR DEVIATION
# ============================================================

print("\n9. Creating overall behavior deviation...")

zscore_columns = [
    f"{col}_zscore"
    for col in baseline_columns
]

df["behavior_deviation"] = (
    df[zscore_columns]
    .abs()
    .mean(axis=1)
)


# ============================================================
# CLEANUP
# ============================================================

print("\n10. Cleaning feature table...")

df = df.replace([np.inf, -np.inf], np.nan)

numeric_columns = df.select_dtypes(include=[np.number]).columns

df[numeric_columns] = df[numeric_columns].fillna(0)

df = df.sort_values(["user", "day"])


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
print("BEHAVIORAL FEATURES CREATED")
print("=" * 60)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Users: {df['user'].nunique():,}")

print(f"\nGround-truth malicious days: {df['ground_truth'].sum():,}")

print(f"\nSaved:")
print(OUTPUT_FILE)

print("\nBehavioral features added:")

new_features = [
    "day_of_week",
    "is_weekend",
    "total_activity",
    "active_channels",
    "device_activity_ratio",
    "file_activity_ratio",
    "file_diversity",
    "email_activity_ratio",
    "avg_email_size",
    "avg_recipients_per_email",
    "web_activity_ratio",
    "url_diversity",
    "domain_diversity",
    "behavior_deviation",
]

for feature in new_features:
    print(f"  - {feature}")