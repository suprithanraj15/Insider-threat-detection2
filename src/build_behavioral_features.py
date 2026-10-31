
from pathlib import Path

import numpy as np
import pandas as pd


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

ENGINE = "fastparquet"

# Minimum historical observations required before calculating
# a meaningful per-user baseline.
MIN_HISTORY = 3


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("BUILDING BEHAVIORAL FEATURES")
print("=" * 60)

print("\n1. Loading labeled user-day data...")

df = pd.read_parquet(INPUT_FILE, engine=ENGINE)
df["day"] = pd.to_datetime(df["day"], errors="raise")

required_columns = [
    "user",
    "day",
    "ground_truth",
    "logon_events",
    "device_events",
    "file_events",
    "emails_sent",
    "web_visits",
    "unique_files",
    "total_email_size",
    "recipient_count",
    "unique_urls",
    "unique_domains",
]

missing = [col for col in required_columns if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

if df[["user", "day"]].isna().any().any():
    raise ValueError("'user' and 'day' must not contain missing values.")

if df.duplicated(["user", "day"]).any():
    raise ValueError("Expected one row per user per day; duplicates found.")

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

df["active_channels"] = sum(
    (df[col] > 0).astype(int) for col in activity_columns
)


# ============================================================
# DEVICE BEHAVIOR
# ============================================================

print("\n4. Creating device behavior features...")

df["device_activity_ratio"] = (
    df["device_events"] / (df["total_activity"] + 1)
)


# ============================================================
# FILE BEHAVIOR
# ============================================================

print("\n5. Creating file behavior features...")

df["file_activity_ratio"] = (
    df["file_events"] / (df["total_activity"] + 1)
)

df["file_diversity"] = (
    df["unique_files"] / (df["file_events"] + 1)
)


# ============================================================
# EMAIL BEHAVIOR
# ============================================================

print("\n6. Creating email behavior features...")

df["email_activity_ratio"] = (
    df["emails_sent"] / (df["total_activity"] + 1)
)

df["avg_email_size"] = (
    df["total_email_size"] / (df["emails_sent"] + 1)
)

df["avg_recipients_per_email"] = (
    df["recipient_count"] / (df["emails_sent"] + 1)
)


# ============================================================
# WEB BEHAVIOR
# ============================================================

print("\n7. Creating web behavior features...")

df["web_activity_ratio"] = (
    df["web_visits"] / (df["total_activity"] + 1)
)

df["url_diversity"] = (
    df["unique_urls"] / (df["web_visits"] + 1)
)

df["domain_diversity"] = (
    df["unique_domains"] / (df["web_visits"] + 1)
)


# ============================================================
# HISTORICAL USER BASELINES
# ============================================================

print("\n8. Creating past-only user behavioral baselines...")

# IMPORTANT:
# Baselines for a given user-day use only that user's earlier
# rows. The current day is excluded with shift(1).
#
# This implementation uses all previous observations, including
# previously labeled malicious days. It does not use future labels
# to decide which historical rows are normal.

df = df.sort_values(["user", "day"]).reset_index(drop=True)

baseline_columns = [
    "total_activity",
    "logon_events",
    "device_events",
    "file_events",
    "emails_sent",
    "web_visits",
]

grouped = df.groupby("user", sort=False)

for col in baseline_columns:
    history = grouped[col].shift(1)

    historical_mean = (
        history.groupby(df["user"], sort=False)
        .expanding(min_periods=MIN_HISTORY)
        .mean()
        .reset_index(level=0, drop=True)
        .sort_index()
    )

    historical_std = (
        history.groupby(df["user"], sort=False)
        .expanding(min_periods=MIN_HISTORY)
        .std()
        .reset_index(level=0, drop=True)
        .sort_index()
    )

    # A zero or unavailable standard deviation cannot provide a
    # useful z-score. Such scores are set to zero.
    valid_std = historical_std.where(historical_std > 0)

    df[f"{col}_zscore"] = (
        (df[col] - historical_mean) / valid_std
    ).replace([np.inf, -np.inf], np.nan).fillna(0)


# ============================================================
# COMBINED BEHAVIOR DEVIATION
# ============================================================

print("\n9. Creating overall behavior deviation...")

zscore_columns = [
    f"{col}_zscore" for col in baseline_columns
]

df["behavior_deviation"] = (
    df[zscore_columns].abs().mean(axis=1)
)


# ============================================================
# CLEANUP
# ============================================================

print("\n10. Cleaning feature table...")

df = df.replace([np.inf, -np.inf], np.nan)

numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = df[numeric_columns].fillna(0)

df = df.sort_values(["user", "day"]).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

df.to_parquet(
    OUTPUT_FILE,
    engine=ENGINE,
    index=False,
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
print(f"Ground-truth malicious days: {df['ground_truth'].sum():,}")

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

print("\nHistorical baseline z-scores:")
for col in baseline_columns:
    print(f"  - {col}_zscore")

print(f"\nSaved: {OUTPUT_FILE}")
