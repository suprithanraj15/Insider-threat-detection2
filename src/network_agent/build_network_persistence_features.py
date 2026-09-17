from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

HTTP_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "http_daily_features.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "network_persistence_features.parquet"
)


# ============================================================
# CONFIGURATION
# ============================================================

HIGH_WEB_PERCENTILE = 0.90
HIGH_URL_PERCENTILE = 0.90
HIGH_DOMAIN_PERCENTILE = 0.90


# ============================================================
# LOAD HTTP DAILY FEATURES
# ============================================================

print("=" * 70)
print("CERT r4.1 - NETWORK PERSISTENCE FEATURE ENGINEERING")
print("=" * 70)

print("\nLoading HTTP daily features...")

http = pd.read_parquet(
    HTTP_FILE,
    engine="fastparquet"
)

print(f"HTTP rows: {len(http):,}")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "user",
    "day",
    "web_visits",
    "unique_urls",
    "unique_domains"
]

missing = [
    col
    for col in required_columns
    if col not in http.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# NORMALIZE USER
# ============================================================

http["user"] = (
    http["user"]
    .astype(str)
    .str.strip()
    .str.lower()
)


# ============================================================
# NORMALIZE DATE
# ============================================================

print("\nNormalizing dates...")

# HTTP daily features store day as nanosecond timestamp
http["day"] = pd.to_datetime(
    pd.to_numeric(
        http["day"],
        errors="coerce"
    ),
    unit="ns",
    errors="coerce"
).dt.normalize()


invalid_dates = http["day"].isna().sum()

print(
    f"Invalid dates: {invalid_dates:,}"
)


# ============================================================
# SORT BY USER AND DATE
# ============================================================

http = (
    http
    .sort_values(
        ["user", "day"]
    )
    .reset_index(drop=True)
)


# ============================================================
# NUMERIC CONVERSION
# ============================================================

numeric_columns = [
    "web_visits",
    "unique_urls",
    "unique_domains"
]

for col in numeric_columns:

    http[col] = pd.to_numeric(
        http[col],
        errors="coerce"
    ).fillna(0)


# ============================================================
# PERSONAL BASELINE FEATURES
# ============================================================

print("\nCreating personal network baselines...")

# Previous 7-day mean.
# shift(1) ensures today's activity is NOT included.
http["web_visits_prev7_mean"] = (
    http
    .groupby("user")["web_visits"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            window=7,
            min_periods=1
        )
        .mean()
    )
)

http["unique_urls_prev7_mean"] = (
    http
    .groupby("user")["unique_urls"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            window=7,
            min_periods=1
        )
        .mean()
    )
)

http["unique_domains_prev7_mean"] = (
    http
    .groupby("user")["unique_domains"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            window=7,
            min_periods=1
        )
        .mean()
    )
)


# ============================================================
# PERSONAL BASELINE RATIOS
# ============================================================

print("\nCreating baseline ratios...")

EPSILON = 1e-6

http["web_activity_ratio_to_baseline"] = (
    http["web_visits"]
    /
    (
        http["web_visits_prev7_mean"]
        + EPSILON
    )
)

http["url_diversity_ratio_to_baseline"] = (
    http["unique_urls"]
    /
    (
        http["unique_urls_prev7_mean"]
        + EPSILON
    )
)

http["domain_diversity_ratio_to_baseline"] = (
    http["unique_domains"]
    /
    (
        http["unique_domains_prev7_mean"]
        + EPSILON
    )
)


# ============================================================
# RAW CHANGE FROM PERSONAL BASELINE
# ============================================================

http["web_visits_absolute_change"] = (
    http["web_visits"]
    -
    http["web_visits_prev7_mean"]
)

http["unique_urls_absolute_change"] = (
    http["unique_urls"]
    -
    http["unique_urls_prev7_mean"]
)

http["unique_domains_absolute_change"] = (
    http["unique_domains"]
    -
    http["unique_domains_prev7_mean"]
)


# ============================================================
# GLOBAL NETWORK PERCENTILE THRESHOLDS
# ============================================================

print("\nCalculating global network percentile thresholds...")

web_threshold = http["web_visits"].quantile(
    HIGH_WEB_PERCENTILE
)

url_threshold = http["unique_urls"].quantile(
    HIGH_URL_PERCENTILE
)

domain_threshold = http["unique_domains"].quantile(
    HIGH_DOMAIN_PERCENTILE
)

print(
    f"Web visits 90th percentile    : "
    f"{web_threshold:.4f}"
)

print(
    f"Unique URLs 90th percentile   : "
    f"{url_threshold:.4f}"
)

print(
    f"Unique domains 90th percentile: "
    f"{domain_threshold:.4f}"
)


# ============================================================
# GLOBAL HIGH-ACTIVITY FLAGS
# ============================================================

http["high_web_activity_flag"] = (
    http["web_visits"] >= web_threshold
).astype(int)

http["high_url_diversity_flag"] = (
    http["unique_urls"] >= url_threshold
).astype(int)

http["high_domain_diversity_flag"] = (
    http["unique_domains"] >= domain_threshold
).astype(int)


# ============================================================
# NETWORK INTENSITY FLAG
# ============================================================

http["network_intensity_flag"] = (
    (
        http["high_web_activity_flag"] == 1
    )
    &
    (
        (
            http["high_url_diversity_flag"] == 1
        )
        |
        (
            http["high_domain_diversity_flag"] == 1
        )
    )
).astype(int)


# ============================================================
# PREVIOUS 7-DAY HIGH-ACTIVITY COUNT
# ============================================================

print("\nCreating 7-day persistence features...")

http["high_web_days_prev7"] = (
    http
    .groupby("user")["high_web_activity_flag"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            window=7,
            min_periods=1
        )
        .sum()
    )
)

http["high_url_days_prev7"] = (
    http
    .groupby("user")["high_url_diversity_flag"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            window=7,
            min_periods=1
        )
        .sum()
    )
)

http["high_domain_days_prev7"] = (
    http
    .groupby("user")["high_domain_diversity_flag"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            window=7,
            min_periods=1
        )
        .sum()
    )
)


# ============================================================
# PREVIOUS 14-DAY HIGH-ACTIVITY COUNT
# ============================================================

http["high_web_days_prev14"] = (
    http
    .groupby("user")["high_web_activity_flag"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            window=14,
            min_periods=1
        )
        .sum()
    )
)


# ============================================================
# PERSISTENCE RATIOS
# ============================================================

http["network_persistence_ratio"] = (
    http["high_web_days_prev7"]
    / 7.0
)

http["network_persistence_ratio_14d"] = (
    http["high_web_days_prev14"]
    / 14.0
)


# ============================================================
# CURRENT + PREVIOUS ACTIVITY PERSISTENCE
# ============================================================

http["current_high_web_with_past_activity"] = (
    (
        http["high_web_activity_flag"] == 1
    )
    &
    (
        http["high_web_days_prev7"] >= 1
    )
).astype(int)


http["current_high_web_with_strong_persistence"] = (
    (
        http["high_web_activity_flag"] == 1
    )
    &
    (
        http["high_web_days_prev7"] >= 3
    )
).astype(int)


# ============================================================
# NETWORK HIGH-ACTIVITY STREAK
# ============================================================

print("\nCalculating current high-activity streak...")

def calculate_streak(values):

    streaks = []
    current = 0

    for value in values:

        if value == 1:
            current += 1
        else:
            current = 0

        streaks.append(current)

    return pd.Series(
        streaks,
        index=values.index
    )


http["network_high_activity_streak"] = (
    http
    .groupby("user")["high_web_activity_flag"]
    .transform(calculate_streak)
)


# ============================================================
# DIVERSITY RELATIONSHIP FEATURES
# ============================================================

http["url_domain_ratio"] = (
    http["unique_urls"]
    /
    (
        http["unique_domains"]
        + EPSILON
    )
)


http["network_diversity_signal"] = (
    (
        http["unique_urls"] >= url_threshold
    )
    &
    (
        http["unique_domains"] >= domain_threshold
    )
).astype(int)


# ============================================================
# PERSONAL BASELINE AGREEMENT
# ============================================================

http["personal_web_increase_flag"] = (
    http["web_activity_ratio_to_baseline"] >= 1.25
).astype(int)

http["strong_personal_web_increase_flag"] = (
    http["web_activity_ratio_to_baseline"] >= 1.50
).astype(int)


http["personal_diversity_increase_flag"] = (
    (
        http["url_diversity_ratio_to_baseline"] >= 1.25
    )
    |
    (
        http["domain_diversity_ratio_to_baseline"] >= 1.25
    )
).astype(int)


# ============================================================
# COMBINED PERSISTENCE SIGNAL
# ============================================================

http["network_persistent_behavior_flag"] = (
    (
        http["high_web_days_prev7"] >= 3
    )
    &
    (
        http["high_web_activity_flag"] == 1
    )
).astype(int)


# ============================================================
# CLEAN INF / NAN
# ============================================================

http = http.replace(
    [np.inf, -np.inf],
    np.nan
)

numeric_output_columns = [
    col
    for col in http.columns
    if col not in ["user", "day"]
]

http[numeric_output_columns] = (
    http[numeric_output_columns]
    .fillna(0)
)


# ============================================================
# SELECT OUTPUT COLUMNS
# ============================================================

output_columns = [
    "user",
    "day",

    # Raw network activity
    "web_visits",
    "unique_urls",
    "unique_domains",

    # Personal baselines
    "web_visits_prev7_mean",
    "unique_urls_prev7_mean",
    "unique_domains_prev7_mean",

    # Baseline ratios
    "web_activity_ratio_to_baseline",
    "url_diversity_ratio_to_baseline",
    "domain_diversity_ratio_to_baseline",

    # Absolute changes
    "web_visits_absolute_change",
    "unique_urls_absolute_change",
    "unique_domains_absolute_change",

    # Global activity
    "high_web_activity_flag",
    "high_url_diversity_flag",
    "high_domain_diversity_flag",
    "network_intensity_flag",

    # Persistence
    "high_web_days_prev7",
    "high_url_days_prev7",
    "high_domain_days_prev7",
    "high_web_days_prev14",
    "network_persistence_ratio",
    "network_persistence_ratio_14d",

    # Persistence combinations
    "current_high_web_with_past_activity",
    "current_high_web_with_strong_persistence",
    "network_high_activity_streak",

    # Diversity
    "url_domain_ratio",
    "network_diversity_signal",

    # Personal increase
    "personal_web_increase_flag",
    "strong_personal_web_increase_flag",
    "personal_diversity_increase_flag",

    # Final persistence indicator
    "network_persistent_behavior_flag"
]


# ============================================================
# SAVE
# ============================================================

output = http[output_columns].copy()

output.to_parquet(
    OUTPUT_FILE,
    engine="fastparquet",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("NETWORK PERSISTENCE FEATURE SUMMARY")
print("=" * 70)

print(
    f"Rows generated: {len(output):,}"
)

print(
    f"Users: {output['user'].nunique():,}"
)

print(
    f"Date range: "
    f"{output['day'].min().date()} "
    f"to "
    f"{output['day'].max().date()}"
)


print("\nPersistence feature statistics:")

for col in [
    "high_web_days_prev7",
    "high_web_days_prev14",
    "network_persistence_ratio",
    "network_persistence_ratio_14d",
    "network_high_activity_streak"
]:

    print(
        f"{col}: "
        f"mean={output[col].mean():.4f}, "
        f"median={output[col].median():.4f}, "
        f"max={output[col].max():.4f}"
    )


print("\nSignal counts:")

for col in [
    "high_web_activity_flag",
    "high_url_diversity_flag",
    "high_domain_diversity_flag",
    "network_intensity_flag",
    "current_high_web_with_past_activity",
    "current_high_web_with_strong_persistence",
    "network_diversity_signal",
    "personal_web_increase_flag",
    "strong_personal_web_increase_flag",
    "personal_diversity_increase_flag",
    "network_persistent_behavior_flag"
]:

    print(
        f"{col}: "
        f"{int(output[col].sum()):,}"
    )


print("\nOutput saved to:")
print(OUTPUT_FILE)


print("\n" + "=" * 70)
print("NETWORK PERSISTENCE FEATURE ENGINEERING COMPLETE")
print("=" * 70)