# ============================================================
# CERT r4.1 - NETWORK AGENT V2
# ============================================================
#
# Purpose:
#   Network-focused insider-threat detection using:
#   1. Behavioral network features
#   2. Temporal network behavior
#   3. Network persistence
#   4. Personal-baseline deviation
#   5. Multi-signal evidence aggregation
#
# Output:
#   data/outputs/network_agent_v2_results.parquet
#
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

BEHAVIORAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "final_behavioral_features.parquet"
)

# CORRECT TEMPORAL FEATURE FILE
# Created by build_temporal_features.py
TEMPORAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "temporal_behavioral_features.parquet"
)

PERSISTENCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "network_persistence_features.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "network_agent_v2_results.parquet"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_numeric(data, column, default=0.0):
    """
    Convert a column to numeric safely.
    Missing/non-numeric values become the supplied default.
    """
    if column not in data.columns:
        return pd.Series(
            default,
            index=data.index,
            dtype=float
        )

    return pd.to_numeric(
        data[column],
        errors="coerce"
    ).fillna(default)


def safe_bool(data, column):
    """
    Convert a column into a boolean signal safely.
    """
    if column not in data.columns:
        return pd.Series(False, index=data.index)

    return data[column].fillna(False).astype(bool)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("NETWORK AGENT V2")
print("=" * 70)


# ============================================================
# LOAD BEHAVIORAL FEATURES
# ============================================================

print("\nLoading behavioral features...")

if not BEHAVIORAL_FILE.exists():
    raise FileNotFoundError(
        f"\nBehavioral features not found:\n{BEHAVIORAL_FILE}"
    )

behavioral = pd.read_parquet(BEHAVIORAL_FILE)

print(f"Behavioral rows: {len(behavioral):,}")


# ============================================================
# LOAD TEMPORAL FEATURES
# ============================================================

print("\nLoading temporal features...")

if not TEMPORAL_FILE.exists():
    raise FileNotFoundError(
        f"\nTemporal behavioral features not found:\n{TEMPORAL_FILE}"
    )

temporal = pd.read_parquet(TEMPORAL_FILE)

print(f"Temporal rows: {len(temporal):,}")


# ============================================================
# LOAD PERSISTENCE FEATURES
# ============================================================

print("\nLoading persistence features...")

if not PERSISTENCE_FILE.exists():
    raise FileNotFoundError(
        f"\nNetwork persistence features not found:\n{PERSISTENCE_FILE}"
    )

persistence = pd.read_parquet(PERSISTENCE_FILE)

print(f"Persistence rows: {len(persistence):,}")


# ============================================================
# NORMALIZE USER + DAY
# ============================================================

for df in [behavioral, temporal, persistence]:

    if "user" in df.columns:
        df["user"] = (
            df["user"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    if "day" in df.columns:
        df["day"] = pd.to_datetime(
            df["day"],
            errors="coerce"
        ).dt.normalize()


# ============================================================
# REMOVE DUPLICATES
# ============================================================

behavioral = behavioral.drop_duplicates(
    subset=["user", "day"]
)

temporal = temporal.drop_duplicates(
    subset=["user", "day"]
)

persistence = persistence.drop_duplicates(
    subset=["user", "day"]
)


# ============================================================
# MERGE FEATURE GROUPS
# ============================================================

print("\nMerging feature groups...")

data = behavioral.copy()

data = data.merge(
    temporal,
    on=["user", "day"],
    how="left",
    suffixes=("", "_temporal")
)

data = data.merge(
    persistence,
    on=["user", "day"],
    how="left",
    suffixes=("", "_persistence")
)

print(f"Merged rows: {len(data):,}")


# ============================================================
# NETWORK BASE FEATURES
# ============================================================

data["web_visits"] = safe_numeric(
    data,
    "web_visits"
)

data["unique_urls"] = safe_numeric(
    data,
    "unique_urls"
)

data["unique_domains"] = safe_numeric(
    data,
    "unique_domains"
)


# ============================================================
# NETWORK Z-SCORE
# ============================================================

if "web_visits_clean_zscore" in data.columns:

    data["web_visits_clean_zscore"] = safe_numeric(
        data,
        "web_visits_clean_zscore"
    )

else:

    mean_web = data["web_visits"].mean()
    std_web = data["web_visits"].std()

    if std_web > 0:

        data["web_visits_clean_zscore"] = (
            data["web_visits"] - mean_web
        ) / std_web

    else:

        data["web_visits_clean_zscore"] = 0.0


# ============================================================
# TEMPORAL FEATURES
# ============================================================

data["high_web_days_prev7"] = safe_numeric(
    data,
    "high_web_days_prev7"
)

data["high_web_days_prev14"] = safe_numeric(
    data,
    "high_web_days_prev14"
)

data["network_persistence_ratio"] = safe_numeric(
    data,
    "network_persistence_ratio"
)

data["web_visits_prev7_mean"] = safe_numeric(
    data,
    "web_visits_prev7_mean"
)

data["web_visits_prev14_mean"] = safe_numeric(
    data,
    "web_visits_prev14_mean"
)

data["unique_urls_prev7_mean"] = safe_numeric(
    data,
    "unique_urls_prev7_mean"
)

data["unique_domains_prev7_mean"] = safe_numeric(
    data,
    "unique_domains_prev7_mean"
)


# ============================================================
# PERSONAL BASELINE FEATURES
# ============================================================

data["personal_web_ratio"] = safe_numeric(
    data,
    "personal_web_ratio"
)

data["personal_url_ratio"] = safe_numeric(
    data,
    "personal_url_ratio"
)

data["personal_domain_ratio"] = safe_numeric(
    data,
    "personal_domain_ratio"
)


# ============================================================
# SIGNAL CREATION
# ============================================================

print("\nCreating Network Agent V2 signals...")


# ------------------------------------------------------------
# 1. WEB ANOMALY
# ------------------------------------------------------------

data["v2_web_anomaly"] = (
    data["web_visits_clean_zscore"] >= 3
)


data["v2_strong_web_anomaly"] = (
    data["web_visits_clean_zscore"] >= 5
)


# ------------------------------------------------------------
# 2. HIGH WEB ACTIVITY
# ------------------------------------------------------------

web_threshold = data["web_visits"].quantile(0.90)

data["v2_high_web_activity"] = (
    data["web_visits"] >= web_threshold
)


# ------------------------------------------------------------
# 3. NETWORK INTENSITY
# ------------------------------------------------------------

data["v2_network_intensity"] = (
    (data["web_visits"] >= web_threshold)
    &
    (
        data["unique_urls"]
        >= data["unique_urls"].quantile(0.75)
    )
)


# ------------------------------------------------------------
# 4. URL DIVERSITY
# ------------------------------------------------------------

url_threshold = data["unique_urls"].quantile(0.90)

data["v2_url_diversity"] = (
    data["unique_urls"] >= url_threshold
)


# ------------------------------------------------------------
# 5. DOMAIN DIVERSITY
# ------------------------------------------------------------

domain_threshold = data["unique_domains"].quantile(0.90)

data["v2_domain_diversity"] = (
    data["unique_domains"] >= domain_threshold
)


# ------------------------------------------------------------
# 6. PERSISTENT ACTIVITY
# ------------------------------------------------------------

data["v2_persistent_activity"] = (
    data["high_web_days_prev7"] >= 3
)


data["v2_strong_persistent_activity"] = (
    data["high_web_days_prev7"] >= 5
)


data["v2_persistent_activity_14d"] = (
    data["high_web_days_prev14"] >= 5
)


# ------------------------------------------------------------
# 7. CURRENT / PAST AGREEMENT
# ------------------------------------------------------------

data["v2_current_past_agreement"] = (
    data["v2_high_web_activity"]
    &
    (
        data["high_web_days_prev7"] >= 2
    )
)


data["v2_strong_persistence_agreement"] = (
    data["v2_strong_persistent_activity"]
    &
    data["v2_web_anomaly"]
)


# ------------------------------------------------------------
# 8. PERSONAL WEB CHANGE
# ------------------------------------------------------------

data["v2_personal_web_increase"] = (
    data["personal_web_ratio"] >= 2.0
)


data["v2_strong_personal_web_increase"] = (
    data["personal_web_ratio"] >= 3.0
)


# ------------------------------------------------------------
# 9. PERSONAL DIVERSITY CHANGE
# ------------------------------------------------------------

data["v2_personal_diversity_change"] = (
    (
        data["personal_url_ratio"] >= 1.5
    )
    |
    (
        data["personal_domain_ratio"] >= 1.5
    )
)


# ------------------------------------------------------------
# 10. TEMPORAL CHANGE
# ------------------------------------------------------------

data["v2_temporal_change"] = (
    (
        data["web_visits_prev7_mean"] > 0
    )
    &
    (
        data["web_visits"]
        >= data["web_visits_prev7_mean"] * 1.5
    )
)


data["v2_strong_temporal_change"] = (
    (
        data["web_visits_prev7_mean"] > 0
    )
    &
    (
        data["web_visits"]
        >= data["web_visits_prev7_mean"] * 2.0
    )
)


# ------------------------------------------------------------
# 11. BEHAVIOR CHANGE AGREEMENT
# ------------------------------------------------------------

data["v2_behavior_change_agreement"] = (
    data["v2_temporal_change"]
    &
    (
        data["v2_url_diversity"]
        |
        data["v2_domain_diversity"]
    )
)


# ============================================================
# RISK SCORE
# ============================================================

print("\nCalculating V2 network risk score...")

data["network_v2_raw_score"] = 0.0


# Strong anomaly
data.loc[
    data["v2_strong_web_anomaly"],
    "network_v2_raw_score"
] += 5.0


# Regular anomaly
data.loc[
    data["v2_web_anomaly"],
    "network_v2_raw_score"
] += 3.0


# High activity
data.loc[
    data["v2_high_web_activity"],
    "network_v2_raw_score"
] += 1.0


# Network intensity
data.loc[
    data["v2_network_intensity"],
    "network_v2_raw_score"
] += 1.5


# URL diversity
data.loc[
    data["v2_url_diversity"],
    "network_v2_raw_score"
] += 1.0


# Domain diversity
data.loc[
    data["v2_domain_diversity"],
    "network_v2_raw_score"
] += 1.0


# Persistence
data.loc[
    data["v2_persistent_activity"],
    "network_v2_raw_score"
] += 1.5


# Strong persistence
data.loc[
    data["v2_strong_persistent_activity"],
    "network_v2_raw_score"
] += 2.0


# 14-day persistence
data.loc[
    data["v2_persistent_activity_14d"],
    "network_v2_raw_score"
] += 1.5


# Current/past agreement
data.loc[
    data["v2_current_past_agreement"],
    "network_v2_raw_score"
] += 1.5


# Strong persistence agreement
data.loc[
    data["v2_strong_persistence_agreement"],
    "network_v2_raw_score"
] += 2.0


# Personal increase
data.loc[
    data["v2_personal_web_increase"],
    "network_v2_raw_score"
] += 2.0


# Strong personal increase
data.loc[
    data["v2_strong_personal_web_increase"],
    "network_v2_raw_score"
] += 2.5


# Personal diversity
data.loc[
    data["v2_personal_diversity_change"],
    "network_v2_raw_score"
] += 1.0


# Temporal change
data.loc[
    data["v2_temporal_change"],
    "network_v2_raw_score"
] += 1.5


# Strong temporal change
data.loc[
    data["v2_strong_temporal_change"],
    "network_v2_raw_score"
] += 2.0


# Behavioral agreement
data.loc[
    data["v2_behavior_change_agreement"],
    "network_v2_raw_score"
] += 2.0


# ============================================================
# NORMALIZE SCORE TO 0-100
# ============================================================

max_raw_score = data["network_v2_raw_score"].max()

print(
    f"Maximum raw score observed: "
    f"{max_raw_score:.4f}"
)

if max_raw_score > 0:

    data["network_v2_score"] = (
        data["network_v2_raw_score"]
        / max_raw_score
        * 100
    )

else:

    data["network_v2_score"] = 0.0


data["network_v2_score"] = (
    data["network_v2_score"]
    .clip(0, 100)
)


# ============================================================
# EVIDENCE COUNT
# ============================================================

signal_columns = [
    "v2_web_anomaly",
    "v2_strong_web_anomaly",
    "v2_high_web_activity",
    "v2_network_intensity",
    "v2_url_diversity",
    "v2_domain_diversity",
    "v2_persistent_activity",
    "v2_strong_persistent_activity",
    "v2_persistent_activity_14d",
    "v2_current_past_agreement",
    "v2_strong_persistence_agreement",
    "v2_personal_web_increase",
    "v2_strong_personal_web_increase",
    "v2_personal_diversity_change",
    "v2_temporal_change",
    "v2_strong_temporal_change",
    "v2_behavior_change_agreement",
]


data["network_v2_evidence_count"] = (
    data[signal_columns]
    .astype(int)
    .sum(axis=1)
)


# ============================================================
# STATUS
# ============================================================

data["network_v2_status"] = "NO_NETWORK_EVIDENCE"

data.loc[
    (
        data["network_v2_evidence_count"] >= 1
    )
    &
    (
        data["network_v2_evidence_count"] <= 2
    ),
    "network_v2_status"
] = "WEAK_NETWORK_V2_EVIDENCE"


data.loc[
    (
        data["network_v2_evidence_count"] >= 3
    )
    &
    (
        data["network_v2_evidence_count"] <= 5
    ),
    "network_v2_status"
] = "MODERATE_NETWORK_V2_EVIDENCE"


data.loc[
    data["network_v2_evidence_count"] >= 6,
    "network_v2_status"
] = "STRONG_NETWORK_V2_EVIDENCE"


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("NETWORK AGENT V2 SUMMARY")
print("=" * 70)

print(f"Rows: {len(data):,}")

print(
    f"Users: "
    f"{data['user'].nunique():,}"
)

print(
    f"Score mean: "
    f"{data['network_v2_score'].mean():.4f}"
)

print(
    f"Score median: "
    f"{data['network_v2_score'].median():.4f}"
)

print(
    f"Score maximum: "
    f"{data['network_v2_score'].max():.4f}"
)


# ============================================================
# STATUS DISTRIBUTION
# ============================================================

print("\nStatus distribution:")

print(
    data["network_v2_status"]
    .value_counts()
)


# ============================================================
# EVIDENCE DISTRIBUTION
# ============================================================

print("\nEvidence-count distribution:")

print(
    data["network_v2_evidence_count"]
    .value_counts()
    .sort_index()
)


# ============================================================
# SIGNAL COUNTS
# ============================================================

print("\nSignal counts:")

for column in signal_columns:

    count = int(
        data[column].sum()
    )

    print(
        f"{column}: {count:,}"
    )


# ============================================================
# TOP 20
# ============================================================

print("\n" + "=" * 70)
print("TOP 20 NETWORK V2 SCORES")
print("=" * 70)

top_columns = [
    "user",
    "day",
    "web_visits",
    "unique_urls",
    "unique_domains",
    "web_visits_clean_zscore",
    "high_web_days_prev7",
    "network_persistence_ratio",
    "network_v2_score",
    "network_v2_evidence_count",
    "network_v2_status",
]

available_top_columns = [
    column
    for column in top_columns
    if column in data.columns
]

print(
    data[
        available_top_columns
    ]
    .sort_values(
        "network_v2_score",
        ascending=False
    )
    .head(20)
    .to_string(index=False)
)


# ============================================================
# SAVE
# ============================================================

data.to_parquet(
    OUTPUT_FILE,
    index=False
)

print("\nOutput saved to:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("NETWORK AGENT V2 COMPLETE")
print("=" * 70)