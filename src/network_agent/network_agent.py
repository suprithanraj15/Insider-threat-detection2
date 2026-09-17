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

TEMPORAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "temporal_behavioral_features.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "network_agent_results.parquet"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("CERT r4.1 - NETWORK AGENT V1")
print("=" * 70)

print("\nLoading behavioral features...")

behavioral = pd.read_parquet(
    BEHAVIORAL_FILE,
    engine="fastparquet"
)

print(f"Behavioral rows: {len(behavioral):,}")


print("\nLoading temporal features...")

temporal = pd.read_parquet(
    TEMPORAL_FILE,
    engine="fastparquet"
)

print(f"Temporal rows: {len(temporal):,}")


# ============================================================
# NORMALIZE KEYS
# ============================================================

for df in [behavioral, temporal]:

    df["user"] = (
        df["user"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["day"] = pd.to_datetime(
        df["day"],
        errors="coerce"
    ).dt.strftime("%Y-%m-%d")


# ============================================================
# SELECT FEATURES
# ============================================================

behavioral_columns = [
    "user",
    "day",
    "web_visits",
    "unique_urls",
    "unique_domains",
    "web_activity_ratio",
    "url_diversity",
    "domain_diversity",
    "web_visits_clean_zscore",
    "clean_behavior_deviation"
]

temporal_columns = [
    "user",
    "day",
    "web_visits_prev7_mean",
    "web_visits_change_7d",
    "temporal_change_count",
    "temporal_change_score",
    "persistent_behavior_flag"
]


# ============================================================
# CHECK COLUMNS
# ============================================================

for col in behavioral_columns:

    if col not in behavioral.columns:
        raise ValueError(
            f"Missing behavioral column: {col}"
        )


for col in temporal_columns:

    if col not in temporal.columns:
        raise ValueError(
            f"Missing temporal column: {col}"
        )


# ============================================================
# MERGE
# ============================================================

print("\nMerging network features...")

data = behavioral[behavioral_columns].merge(
    temporal[temporal_columns],
    on=["user", "day"],
    how="inner"
)

print(f"Merged rows: {len(data):,}")


# ============================================================
# CLEAN NUMERIC VALUES
# ============================================================

numeric_columns = [
    "web_visits",
    "unique_urls",
    "unique_domains",
    "web_activity_ratio",
    "url_diversity",
    "domain_diversity",
    "web_visits_clean_zscore",
    "clean_behavior_deviation",
    "web_visits_prev7_mean",
    "web_visits_change_7d",
    "temporal_change_count",
    "temporal_change_score",
    "persistent_behavior_flag"
]

for col in numeric_columns:

    data[col] = pd.to_numeric(
        data[col],
        errors="coerce"
    ).fillna(0)


# ============================================================
# NETWORK SIGNALS
# ============================================================

print("\nCreating network signals...")


# ------------------------------------------------------------
# 1. PERSONAL WEB ACTIVITY ANOMALY
# ------------------------------------------------------------

data["network_web_anomaly_signal"] = (
    data["web_visits_clean_zscore"].abs() >= 2.5
).astype(int)


# ------------------------------------------------------------
# 2. STRONG PERSONAL WEB ANOMALY
# ------------------------------------------------------------

data["network_strong_web_anomaly_signal"] = (
    data["web_visits_clean_zscore"].abs() >= 3.0
).astype(int)


# ------------------------------------------------------------
# 3. HIGH WEB ACTIVITY
# ------------------------------------------------------------

data["network_high_web_activity_signal"] = (
    data["web_visits"] >= 150
).astype(int)


# ------------------------------------------------------------
# 4. HIGH URL DIVERSITY
# ------------------------------------------------------------

data["network_high_url_diversity_signal"] = (
    data["unique_urls"] >= 50
).astype(int)


# ------------------------------------------------------------
# 5. HIGH DOMAIN DIVERSITY
# ------------------------------------------------------------

data["network_high_domain_diversity_signal"] = (
    data["unique_domains"] >= 50
).astype(int)


# ------------------------------------------------------------
# 6. NETWORK INTENSITY
# ------------------------------------------------------------

data["network_intensity_signal"] = (
    (
        data["web_visits"] >= 150
    )
    &
    (
        (
            data["unique_urls"] >= 50
        )
        |
        (
            data["unique_domains"] >= 50
        )
    )
).astype(int)


# ------------------------------------------------------------
# 7. TEMPORAL CHANGE
# ------------------------------------------------------------

data["network_temporal_change_signal"] = (
    data["web_visits_change_7d"].abs() >= 0.50
).astype(int)


# ------------------------------------------------------------
# 8. STRONG TEMPORAL CHANGE
# ------------------------------------------------------------

data["network_strong_temporal_signal"] = (
    data["web_visits_change_7d"].abs() >= 1.0
).astype(int)


# ------------------------------------------------------------
# 9. BEHAVIOR + TEMPORAL AGREEMENT
# ------------------------------------------------------------

data["network_behavior_change_signal"] = (
    (
        data["web_visits_clean_zscore"].abs() >= 2.5
    )
    &
    (
        data["web_visits_change_7d"].abs() >= 0.50
    )
).astype(int)


# ------------------------------------------------------------
# 10. TEMPORAL SCORE SIGNAL
# ------------------------------------------------------------

data["network_temporal_score_signal"] = (
    data["temporal_change_score"] >= 0.75
).astype(int)


# ------------------------------------------------------------
# 11. PERSISTENCE
# ------------------------------------------------------------

data["network_persistence_signal"] = (
    data["persistent_behavior_flag"] == 1
).astype(int)


# ============================================================
# SCORE COMPONENTS
# ============================================================

print("\nCalculating network anomaly score...")


# Personal behavioral evidence
data["network_behavior_score"] = (
    data["network_strong_web_anomaly_signal"] * 5.0
    +
    (
        (
            data["network_web_anomaly_signal"] == 1
        )
        &
        (
            data["network_strong_web_anomaly_signal"] == 0
        )
    ).astype(int) * 3.0
)


# Temporal evidence
data["network_temporal_score_component"] = (
    data["network_strong_temporal_signal"] * 3.0
    +
    (
        (
            data["network_temporal_change_signal"] == 1
        )
        &
        (
            data["network_strong_temporal_signal"] == 0
        )
    ).astype(int) * 1.0
)


# Agreement between baseline and temporal change
data["network_agreement_score"] = (
    data["network_behavior_change_signal"] * 3.0
)


# Raw network intensity
data["network_activity_score"] = (
    data["network_high_web_activity_signal"] * 1.0
    +
    data["network_high_url_diversity_signal"] * 0.75
    +
    data["network_high_domain_diversity_signal"] * 0.75
    +
    data["network_intensity_signal"] * 1.5
)


# Temporal score
data["network_temporal_score_bonus"] = (
    data["network_temporal_score_signal"] * 1.5
)


# Persistence
data["network_persistence_score"] = (
    data["network_persistence_signal"] * 1.0
)


# ============================================================
# TOTAL RAW SCORE
# ============================================================

data["network_raw_score"] = (
    data["network_behavior_score"]
    +
    data["network_temporal_score_component"]
    +
    data["network_agreement_score"]
    +
    data["network_activity_score"]
    +
    data["network_temporal_score_bonus"]
    +
    data["network_persistence_score"]
)


# Maximum theoretical score
MAX_SCORE = 20.0


# Normalize to 0-100
data["network_score"] = (
    data["network_raw_score"]
    / MAX_SCORE
    * 100
)

data["network_score"] = data["network_score"].clip(
    lower=0,
    upper=100
)


# ============================================================
# EVIDENCE COUNT
# ============================================================

evidence_columns = [
    "network_web_anomaly_signal",
    "network_high_web_activity_signal",
    "network_high_url_diversity_signal",
    "network_high_domain_diversity_signal",
    "network_intensity_signal",
    "network_temporal_change_signal",
    "network_strong_temporal_signal",
    "network_behavior_change_signal",
    "network_temporal_score_signal",
    "network_persistence_signal"
]

data["network_evidence_count"] = (
    data[evidence_columns]
    .sum(axis=1)
)


# ============================================================
# STATUS
# ============================================================

data["network_status"] = "NO_NETWORK_EVIDENCE"


strong_condition = (
    (
        data["network_strong_web_anomaly_signal"] == 1
    )
    |
    (
        data["network_behavior_change_signal"] == 1
    )
    |
    (
        (
            data["network_intensity_signal"] == 1
        )
        &
        (
            data["network_strong_temporal_signal"] == 1
        )
    )
    |
    (
        data["network_score"] >= 60
    )
)


moderate_condition = (
    (
        data["network_web_anomaly_signal"] == 1
    )
    |
    (
        data["network_temporal_change_signal"] == 1
    )
    |
    (
        data["network_intensity_signal"] == 1
    )
    |
    (
        data["network_evidence_count"] >= 3
    )
    |
    (
        data["network_score"] >= 30
    )
)


weak_condition = (
    (
        data["network_evidence_count"] >= 1
    )
    |
    (
        data["network_high_web_activity_signal"] == 1
    )
)


data.loc[
    weak_condition,
    "network_status"
] = "WEAK_NETWORK_EVIDENCE"


data.loc[
    moderate_condition,
    "network_status"
] = "MODERATE_NETWORK_EVIDENCE"


data.loc[
    strong_condition,
    "network_status"
] = "STRONG_NETWORK_EVIDENCE"


# ============================================================
# SAFETY CONSISTENCY CHECK
# ============================================================

invalid_strong = (
    (
        data["network_status"]
        == "STRONG_NETWORK_EVIDENCE"
    )
    &
    (
        (
            data["network_score"] <= 0
        )
        |
        (
            data["network_evidence_count"] <= 0
        )
    )
)

data.loc[
    invalid_strong,
    "network_status"
] = "NO_NETWORK_EVIDENCE"


# ============================================================
# OUTPUT COLUMNS
# ============================================================

output_columns = [
    "user",
    "day",

    "web_visits",
    "unique_urls",
    "unique_domains",

    "web_activity_ratio",
    "url_diversity",
    "domain_diversity",

    "web_visits_clean_zscore",
    "clean_behavior_deviation",

    "web_visits_prev7_mean",
    "web_visits_change_7d",
    "temporal_change_count",
    "temporal_change_score",
    "persistent_behavior_flag",

    "network_web_anomaly_signal",
    "network_strong_web_anomaly_signal",
    "network_high_web_activity_signal",
    "network_high_url_diversity_signal",
    "network_high_domain_diversity_signal",
    "network_intensity_signal",
    "network_temporal_change_signal",
    "network_strong_temporal_signal",
    "network_behavior_change_signal",
    "network_temporal_score_signal",
    "network_persistence_signal",

    "network_raw_score",
    "network_score",
    "network_evidence_count",
    "network_status"
]


data[output_columns].to_parquet(
    OUTPUT_FILE,
    engine="fastparquet",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("NETWORK AGENT SUMMARY")
print("=" * 70)

print(
    f"Total rows: {len(data):,}"
)

print("\nStatus distribution:")

print(
    data["network_status"]
    .value_counts()
    .to_string()
)


print("\nEvidence count distribution:")

print(
    data["network_evidence_count"]
    .value_counts()
    .sort_index()
    .to_string()
)


print("\nSignal counts:")

for col in evidence_columns:

    print(
        f"{col}: "
        f"{int(data[col].sum()):,}"
    )


print("\nScore statistics:")

print(
    f"Mean   : {data['network_score'].mean():.4f}"
)

print(
    f"Median : {data['network_score'].median():.4f}"
)

print(
    f"Max    : {data['network_score'].max():.4f}"
)


print("\nOutput saved to:")

print(OUTPUT_FILE)


print("\n" + "=" * 70)
print("NETWORK AGENT V1 COMPLETE")
print("=" * 70)