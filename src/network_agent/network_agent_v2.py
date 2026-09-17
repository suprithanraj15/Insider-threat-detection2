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
# LOAD DATA
# ============================================================

print("=" * 70)
print("NETWORK AGENT V2")
print("=" * 70)

print("\nLoading behavioral features...")

behavioral = pd.read_parquet(
    BEHAVIORAL_FILE,
    engine="fastparquet"
)

print(
    f"Behavioral rows: {len(behavioral):,}"
)


print("\nLoading temporal features...")

temporal = pd.read_parquet(
    TEMPORAL_FILE,
    engine="fastparquet"
)

print(
    f"Temporal rows: {len(temporal):,}"
)


print("\nLoading persistence features...")

persistence = pd.read_parquet(
    PERSISTENCE_FILE,
    engine="fastparquet"
)

print(
    f"Persistence rows: {len(persistence):,}"
)


# ============================================================
# NORMALIZE KEYS
# ============================================================

def normalize_keys(df):

    df = df.copy()

    df["user"] = (
        df["user"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["day"] = pd.to_datetime(
        df["day"],
        errors="coerce"
    ).dt.normalize()

    return df


behavioral = normalize_keys(behavioral)
temporal = normalize_keys(temporal)
persistence = normalize_keys(persistence)


# ============================================================
# SELECT BEHAVIORAL COLUMNS
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
    "clean_behavior_deviation",

    "total_activity_clean_zscore",

    "web_visits_baseline_mean",
    "web_visits_baseline_std"
]


behavioral_columns = [
    col
    for col in behavioral_columns
    if col in behavioral.columns
]


behavioral = behavioral[
    behavioral_columns
].copy()


# ============================================================
# SELECT TEMPORAL COLUMNS
# ============================================================

temporal_columns = [
    "user",
    "day",

    "web_visits_prev7_mean",
    "web_visits_change_7d",

    "temporal_change_count",
    "temporal_change_score",

    "persistent_behavior_flag"
]


temporal_columns = [
    col
    for col in temporal_columns
    if col in temporal.columns
]


temporal = temporal[
    temporal_columns
].copy()


# ============================================================
# SELECT PERSISTENCE COLUMNS
# ============================================================

# IMPORTANT:
# high_web_days_prev7 and high_web_days_prev14 are required
# for the V2 persistence signals.

persistence_columns = [
    "user",
    "day",

    # Current network activity
    "high_web_activity_flag",

    # Network diversity
    "high_url_diversity_flag",
    "high_domain_diversity_flag",

    # Combined network intensity
    "network_intensity_flag",

    # Persistence counts
    "high_web_days_prev7",
    "high_web_days_prev14",

    # Persistence ratios
    "network_persistence_ratio",
    "network_persistence_ratio_14d",

    # Current + previous activity
    "current_high_web_with_past_activity",
    "current_high_web_with_strong_persistence",

    # Personal increase
    "personal_web_increase_flag",
    "strong_personal_web_increase_flag",

    # Personal diversity
    "personal_diversity_increase_flag",

    # Final persistence flag
    "network_persistent_behavior_flag"
]


persistence_columns = [
    col
    for col in persistence_columns
    if col in persistence.columns
]


persistence = persistence[
    persistence_columns
].copy()


# ============================================================
# MERGE
# ============================================================

print("\nMerging feature groups...")

data = behavioral.merge(
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


print(
    f"Merged rows: {len(data):,}"
)


# ============================================================
# NUMERIC CLEANUP
# ============================================================

for col in data.columns:

    if col not in ["user", "day"]:

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )


data = data.replace(
    [np.inf, -np.inf],
    np.nan
)


numeric_columns = [
    col
    for col in data.columns
    if col not in ["user", "day"]
]


data[numeric_columns] = (
    data[numeric_columns]
    .fillna(0)
)


# ============================================================
# V2 SIGNAL CREATION
# ============================================================

print("\nCreating Network Agent V2 signals...")


# ============================================================
# 1. PERSONAL WEB ANOMALY
# ============================================================

data["v2_web_anomaly"] = (
    data["web_visits_clean_zscore"].abs() >= 2.5
).astype(int)


data["v2_strong_web_anomaly"] = (
    data["web_visits_clean_zscore"].abs() >= 3.0
).astype(int)


# ============================================================
# 2. HIGH NETWORK ACTIVITY
# ============================================================

data["v2_high_web_activity"] = (
    data["high_web_activity_flag"] == 1
).astype(int)


# ============================================================
# 3. NETWORK INTENSITY
# ============================================================

data["v2_network_intensity"] = (
    data["network_intensity_flag"] == 1
).astype(int)


# ============================================================
# 4. URL / DOMAIN DIVERSITY
# ============================================================

data["v2_url_diversity"] = (
    data["high_url_diversity_flag"] == 1
).astype(int)


data["v2_domain_diversity"] = (
    data["high_domain_diversity_flag"] == 1
).astype(int)


# ============================================================
# 5. 7-DAY PERSISTENCE
# ============================================================

data["v2_persistent_activity"] = (
    data["high_web_days_prev7"] >= 3
).astype(int)


data["v2_strong_persistent_activity"] = (
    data["high_web_days_prev7"] >= 5
).astype(int)


# ============================================================
# 6. 14-DAY PERSISTENCE
# ============================================================

data["v2_persistent_activity_14d"] = (
    data["high_web_days_prev14"] >= 7
).astype(int)


# ============================================================
# 7. CURRENT + PAST AGREEMENT
# ============================================================

data["v2_current_past_agreement"] = (
    data["current_high_web_with_past_activity"] == 1
).astype(int)


data["v2_strong_persistence_agreement"] = (
    data["current_high_web_with_strong_persistence"] == 1
).astype(int)


# ============================================================
# 8. PERSONAL WEB INCREASE
# ============================================================

data["v2_personal_web_increase"] = (
    data["personal_web_increase_flag"] == 1
).astype(int)


data["v2_strong_personal_web_increase"] = (
    data["strong_personal_web_increase_flag"] == 1
).astype(int)


# ============================================================
# 9. PERSONAL DIVERSITY CHANGE
# ============================================================

data["v2_personal_diversity_change"] = (
    data["personal_diversity_increase_flag"] == 1
).astype(int)


# ============================================================
# 10. TEMPORAL CHANGE
# ============================================================

data["v2_temporal_change"] = (
    data["web_visits_change_7d"].abs() >= 0.5
).astype(int)


data["v2_strong_temporal_change"] = (
    data["web_visits_change_7d"].abs() >= 1.0
).astype(int)


# ============================================================
# 11. BEHAVIOR + TEMPORAL AGREEMENT
# ============================================================

data["v2_behavior_change_agreement"] = (
    (
        data["v2_web_anomaly"] == 1
    )
    &
    (
        data["v2_temporal_change"] == 1
    )
).astype(int)


# ============================================================
# V2 RAW SCORE
# ============================================================

print("\nCalculating V2 network risk score...")


data["network_v2_raw_score"] = 0.0


# ------------------------------------------------------------
# PERSONAL ANOMALY
# ------------------------------------------------------------

data.loc[
    data["v2_web_anomaly"] == 1,
    "network_v2_raw_score"
] += 3.0


data.loc[
    data["v2_strong_web_anomaly"] == 1,
    "network_v2_raw_score"
] += 3.0


# ------------------------------------------------------------
# HIGH NETWORK ACTIVITY
# ------------------------------------------------------------

data.loc[
    data["v2_high_web_activity"] == 1,
    "network_v2_raw_score"
] += 1.0


# ------------------------------------------------------------
# NETWORK INTENSITY
# ------------------------------------------------------------

data.loc[
    data["v2_network_intensity"] == 1,
    "network_v2_raw_score"
] += 2.0


# ------------------------------------------------------------
# NETWORK DIVERSITY
# ------------------------------------------------------------

data.loc[
    data["v2_url_diversity"] == 1,
    "network_v2_raw_score"
] += 0.5


data.loc[
    data["v2_domain_diversity"] == 1,
    "network_v2_raw_score"
] += 0.5


# ------------------------------------------------------------
# 7-DAY PERSISTENCE
# ------------------------------------------------------------

data.loc[
    data["v2_persistent_activity"] == 1,
    "network_v2_raw_score"
] += 2.0


data.loc[
    data["v2_strong_persistent_activity"] == 1,
    "network_v2_raw_score"
] += 2.0


# ------------------------------------------------------------
# 14-DAY PERSISTENCE
# ------------------------------------------------------------

data.loc[
    data["v2_persistent_activity_14d"] == 1,
    "network_v2_raw_score"
] += 1.0


# ------------------------------------------------------------
# CURRENT + PAST AGREEMENT
# ------------------------------------------------------------

data.loc[
    data["v2_current_past_agreement"] == 1,
    "network_v2_raw_score"
] += 1.0


data.loc[
    data["v2_strong_persistence_agreement"] == 1,
    "network_v2_raw_score"
] += 1.0


# ------------------------------------------------------------
# PERSONAL WEB INCREASE
# ------------------------------------------------------------

data.loc[
    data["v2_personal_web_increase"] == 1,
    "network_v2_raw_score"
] += 3.0


data.loc[
    data["v2_strong_personal_web_increase"] == 1,
    "network_v2_raw_score"
] += 2.0


# ------------------------------------------------------------
# PERSONAL DIVERSITY CHANGE
# ------------------------------------------------------------

data.loc[
    data["v2_personal_diversity_change"] == 1,
    "network_v2_raw_score"
] += 0.5


# ------------------------------------------------------------
# TEMPORAL CHANGE
# ------------------------------------------------------------

data.loc[
    data["v2_temporal_change"] == 1,
    "network_v2_raw_score"
] += 1.0


data.loc[
    data["v2_strong_temporal_change"] == 1,
    "network_v2_raw_score"
] += 2.0


# ------------------------------------------------------------
# BEHAVIOR + TEMPORAL AGREEMENT
# ------------------------------------------------------------

data.loc[
    data["v2_behavior_change_agreement"] == 1,
    "network_v2_raw_score"
] += 3.0


# ============================================================
# SCORE NORMALIZATION
# ============================================================

MAX_RAW_SCORE = (
    data["network_v2_raw_score"].max()
)


print(
    f"Maximum raw score observed: "
    f"{MAX_RAW_SCORE:.4f}"
)


if MAX_RAW_SCORE > 0:

    data["network_v2_score"] = (
        data["network_v2_raw_score"]
        /
        MAX_RAW_SCORE
        * 100
    )

else:

    data["network_v2_score"] = 0.0


# ============================================================
# EVIDENCE COUNT
# ============================================================

evidence_signals = [
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
    "v2_behavior_change_agreement"
]


data["network_v2_evidence_count"] = (
    data[evidence_signals]
    .sum(axis=1)
)


# ============================================================
# STATUS
# ============================================================

data["network_v2_status"] = (
    "NO_NETWORK_EVIDENCE"
)


# ============================================================
# STRONG CONDITION
# ============================================================

strong_condition = (
    (
        data["v2_strong_web_anomaly"] == 1
    )
    |
    (
        data["v2_strong_personal_web_increase"] == 1
    )
    |
    (
        data["v2_behavior_change_agreement"] == 1
    )
    |
    (
        (
            data["v2_strong_persistent_activity"] == 1
        )
        &
        (
            data["v2_network_intensity"] == 1
        )
    )
    |
    (
        data["network_v2_score"] >= 60
    )
)


data.loc[
    strong_condition,
    "network_v2_status"
] = "STRONG_NETWORK_V2_EVIDENCE"


# ============================================================
# MODERATE CONDITION
# ============================================================

moderate_condition = (
    (
        data["network_v2_score"] >= 30
    )
    |
    (
        (
            data["v2_persistent_activity"] == 1
        )
        &
        (
            data["v2_high_web_activity"] == 1
        )
    )
    |
    (
        (
            data["v2_network_intensity"] == 1
        )
        &
        (
            data["network_v2_evidence_count"] >= 2
        )
    )
)


data.loc[
    (
        moderate_condition
        &
        ~strong_condition
    ),
    "network_v2_status"
] = "MODERATE_NETWORK_V2_EVIDENCE"


# ============================================================
# WEAK CONDITION
# ============================================================

weak_condition = (
    (
        data["network_v2_score"] > 0
    )
    |
    (
        data["v2_high_web_activity"] == 1
    )
)


data.loc[
    (
        weak_condition
        &
        ~strong_condition
        &
        ~moderate_condition
    ),
    "network_v2_status"
] = "WEAK_NETWORK_V2_EVIDENCE"


# ============================================================
# SAFETY CHECK
# ============================================================

invalid_strong = (
    (
        data["network_v2_status"]
        == "STRONG_NETWORK_V2_EVIDENCE"
    )
    &
    (
        (
            data["network_v2_score"] <= 0
        )
        |
        (
            data["network_v2_evidence_count"] <= 0
        )
    )
)


data.loc[
    invalid_strong,
    "network_v2_status"
] = "NO_NETWORK_EVIDENCE"


# ============================================================
# OUTPUT COLUMNS
# ============================================================

output_columns = [
    "user",
    "day",

    # Raw network activity
    "web_visits",
    "unique_urls",
    "unique_domains",

    # Personal anomaly
    "web_visits_clean_zscore",
    "clean_behavior_deviation",

    # Temporal
    "web_visits_prev7_mean",
    "web_visits_change_7d",

    # Persistence
    "high_web_days_prev7",
    "high_web_days_prev14",
    "network_persistence_ratio",
    "network_persistence_ratio_14d",

    # V2 signals
    *evidence_signals,

    # Final score
    "network_v2_raw_score",
    "network_v2_score",
    "network_v2_evidence_count",
    "network_v2_status"
]


# Remove duplicate columns while preserving order

output_columns = list(
    dict.fromkeys(output_columns)
)


output_columns = [
    col
    for col in output_columns
    if col in data.columns
]


output = data[
    output_columns
].copy()


# ============================================================
# SAVE
# ============================================================

output.to_parquet(
    OUTPUT_FILE,
    engine="fastparquet",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("NETWORK AGENT V2 SUMMARY")
print("=" * 70)

print(
    f"Rows: {len(output):,}"
)

print(
    f"Users: {output['user'].nunique():,}"
)

print(
    f"Score mean: "
    f"{output['network_v2_score'].mean():.4f}"
)

print(
    f"Score median: "
    f"{output['network_v2_score'].median():.4f}"
)

print(
    f"Score maximum: "
    f"{output['network_v2_score'].max():.4f}"
)


# ============================================================
# STATUS DISTRIBUTION
# ============================================================

print("\nStatus distribution:")

print(
    output["network_v2_status"]
    .value_counts()
    .to_string()
)


# ============================================================
# EVIDENCE DISTRIBUTION
# ============================================================

print("\nEvidence-count distribution:")

print(
    output["network_v2_evidence_count"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# SIGNAL COUNTS
# ============================================================

print("\nSignal counts:")

for signal in evidence_signals:

    print(
        f"{signal}: "
        f"{int(output[signal].sum()):,}"
    )


# ============================================================
# TOP 20 SCORES
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
    "network_v2_status"
]


print(
    output[
        top_columns
    ]
    .sort_values(
        "network_v2_score",
        ascending=False
    )
    .head(20)
    .to_string(index=False)
)


# ============================================================
# FINAL
# ============================================================

print("\nOutput saved to:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("NETWORK AGENT V2 COMPLETE")
print("=" * 70)