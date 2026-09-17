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

RULE_CONTEXT_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "rule_context_features.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "device_agent_results.parquet"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("CERT r4.1 - DEVICE AGENT")
print("=" * 70)


# ============================================================
# LOAD BEHAVIORAL FEATURES
# ============================================================

print("\nLoading behavioral features...")

behavioral = pd.read_parquet(
    BEHAVIORAL_FILE,
    engine="fastparquet"
)

behavioral["user"] = (
    behavioral["user"]
    .astype(str)
    .str.strip()
    .str.lower()
)

behavioral["day"] = pd.to_datetime(
    behavioral["day"],
    errors="coerce"
).dt.normalize()

print(
    f"Behavioral rows: {len(behavioral):,}"
)


# ============================================================
# LOAD TEMPORAL FEATURES
# ============================================================

print("\nLoading temporal features...")

temporal = pd.read_parquet(
    TEMPORAL_FILE,
    engine="fastparquet"
)

temporal["user"] = (
    temporal["user"]
    .astype(str)
    .str.strip()
    .str.lower()
)

temporal["day"] = pd.to_datetime(
    temporal["day"],
    errors="coerce"
).dt.normalize()

print(
    f"Temporal rows: {len(temporal):,}"
)


# ============================================================
# LOAD RULE CONTEXT
# ============================================================

print("\nLoading rule context...")

context = pd.read_parquet(
    RULE_CONTEXT_FILE,
    engine="fastparquet"
)

context["user"] = (
    context["user"]
    .astype(str)
    .str.strip()
    .str.lower()
)

context["day"] = pd.to_datetime(
    context["day"],
    errors="coerce"
).dt.normalize()

print(
    f"Context rows: {len(context):,}"
)


# ============================================================
# SELECT DEVICE FEATURES
# ============================================================

behavioral_columns = [
    "user",
    "day",
    "device_events",
    "device_pc_count",
    "device_connects",
    "device_disconnects",
    "device_activity_ratio",
    "device_events_clean_zscore",
    "clean_behavior_deviation",
]

temporal_columns = [
    "user",
    "day",
    "device_events_prev7_mean",
    "device_events_change_7d",
    "temporal_change_count",
    "temporal_change_score",
    "persistent_behavior_flag",
]

context_columns = [
    "user",
    "day",
    "connect_events",
    "disconnect_events",
    "connect_disconnect_ratio",
]


behavioral_columns = [
    c for c in behavioral_columns
    if c in behavioral.columns
]

temporal_columns = [
    c for c in temporal_columns
    if c in temporal.columns
]

context_columns = [
    c for c in context_columns
    if c in context.columns
]


behavioral = behavioral[behavioral_columns]
temporal = temporal[temporal_columns]
context = context[context_columns]


# ============================================================
# MERGE
# ============================================================

print("\nMerging device features...")

df = behavioral.merge(
    temporal,
    on=["user", "day"],
    how="left"
)

df = df.merge(
    context,
    on=["user", "day"],
    how="left"
)

print(
    f"Merged rows: {len(df):,}"
)


# ============================================================
# NUMERIC CLEANUP
# ============================================================

numeric_columns = [
    c for c in df.columns
    if c not in ["user", "day"]
]

for col in numeric_columns:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    ).fillna(0)


# ============================================================
# DEVICE SIGNALS
# ============================================================

# 1. Any device activity
df["device_activity_signal"] = (
    df["device_events"] > 0
).astype(int)


# 2. High device activity
df["device_high_volume_signal"] = (
    df["device_events"] >= 4
).astype(int)


# 3. Extreme device activity
df["device_extreme_volume_signal"] = (
    df["device_events"] >= 8
).astype(int)


# 4. Device activity across multiple PCs
df["device_multiple_pc_signal"] = (
    df["device_pc_count"] >= 2
).astype(int)


# 5. Personal behavioral anomaly
df["device_behavioral_anomaly_signal"] = (
    df["device_events_clean_zscore"] >= 2.5
).astype(int)


# 6. Strong personal anomaly
df["device_strong_zscore_signal"] = (
    df["device_events_clean_zscore"] >= 3.0
).astype(int)


# 7. Recent temporal change
df["device_temporal_change_signal"] = (
    df["device_events_change_7d"].abs() >= 0.5
).astype(int)


# 8. Strong temporal change
df["device_strong_temporal_signal"] = (
    df["device_events_change_7d"].abs() >= 1.0
).astype(int)


# 9. Behavioral + temporal agreement
df["device_behavior_change_signal"] = (
    (df["device_behavioral_anomaly_signal"] == 1)
    &
    (df["device_temporal_change_signal"] == 1)
).astype(int)


# 10. Connect/disconnect pairing
df["device_balanced_usage_signal"] = (
    (df["connect_events"] > 0)
    &
    (df["disconnect_events"] > 0)
).astype(int)


# 11. Repeated device interaction
df["device_repeated_interaction_signal"] = (
    df["device_connects"] >= 2
).astype(int)


# ============================================================
# DEVICE EVIDENCE GROUPS
# ============================================================

# Stronger weight for personal behavioral anomalies
df["device_behavioral_evidence_score"] = (
    5.0 * df["device_strong_zscore_signal"]
    +
    3.0 * df["device_behavioral_anomaly_signal"]
)


# Temporal evidence
df["device_temporal_evidence_score"] = (
    3.0 * df["device_strong_temporal_signal"]
    +
    1.0 * df["device_temporal_change_signal"]
)


# Agreement between independent behavioral dimensions
df["device_agreement_score"] = (
    3.0 * df["device_behavior_change_signal"]
)


# Activity intensity
df["device_intensity_score"] = (
    1.0 * df["device_high_volume_signal"]
    +
    2.0 * df["device_extreme_volume_signal"]
)


# Repeated interaction
df["device_interaction_score"] = (
    0.75 * df["device_repeated_interaction_signal"]
)


# Multiple-PC evidence is only supporting evidence
df["device_context_score"] = (
    0.5 * df["device_multiple_pc_signal"]
)


# Connect/disconnect pairing is supporting evidence
df["device_pairing_score"] = (
    0.25 * df["device_balanced_usage_signal"]
)


# ============================================================
# RAW DEVICE SCORE
# ============================================================

df["device_agent_score_raw"] = (
    df["device_behavioral_evidence_score"]
    +
    df["device_temporal_evidence_score"]
    +
    df["device_agreement_score"]
    +
    df["device_intensity_score"]
    +
    df["device_interaction_score"]
    +
    df["device_context_score"]
    +
    df["device_pairing_score"]
)


# Maximum theoretical score
MAX_SCORE = 17.5


df["device_agent_score"] = (
    df["device_agent_score_raw"]
    / MAX_SCORE
    * 100
)


df["device_agent_score"] = (
    df["device_agent_score"]
    .clip(0, 100)
)


# ============================================================
# INDEPENDENT EVIDENCE COUNT
# ============================================================

df["device_evidence_count"] = (
    df["device_behavioral_anomaly_signal"]
    +
    df["device_temporal_change_signal"]
    +
    df["device_high_volume_signal"]
    +
    df["device_extreme_volume_signal"]
    +
    df["device_repeated_interaction_signal"]
    +
    df["device_multiple_pc_signal"]
    +
    df["device_behavior_change_signal"]
)


# ============================================================
# DEVICE STATUS
# ============================================================

df["device_agent_status"] = (
    "NO_DEVICE_EVIDENCE"
)


# ------------------------------------------------------------
# STRONG DEVICE EVIDENCE
#
# Strong status requires genuinely strong evidence.
# ------------------------------------------------------------

strong_condition = (
    (df["device_strong_zscore_signal"] == 1)
    |
    (df["device_behavior_change_signal"] == 1)
    |
    (
        (df["device_extreme_volume_signal"] == 1)
        &
        (df["device_strong_temporal_signal"] == 1)
    )
)


# ------------------------------------------------------------
# MODERATE DEVICE EVIDENCE
# ------------------------------------------------------------

moderate_condition = (
    (
        (df["device_behavioral_anomaly_signal"] == 1)
        &
        (df["device_temporal_change_signal"] == 1)
    )
    |
    (df["device_agent_score"] >= 25)
    |
    (df["device_evidence_count"] >= 3)
)


# ------------------------------------------------------------
# WEAK DEVICE EVIDENCE
# ------------------------------------------------------------

weak_condition = (
    df["device_agent_score"] > 0
)


# ------------------------------------------------------------
# STATUS ASSIGNMENT
#
# Weak → Moderate → Strong
#
# Strong is assigned last so it has priority.
# ------------------------------------------------------------

df.loc[
    weak_condition,
    "device_agent_status"
] = "WEAK_DEVICE_EVIDENCE"


df.loc[
    moderate_condition,
    "device_agent_status"
] = "MODERATE_DEVICE_EVIDENCE"


df.loc[
    strong_condition,
    "device_agent_status"
] = "STRONG_DEVICE_EVIDENCE"


# ============================================================
# SAFETY CONSISTENCY CHECK
# ============================================================

# A row with no score/evidence must never be STRONG.
invalid_strong = (
    (df["device_agent_status"] == "STRONG_DEVICE_EVIDENCE")
    &
    (
        (df["device_agent_score"] <= 0)
        |
        (df["device_evidence_count"] <= 0)
    )
)

invalid_count = int(invalid_strong.sum())

if invalid_count > 0:

    print(
        "\nWARNING:"
        f" {invalid_count:,} STRONG rows have "
        "zero score/evidence."
    )

    # Safety correction
    df.loc[
        invalid_strong,
        "device_agent_status"
    ] = "NO_DEVICE_EVIDENCE"


# ============================================================
# EXPLANATION
# ============================================================

def create_explanation(row):

    evidence = []

    if row["device_strong_zscore_signal"]:
        evidence.append(
            "strong personal device-activity anomaly"
        )

    elif row["device_behavioral_anomaly_signal"]:
        evidence.append(
            "abnormal device activity"
        )

    if row["device_strong_temporal_signal"]:
        evidence.append(
            "strong recent change in device activity"
        )

    elif row["device_temporal_change_signal"]:
        evidence.append(
            "recent change in device activity"
        )

    if row["device_extreme_volume_signal"]:
        evidence.append(
            "very high device activity"
        )

    elif row["device_high_volume_signal"]:
        evidence.append(
            "high device activity"
        )

    if row["device_repeated_interaction_signal"]:
        evidence.append(
            "repeated device interactions"
        )

    if row["device_multiple_pc_signal"]:
        evidence.append(
            "device activity across multiple PCs"
        )

    if row["device_behavior_change_signal"]:
        evidence.append(
            "behavioral and temporal anomaly agreement"
        )

    if not evidence:
        return (
            "No significant device anomaly evidence."
        )

    return (
        "Device evidence: "
        + ", ".join(evidence)
        + "."
    )


df["device_explanation"] = df.apply(
    create_explanation,
    axis=1
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_parquet(
    OUTPUT_FILE,
    engine="fastparquet",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DEVICE AGENT SUMMARY")
print("=" * 70)

print(
    f"Rows processed: {len(df):,}"
)


print("\nStatus distribution:")

print(
    df["device_agent_status"]
    .value_counts()
    .to_string()
)


print("\nEvidence-count distribution:")

print(
    df["device_evidence_count"]
    .value_counts()
    .sort_index()
    .to_string()
)


print("\nSignal counts:")

signal_columns = [
    "device_activity_signal",
    "device_high_volume_signal",
    "device_extreme_volume_signal",
    "device_multiple_pc_signal",
    "device_behavioral_anomaly_signal",
    "device_strong_zscore_signal",
    "device_temporal_change_signal",
    "device_strong_temporal_signal",
    "device_behavior_change_signal",
    "device_balanced_usage_signal",
    "device_repeated_interaction_signal",
]


for signal in signal_columns:

    print(
        f"{signal:<42}"
        f"{int(df[signal].sum()):,}"
    )


print("\nScore statistics:")

print(
    df["device_agent_score"]
    .describe()
    .to_string()
)


print("\nSaved to:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("DEVICE AGENT COMPLETE")
print("=" * 70)