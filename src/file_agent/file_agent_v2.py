from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"

BEHAVIORAL_FILE = OUTPUT_DIR / "final_behavioral_features.parquet"
TEMPORAL_FILE = OUTPUT_DIR / "temporal_behavioral_features.parquet"
CONTEXT_FILE = OUTPUT_DIR / "rule_context_features.parquet"


print("=" * 80)
print("FILE AGENT V2")
print("=" * 80)


# ================================================================
# 1. LOAD DATA
# ================================================================

behavioral = pd.read_parquet(
    BEHAVIORAL_FILE,
    engine="fastparquet"
)

temporal = pd.read_parquet(
    TEMPORAL_FILE,
    engine="fastparquet"
)

context = pd.read_parquet(
    CONTEXT_FILE,
    engine="fastparquet"
)


# ================================================================
# 2. NORMALIZE
# ================================================================

def normalize(df):

    df = df.copy()

    df["user"] = (
        df["user"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df["day"] = (
        pd.to_datetime(df["day"])
        .dt.normalize()
    )

    return df


behavioral = normalize(behavioral)
temporal = normalize(temporal)
context = normalize(context)


print(f"\nBehavioral rows : {len(behavioral):,}")
print(f"Temporal rows   : {len(temporal):,}")
print(f"Context rows    : {len(context):,}")


# ================================================================
# 3. MERGE FILE FEATURES
# ================================================================

data = behavioral.merge(
    temporal[
        [
            "user",
            "day",
            "file_events_prev7_mean",
            "file_events_change_7d",
            "temporal_change_count",
            "temporal_change_score",
        ]
    ],
    on=["user", "day"],
    how="left"
)

data = data.merge(
    context[
        [
            "user",
            "day",
            "unique_file_extensions",
            "executable_files",
            "archive_files",
        ]
    ],
    on=["user", "day"],
    how="left"
)


print(f"Merged rows     : {len(data):,}")


# ================================================================
# 4. HANDLE MISSING VALUES
# ================================================================

numeric_columns = [
    "file_events",
    "file_pc_count",
    "unique_files",
    "file_events_clean_zscore",
    "file_events_prev7_mean",
    "file_events_change_7d",
    "temporal_change_count",
    "temporal_change_score",
    "unique_file_extensions",
    "executable_files",
    "archive_files",
]


for column in numeric_columns:

    if column in data.columns:

        data[column] = (
            pd.to_numeric(
                data[column],
                errors="coerce"
            )
            .fillna(0)
        )


# ================================================================
# 5. PRIMARY ANOMALY SIGNALS
# ================================================================

# ------------------------------------------------
# Strong behavioral deviation
# ------------------------------------------------

data["file_behavioral_anomaly_signal"] = (
    data["file_events_clean_zscore"] >= 2.5
).astype(int)


data["file_strong_zscore_signal"] = (
    data["file_events_clean_zscore"] >= 3.0
).astype(int)


# ------------------------------------------------
# Strong temporal deviation
# ------------------------------------------------

data["file_temporal_change_signal"] = (
    data["file_events_change_7d"].abs() >= 0.5
).astype(int)


data["file_strong_temporal_signal"] = (
    data["file_events_change_7d"].abs() >= 1.0
).astype(int)


# ------------------------------------------------
# Behavioral + temporal agreement
# ------------------------------------------------

data["file_behavior_change_signal"] = (
    (data["file_behavioral_anomaly_signal"] == 1)
    &
    (data["file_temporal_change_signal"] == 1)
).astype(int)


# ================================================================
# 6. SUPPORTING ACTIVITY SIGNALS
# ================================================================

# These signals are intentionally weaker.
# They are useful for context but should not dominate
# the final score.

data["file_high_volume_signal"] = (
    data["file_events"] >= 5
).astype(int)


data["file_high_diversity_signal"] = (
    data["unique_files"] >= 5
).astype(int)


data["file_extension_diversity_signal"] = (
    data["unique_file_extensions"] >= 2
).astype(int)


# ================================================================
# 7. FILE-TYPE CONTEXT
# ================================================================

data["file_executable_signal"] = (
    data["executable_files"] > 0
).astype(int)


data["file_executable_anomaly_signal"] = (
    (data["executable_files"] > 0)
    &
    (data["file_behavioral_anomaly_signal"] == 1)
).astype(int)


data["file_archive_signal"] = (
    data["archive_files"] > 0
).astype(int)


# Archive activity is retained as context,
# but it receives low weight because the diagnostic
# showed weak discrimination by itself.


# ================================================================
# 8. ANOMALOUS VOLUME / DIVERSITY COMBINATIONS
# ================================================================

data["file_volume_anomaly_signal"] = (
    (data["file_high_volume_signal"] == 1)
    &
    (data["file_behavioral_anomaly_signal"] == 1)
).astype(int)


data["file_diversity_anomaly_signal"] = (
    (data["file_high_diversity_signal"] == 1)
    &
    (data["file_behavioral_anomaly_signal"] == 1)
).astype(int)


# ================================================================
# 9. EVIDENCE GROUPS
# ================================================================

# Group 1:
# Behavioral deviation
#
# Only the strongest applicable signal contributes.
#
# This prevents:
#
# zscore
# + strong zscore
# + volume anomaly
# + diversity anomaly
#
# from excessively double-counting the same underlying
# file-volume behavior.

data["behavioral_evidence_score"] = np.select(
    [
        data["file_strong_zscore_signal"] == 1,
        data["file_behavioral_anomaly_signal"] == 1,
    ],
    [
        5.0,
        3.0,
    ],
    default=0.0
)


# ================================================================
# 10. TEMPORAL EVIDENCE GROUP
# ================================================================

data["temporal_evidence_score"] = np.select(
    [
        data["file_strong_temporal_signal"] == 1,
        data["file_temporal_change_signal"] == 1,
    ],
    [
        3.0,
        1.0,
    ],
    default=0.0
)


# ================================================================
# 11. COMBINATION EVIDENCE
# ================================================================

# Agreement between behavioral anomaly and temporal change
#
# This is more meaningful than either signal alone.

data["behavior_temporal_agreement_score"] = (
    3.0 * data["file_behavior_change_signal"]
)


# ================================================================
# 12. ACTIVITY CONTEXT SCORE
# ================================================================

data["activity_context_score"] = (
    0.75 * data["file_high_volume_signal"]
    + 0.75 * data["file_high_diversity_signal"]
    + 0.50 * data["file_extension_diversity_signal"]
)


# ================================================================
# 13. FILE-TYPE CONTEXT SCORE
# ================================================================

data["file_type_context_score"] = (
    2.0 * data["file_executable_signal"]
    + 1.0 * data["file_archive_signal"]
)


# Strong executable + behavioral anomaly
# receives additional evidence.

data["file_type_anomaly_bonus"] = (
    3.0 * data["file_executable_anomaly_signal"]
)


# ================================================================
# 14. FINAL V2 RAW SCORE
# ================================================================

data["file_agent_score_raw"] = (
    data["behavioral_evidence_score"]
    + data["temporal_evidence_score"]
    + data["behavior_temporal_agreement_score"]
    + data["activity_context_score"]
    + data["file_type_context_score"]
    + data["file_type_anomaly_bonus"]
)


# ================================================================
# 15. SCORE NORMALIZATION
# ================================================================

MAX_SCORE = (
    5.0      # behavioral
    + 3.0    # temporal
    + 3.0    # behavioral-temporal agreement
    + 2.0    # activity context
    + 3.0    # file-type context
    + 3.0    # executable anomaly bonus
)


data["file_agent_score"] = (
    data["file_agent_score_raw"]
    / MAX_SCORE
    * 100
)


data["file_agent_score"] = (
    data["file_agent_score"]
    .clip(0, 100)
)


# ================================================================
# 16. EVIDENCE COUNT
# ================================================================

# Count independent evidence categories rather than every
# correlated signal.

data["file_evidence_count"] = (
    (
        data["behavioral_evidence_score"] > 0
    ).astype(int)
    +
    (
        data["temporal_evidence_score"] > 0
    ).astype(int)
    +
    (
        data["file_high_volume_signal"] > 0
    ).astype(int)
    +
    (
        data["file_high_diversity_signal"] > 0
    ).astype(int)
    +
    (
        data["file_extension_diversity_signal"] > 0
    ).astype(int)
    +
    (
        data["file_executable_signal"] > 0
    ).astype(int)
    +
    (
        data["file_archive_signal"] > 0
    ).astype(int)
)


# ================================================================
# 17. STATUS
# ================================================================

def classify_status(row):

    score = row["file_agent_score"]

    evidence = row["file_evidence_count"]

    behavioral_anomaly = (
        row["file_behavioral_anomaly_signal"]
    )

    strong_zscore = (
        row["file_strong_zscore_signal"]
    )

    behavior_change = (
        row["file_behavior_change_signal"]
    )

    executable_anomaly = (
        row["file_executable_anomaly_signal"]
    )

    # Strong evidence requires either:
    #
    # 1. strong behavioral anomaly + temporal agreement
    # 2. strong z-score
    # 3. executable + behavioral anomaly
    # 4. multiple independent evidence categories

    if (
        (
            behavioral_anomaly
            and behavior_change
        )
        or strong_zscore
        or executable_anomaly
        or evidence >= 5
    ):
        return "STRONG_FILE_EVIDENCE"

    # Moderate evidence

    if (
        score >= 25
        or evidence >= 3
        or (
            behavioral_anomaly
            and row["file_temporal_change_signal"]
        )
    ):
        return "MODERATE_FILE_EVIDENCE"

    # Weak evidence

    if (
        score > 0
        or evidence >= 1
    ):
        return "WEAK_FILE_EVIDENCE"

    return "NO_FILE_EVIDENCE"


data["file_agent_status"] = (
    data.apply(
        classify_status,
        axis=1
    )
)


# ================================================================
# 18. EXPLANATION
# ================================================================

def build_explanation(row):

    evidence = []

    if row["file_strong_zscore_signal"]:
        evidence.append(
            "strong deviation from user's normal file activity"
        )

    elif row["file_behavioral_anomaly_signal"]:
        evidence.append(
            "unusual file activity compared with behavioral baseline"
        )

    if row["file_strong_temporal_signal"]:
        evidence.append(
            "strong recent increase in file activity"
        )

    elif row["file_temporal_change_signal"]:
        evidence.append(
            "recent increase in file activity"
        )

    if row["file_behavior_change_signal"]:
        evidence.append(
            "behavioral and temporal anomaly agreement"
        )

    if row["file_high_volume_signal"]:
        evidence.append(
            "high file activity"
        )

    if row["file_high_diversity_signal"]:
        evidence.append(
            "high file diversity"
        )

    if row["file_executable_signal"]:
        evidence.append(
            "executable file activity"
        )

    if row["file_executable_anomaly_signal"]:
        evidence.append(
            "executable activity combined with behavioral anomaly"
        )

    if row["file_archive_signal"]:
        evidence.append(
            "archive file activity"
        )

    if not evidence:
        return (
            "No significant file-based anomaly detected."
        )

    return (
        "File evidence: "
        + ", ".join(evidence)
        + "."
    )


data["file_explanation"] = (
    data.apply(
        build_explanation,
        axis=1
    )
)


# ================================================================
# 19. SUMMARY
# ================================================================

print("\n" + "=" * 80)
print("FILE AGENT V2 SUMMARY")
print("=" * 80)


print("\nStatus distribution:")

print(
    data["file_agent_status"]
    .value_counts()
    .to_string()
)


print("\nEvidence count distribution:")

print(
    data["file_evidence_count"]
    .value_counts()
    .sort_index()
    .to_string()
)


print("\nImportant signal counts:")

important_signals = [
    "file_behavioral_anomaly_signal",
    "file_strong_zscore_signal",
    "file_temporal_change_signal",
    "file_strong_temporal_signal",
    "file_behavior_change_signal",
    "file_volume_anomaly_signal",
    "file_diversity_anomaly_signal",
    "file_executable_signal",
    "file_executable_anomaly_signal",
    "file_high_volume_signal",
    "file_high_diversity_signal",
    "file_archive_signal",
]


for signal in important_signals:

    print(
        f"{signal:45s}: "
        f"{int(data[signal].sum()):,}"
    )


print("\nScore statistics:")

print(
    f"Mean  : "
    f"{data['file_agent_score'].mean():.4f}"
)

print(
    f"Median: "
    f"{data['file_agent_score'].median():.4f}"
)

print(
    f"Max   : "
    f"{data['file_agent_score'].max():.4f}"
)


# ================================================================
# 20. SAVE V2 OUTPUT
# ================================================================

output_file = (
    OUTPUT_DIR /
    "file_agent_v2_results.parquet"
)


output_columns = [
    "user",
    "day",

    # Raw file features
    "file_events",
    "file_pc_count",
    "unique_files",
    "unique_file_extensions",
    "executable_files",
    "archive_files",

    # Temporal features
    "file_events_prev7_mean",
    "file_events_change_7d",
    "temporal_change_count",
    "temporal_change_score",

    # Primary signals
    "file_behavioral_anomaly_signal",
    "file_strong_zscore_signal",
    "file_temporal_change_signal",
    "file_strong_temporal_signal",
    "file_behavior_change_signal",

    # Supporting signals
    "file_high_volume_signal",
    "file_high_diversity_signal",
    "file_extension_diversity_signal",

    # File type signals
    "file_executable_signal",
    "file_executable_anomaly_signal",
    "file_archive_signal",

    # Anomaly combinations
    "file_volume_anomaly_signal",
    "file_diversity_anomaly_signal",

    # Scores
    "behavioral_evidence_score",
    "temporal_evidence_score",
    "behavior_temporal_agreement_score",
    "activity_context_score",
    "file_type_context_score",
    "file_type_anomaly_bonus",

    "file_evidence_count",
    "file_agent_score_raw",
    "file_agent_score",
    "file_agent_status",
    "file_explanation",
]


output_columns = [
    column
    for column in output_columns
    if column in data.columns
]


data[output_columns].to_parquet(
    output_file,
    index=False,
    engine="fastparquet"
)


print("\n" + "=" * 80)
print("FILE AGENT V2 COMPLETE")
print("=" * 80)

print(
    f"Rows   : {len(data):,}"
)

print(
    f"Output : {output_file}"
)