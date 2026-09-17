from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"

BEHAVIORAL_FILE = OUTPUT_DIR / "final_behavioral_features.parquet"
TEMPORAL_FILE = OUTPUT_DIR / "temporal_behavioral_features.parquet"
CONTEXT_FILE = OUTPUT_DIR / "rule_context_features.parquet"


print("=" * 80)
print("FILE AGENT V1")
print("=" * 80)


# ================================================================
# 1. LOAD DATA
# ================================================================

behavioral = pd.read_parquet(BEHAVIORAL_FILE, engine="fastparquet")
temporal = pd.read_parquet(TEMPORAL_FILE, engine="fastparquet")
context = pd.read_parquet(CONTEXT_FILE, engine="fastparquet")


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
# 2. MERGE FILE-RELATED FEATURES
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
# 3. FILE SIGNALS
# ================================================================

# Basic file activity
data["file_activity_signal"] = (
    data["file_events"] > 0
).astype(int)


# High file activity
data["file_high_volume_signal"] = (
    data["file_events"] >= 5
).astype(int)


# High file diversity
data["file_high_diversity_signal"] = (
    data["unique_files"] >= 5
).astype(int)


# Multiple PCs involved in file activity
data["file_multiple_pc_signal"] = (
    data["file_pc_count"] >= 2
).astype(int)


# Executable file activity
data["file_executable_signal"] = (
    data["executable_files"] > 0
).astype(int)


# Archive file activity
data["file_archive_signal"] = (
    data["archive_files"] > 0
).astype(int)


# Multiple file extensions
data["file_extension_diversity_signal"] = (
    data["unique_file_extensions"] >= 2
).astype(int)


# Strong behavioral deviation
data["file_behavioral_anomaly_signal"] = (
    data["file_events_clean_zscore"] >= 2.5
).astype(int)


# Very strong behavioral deviation
data["file_strong_zscore_signal"] = (
    data["file_events_clean_zscore"] >= 3.0
).astype(int)


# Temporal increase
data["file_temporal_change_signal"] = (
    data["file_events_change_7d"].abs() >= 0.5
).astype(int)


# Strong temporal increase
data["file_strong_temporal_signal"] = (
    data["file_events_change_7d"].abs() >= 1.0
).astype(int)


# ================================================================
# 4. COMBINATION SIGNALS
# ================================================================

# File volume + behavioral anomaly
data["file_volume_anomaly_signal"] = (
    (data["file_high_volume_signal"] == 1)
    &
    (data["file_behavioral_anomaly_signal"] == 1)
).astype(int)


# File diversity + behavioral anomaly
data["file_diversity_anomaly_signal"] = (
    (data["file_high_diversity_signal"] == 1)
    &
    (data["file_behavioral_anomaly_signal"] == 1)
).astype(int)


# Executable + behavioral anomaly
data["file_executable_anomaly_signal"] = (
    (data["file_executable_signal"] == 1)
    &
    (data["file_behavioral_anomaly_signal"] == 1)
).astype(int)


# Archive + high volume
data["file_archive_volume_signal"] = (
    (data["file_archive_signal"] == 1)
    &
    (data["file_high_volume_signal"] == 1)
).astype(int)


# Temporal change + behavioral anomaly
data["file_behavior_change_signal"] = (
    (data["file_temporal_change_signal"] == 1)
    &
    (data["file_behavioral_anomaly_signal"] == 1)
).astype(int)


# ================================================================
# 5. EVIDENCE COUNT
# ================================================================

evidence_columns = [
    "file_high_volume_signal",
    "file_high_diversity_signal",
    "file_multiple_pc_signal",
    "file_executable_signal",
    "file_archive_signal",
    "file_extension_diversity_signal",
    "file_behavioral_anomaly_signal",
    "file_strong_zscore_signal",
    "file_temporal_change_signal",
    "file_strong_temporal_signal",
    "file_volume_anomaly_signal",
    "file_diversity_anomaly_signal",
    "file_executable_anomaly_signal",
    "file_archive_volume_signal",
    "file_behavior_change_signal",
]


data["file_evidence_count"] = (
    data[evidence_columns]
    .sum(axis=1)
)


# ================================================================
# 6. WEIGHTED FILE SCORE
# ================================================================

data["file_agent_score_raw"] = (
    1.0 * data["file_high_volume_signal"]
    + 1.0 * data["file_high_diversity_signal"]
    + 0.5 * data["file_multiple_pc_signal"]
    + 1.5 * data["file_executable_signal"]
    + 1.0 * data["file_archive_signal"]
    + 0.5 * data["file_extension_diversity_signal"]
    + 2.0 * data["file_behavioral_anomaly_signal"]
    + 1.0 * data["file_strong_zscore_signal"]
    + 1.0 * data["file_temporal_change_signal"]
    + 1.5 * data["file_strong_temporal_signal"]
    + 2.0 * data["file_volume_anomaly_signal"]
    + 2.0 * data["file_diversity_anomaly_signal"]
    + 2.5 * data["file_executable_anomaly_signal"]
    + 2.0 * data["file_archive_volume_signal"]
    + 2.0 * data["file_behavior_change_signal"]
)


# Maximum possible score
MAX_SCORE = (
    1.0
    + 1.0
    + 0.5
    + 1.5
    + 1.0
    + 0.5
    + 2.0
    + 1.0
    + 1.0
    + 1.5
    + 2.0
    + 2.0
    + 2.5
    + 2.0
    + 2.0
)


data["file_agent_score"] = (
    data["file_agent_score_raw"]
    / MAX_SCORE
    * 100
)


# ================================================================
# 7. STATUS
# ================================================================

def classify_status(row):

    score = row["file_agent_score"]
    evidence = row["file_evidence_count"]

    if score >= 60 or evidence >= 5:
        return "STRONG_FILE_EVIDENCE"

    if score >= 30 or evidence >= 3:
        return "MODERATE_FILE_EVIDENCE"

    if score > 0 or evidence >= 1:
        return "WEAK_FILE_EVIDENCE"

    return "NO_FILE_EVIDENCE"


data["file_agent_status"] = (
    data.apply(
        classify_status,
        axis=1
    )
)


# ================================================================
# 8. EXPLANATION
# ================================================================

def build_explanation(row):

    evidence = []

    if row["file_high_volume_signal"]:
        evidence.append("high file activity")

    if row["file_high_diversity_signal"]:
        evidence.append("high file diversity")

    if row["file_multiple_pc_signal"]:
        evidence.append("file activity across multiple PCs")

    if row["file_executable_signal"]:
        evidence.append("executable file activity")

    if row["file_archive_signal"]:
        evidence.append("archive file activity")

    if row["file_behavioral_anomaly_signal"]:
        evidence.append("file activity behavior anomaly")

    if row["file_temporal_change_signal"]:
        evidence.append("temporal increase in file activity")

    if not evidence:
        return "No significant file-based anomaly detected."

    return "File evidence: " + ", ".join(evidence) + "."


data["file_explanation"] = (
    data.apply(
        build_explanation,
        axis=1
    )
)


# ================================================================
# 9. SUMMARY
# ================================================================

print("\n" + "=" * 80)
print("FILE AGENT SUMMARY")
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


print("\nSignal counts:")

signal_summary = {}

for column in evidence_columns:

    signal_summary[column] = int(
        data[column].sum()
    )

for name, count in signal_summary.items():

    print(
        f"{name:40s}: {count:,}"
    )


print("\nScore statistics:")

print(
    f"Mean  : {data['file_agent_score'].mean():.4f}"
)

print(
    f"Median: {data['file_agent_score'].median():.4f}"
)

print(
    f"Max   : {data['file_agent_score'].max():.4f}"
)


# ================================================================
# 10. SAVE
# ================================================================

output_file = (
    OUTPUT_DIR /
    "file_agent_results.parquet"
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

    # Signals
    *evidence_columns,

    # Final output
    "file_evidence_count",
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
print("FILE AGENT V1 COMPLETE")
print("=" * 80)

print(
    f"Rows   : {len(data):,}"
)

print(
    f"Output : {output_file}"
)