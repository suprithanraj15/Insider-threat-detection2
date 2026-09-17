from pathlib import Path
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RULE_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "rule_agent_results.parquet"
)

TEMPORAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "temporal_behavioral_features.parquet"
)


print("=" * 70)
print("RULE AGENT V2 - STRATEGY TESTING")
print("=" * 70)


# ------------------------------------------------------------
# 1. Load existing Rule Agent results
# ------------------------------------------------------------

print("\n1. Loading Rule Agent results...")

rules = pd.read_parquet(RULE_FILE)

print(f"Rule Agent rows: {len(rules):,}")


# ------------------------------------------------------------
# 2. Load temporal features
# ------------------------------------------------------------

print("\n2. Loading temporal features...")

temporal = pd.read_parquet(TEMPORAL_FILE)

print(f"Temporal rows: {len(temporal):,}")


# ------------------------------------------------------------
# 3. Keep only temporal columns needed
# ------------------------------------------------------------

temporal_columns = [
    "user",
    "day",
    "total_activity_change_7d",
    "logon_events_change_7d",
    "device_events_change_7d",
    "file_events_change_7d",
    "emails_sent_change_7d",
    "web_visits_change_7d",
]


temporal = temporal[temporal_columns].copy()


# ------------------------------------------------------------
# 4. Merge existing rule results with temporal evidence
# ------------------------------------------------------------

print("\n3. Combining Rule Agent and temporal evidence...")

df = rules.merge(
    temporal,
    on=["user", "day"],
    how="inner"
)

print(f"Combined rows: {len(df):,}")


# ------------------------------------------------------------
# 5. Existing rule count
# ------------------------------------------------------------

rule_columns = [
    "rule_high_activity",
    "rule_unusual_logon",
    "rule_unusual_device",
    "rule_unusual_file",
    "rule_unusual_email",
    "rule_unusual_web",
    "rule_behavior_deviation",
]

df["rule_count"] = df[rule_columns].sum(axis=1)


# ------------------------------------------------------------
# 6. Temporal evidence counts
# ------------------------------------------------------------

change_columns = [
    "total_activity_change_7d",
    "logon_events_change_7d",
    "device_events_change_7d",
    "file_events_change_7d",
    "emails_sent_change_7d",
    "web_visits_change_7d",
]


df["temporal_count_1"] = (
    df[change_columns]
    .abs()
    .ge(1.0)
    .sum(axis=1)
)


df["temporal_count_0_5"] = (
    df[change_columns]
    .abs()
    .ge(0.5)
    .sum(axis=1)
)


# ------------------------------------------------------------
# 7. Strong activity rules
# ------------------------------------------------------------

df["strong_activity_rules"] = (
    df["rule_unusual_device"]
    + df["rule_unusual_file"]
)


# ------------------------------------------------------------
# 8. Ground truth
# ------------------------------------------------------------

y_true = df["ground_truth"]


# ------------------------------------------------------------
# Evaluation function
# ------------------------------------------------------------

def evaluate(name, prediction):

    prediction = prediction.astype(int)

    precision = precision_score(
        y_true,
        prediction,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        prediction,
        zero_division=0
    )

    tp = (
        (y_true == 1)
        & (prediction == 1)
    ).sum()

    fp = (
        (y_true == 0)
        & (prediction == 1)
    ).sum()

    fn = (
        (y_true == 1)
        & (prediction == 0)
    ).sum()

    print(
        f"{name:40s}"
        f" TP={tp:3d}"
        f" FP={fp:6d}"
        f" FN={fn:3d}"
        f" Precision={precision:.4f}"
        f" Recall={recall:.4f}"
        f" F1={f1:.4f}"
    )


# ------------------------------------------------------------
# Test strategies
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STRATEGY RESULTS")
print("=" * 70)


# Strategy 1
evaluate(
    "Existing: 1+ rule",
    df["rule_count"] >= 1
)


# Strategy 2
evaluate(
    "2+ existing rules",
    df["rule_count"] >= 2
)


# Strategy 3
evaluate(
    "1 rule + 1 temporal change",
    (df["rule_count"] >= 1)
    & (df["temporal_count_1"] >= 1)
)


# Strategy 4
evaluate(
    "1 rule + 2 temporal changes",
    (df["rule_count"] >= 1)
    & (df["temporal_count_1"] >= 2)
)


# Strategy 5
evaluate(
    "Device/File rule + temporal",
    (df["strong_activity_rules"] >= 1)
    & (df["temporal_count_0_5"] >= 1)
)


# Strategy 6
evaluate(
    "2 existing + temporal",
    (df["rule_count"] >= 2)
    & (df["temporal_count_0_5"] >= 1)
)


# Strategy 7
evaluate(
    "Strong activity OR 3+ rules",
    (df["strong_activity_rules"] >= 1)
    | (df["rule_count"] >= 3)
)


print("\n" + "=" * 70)
print("STRATEGY TESTING COMPLETED")
print("=" * 70)