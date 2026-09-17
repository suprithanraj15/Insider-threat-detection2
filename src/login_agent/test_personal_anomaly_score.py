from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"

BEHAVIORAL_FILE = OUTPUT_DIR / "final_behavioral_features.parquet"
TEMPORAL_FILE = OUTPUT_DIR / "temporal_behavioral_features.parquet"
CONTEXT_FILE = OUTPUT_DIR / "rule_context_features.parquet"
LOGIN_GT_FILE = OUTPUT_DIR / "login_ground_truth.parquet"


print("=" * 80)
print("PERSONAL BASELINE ANOMALY SCORE EXPERIMENT")
print("=" * 80)


# ================================================================
# 1. LOAD
# ================================================================

behavioral = pd.read_parquet(BEHAVIORAL_FILE)
temporal = pd.read_parquet(TEMPORAL_FILE)
context = pd.read_parquet(CONTEXT_FILE)
login_gt = pd.read_parquet(LOGIN_GT_FILE)


def normalize(df):
    df = df.copy()
    df["user"] = df["user"].astype(str).str.lower().str.strip()
    df["day"] = pd.to_datetime(df["day"]).dt.normalize()
    return df


behavioral = normalize(behavioral)
temporal = normalize(temporal)
context = normalize(context)
login_gt = normalize(login_gt)


# ================================================================
# 2. MERGE
# ================================================================

data = behavioral.merge(
    temporal[
        [
            "user",
            "day",
            "logon_events_prev7_mean",
            "logon_events_change_7d",
            "temporal_change_count",
            "temporal_change_score",
        ]
    ],
    on=["user", "day"],
    how="left",
)

data = data.merge(
    context[
        [
            "user",
            "day",
            "after_hours_logons",
            "unique_logon_pcs",
            "total_logon_events",
            "after_hours_logon_ratio",
        ]
    ],
    on=["user", "day"],
    how="left",
)

data = data.merge(
    login_gt[
        [
            "user",
            "day",
            "login_ground_truth",
        ]
    ],
    on=["user", "day"],
    how="left",
)


# ================================================================
# 3. VERIFY MERGE
# ================================================================

print(f"\nTotal rows : {len(data):,}")

print(
    f"Login GT malicious rows : "
    f"{int(data['login_ground_truth'].sum())}"
)

if data["login_ground_truth"].isna().any():
    print("WARNING: Missing login ground-truth values found.")
    data["login_ground_truth"] = (
        data["login_ground_truth"]
        .fillna(0)
        .astype(int)
    )
else:
    data["login_ground_truth"] = (
        data["login_ground_truth"]
        .astype(int)
    )


# ================================================================
# 4. PERSONAL BASELINE
# ================================================================

percentile_features = [
    "logon_events",
    "after_hours_logons",
    "after_hours_logon_ratio",
    "logon_events_change_7d",
    "temporal_change_score",
    "logon_events_clean_zscore",
]


print("\nCalculating personal baselines...")


# Important:
# We calculate each user's normal distribution separately.
# Malicious login days are excluded from the baseline.

normal_data = data[data["login_ground_truth"] == 0].copy()

baseline = (
    normal_data
    .groupby("user")[percentile_features]
    .agg(["count", "mean", "median"])
)


# ================================================================
# 5. PERSONAL PERCENTILES
# ================================================================

for feature in percentile_features:

    pct_column = f"{feature}_personal_pct"

    data[pct_column] = np.nan

    for user, indices in data.groupby("user").groups.items():

        normal_values = normal_data.loc[
            normal_data["user"] == user,
            feature
        ].dropna()

        if len(normal_values) == 0:
            continue

        values = data.loc[indices, feature]

        data.loc[indices, pct_column] = (
            values.apply(
                lambda x:
                (normal_values <= x).mean() * 100
                if pd.notna(x)
                else np.nan
            )
        )


# ================================================================
# 6. ANOMALY STRENGTH
# ================================================================

def anomaly_strength(percentile):

    if pd.isna(percentile):
        return 0.0

    return max(0.0, (percentile - 50.0) / 50.0)


for feature in percentile_features:

    pct_column = f"{feature}_personal_pct"
    anomaly_column = f"{feature}_personal_anomaly"

    data[anomaly_column] = (
        data[pct_column]
        .apply(anomaly_strength)
    )


# ================================================================
# 7. BASIC SCORE
# ================================================================

data["personal_score_basic"] = (
    data["logon_events_personal_anomaly"]
    + data["after_hours_logons_personal_anomaly"]
    + data["after_hours_logon_ratio_personal_anomaly"]
    + data["logon_events_change_7d_personal_anomaly"]
    + data["temporal_change_score_personal_anomaly"]
    + data["logon_events_clean_zscore_personal_anomaly"]
)


# ================================================================
# 8. WEIGHTED SCORE
# ================================================================

data["personal_score_weighted"] = (
    1.0 * data["logon_events_personal_anomaly"]
    + 1.5 * data["after_hours_logons_personal_anomaly"]
    + 1.5 * data["after_hours_logon_ratio_personal_anomaly"]
    + 1.0 * data["logon_events_change_7d_personal_anomaly"]
    + 1.5 * data["temporal_change_score_personal_anomaly"]
    + 1.5 * data["logon_events_clean_zscore_personal_anomaly"]
)


# ================================================================
# 9. EVALUATION FUNCTION
# ================================================================

def evaluate(y_true, scores, threshold):

    predictions = (scores >= threshold).astype(int)

    tp = int(((predictions == 1) & (y_true == 1)).sum())
    tn = int(((predictions == 0) & (y_true == 0)).sum())
    fp = int(((predictions == 1) & (y_true == 0)).sum())
    fn = int(((predictions == 0) & (y_true == 1)).sum())

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    accuracy = (tp + tn) / len(y_true)

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    fpr = 1 - specificity

    return {
        "threshold": threshold,
        "alerts": int(predictions.sum()),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "fpr": fpr,
    }


# ================================================================
# 10. TEST THRESHOLDS
# ================================================================

y_true = data["login_ground_truth"]

results = []


basic_thresholds = np.arange(0.5, 6.01, 0.25)
weighted_thresholds = np.arange(0.5, 8.01, 0.25)


for threshold in basic_thresholds:

    result = evaluate(
        y_true,
        data["personal_score_basic"],
        threshold
    )

    result["score_type"] = "basic"

    results.append(result)


for threshold in weighted_thresholds:

    result = evaluate(
        y_true,
        data["personal_score_weighted"],
        threshold
    )

    result["score_type"] = "weighted"

    results.append(result)


results_df = pd.DataFrame(results)


# ================================================================
# 11. BEST F1
# ================================================================

print("\n" + "=" * 80)
print("BEST RESULTS BY F1")
print("=" * 80)

best_f1 = (
    results_df
    .sort_values(
        ["f1", "precision", "recall"],
        ascending=False
    )
    .head(15)
)

print(
    best_f1[
        [
            "score_type",
            "threshold",
            "alerts",
            "TP",
            "FP",
            "FN",
            "precision",
            "recall",
            "f1",
            "fpr",
        ]
    ].to_string(index=False)
)


# ================================================================
# 12. BEST PRECISION
# ================================================================

print("\n" + "=" * 80)
print("BEST RESULTS BY PRECISION")
print("=" * 80)

best_precision = (
    results_df
    .sort_values(
        ["precision", "f1", "recall"],
        ascending=False
    )
    .head(15)
)

print(
    best_precision[
        [
            "score_type",
            "threshold",
            "alerts",
            "TP",
            "FP",
            "FN",
            "precision",
            "recall",
            "f1",
            "fpr",
        ]
    ].to_string(index=False)
)


# ================================================================
# 13. BEST RESULTS WITH LOW FALSE-POSITIVE RATE
# ================================================================

print("\n" + "=" * 80)
print("BEST RESULTS WITH FPR <= 1%")
print("=" * 80)

low_fpr = results_df[
    results_df["fpr"] <= 0.01
].copy()

if len(low_fpr) == 0:

    print("No tested threshold achieved FPR <= 1%.")

else:

    low_fpr = (
        low_fpr
        .sort_values(
            ["f1", "precision", "recall"],
            ascending=False
        )
        .head(15)
    )

    print(
        low_fpr[
            [
                "score_type",
                "threshold",
                "alerts",
                "TP",
                "FP",
                "FN",
                "precision",
                "recall",
                "f1",
                "fpr",
            ]
        ].to_string(index=False)
    )


# ================================================================
# 14. MALICIOUS LOGIN DAYS
# ================================================================

print("\n" + "=" * 80)
print("MALICIOUS LOGIN DAYS")
print("=" * 80)

malicious = (
    data[data["login_ground_truth"] == 1]
    .copy()
    .sort_values(["user", "day"])
)


malicious_columns = [
    "user",
    "day",
    "logon_events",
    "after_hours_logons",
    "after_hours_logon_ratio",
    "logon_events_change_7d",
    "temporal_change_score",
    "logon_events_clean_zscore",
    "logon_events_personal_pct",
    "after_hours_logons_personal_pct",
    "after_hours_logon_ratio_personal_pct",
    "logon_events_change_7d_personal_pct",
    "temporal_change_score_personal_pct",
    "logon_events_clean_zscore_personal_pct",
    "personal_score_basic",
    "personal_score_weighted",
]


# Only print columns that actually exist.
malicious_columns = [
    column
    for column in malicious_columns
    if column in malicious.columns
]


print(
    malicious[malicious_columns]
    .to_string(index=False)
)


# ================================================================
# 15. CURRENT LOGIN AGENT COMPARISON
# ================================================================

LOGIN_AGENT_FILE = OUTPUT_DIR / "login_agent_results.parquet"


if LOGIN_AGENT_FILE.exists():

    login_agent = pd.read_parquet(LOGIN_AGENT_FILE)

    login_agent["user"] = (
        login_agent["user"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    login_agent["day"] = (
        pd.to_datetime(login_agent["day"])
        .dt.normalize()
    )

    comparison = malicious.merge(
        login_agent[
            [
                "user",
                "day",
                "login_evidence_count",
                "login_agent_score",
                "login_agent_status",
            ]
        ],
        on=["user", "day"],
        how="left",
    )

    print("\n" + "=" * 80)
    print("CURRENT LOGIN AGENT VS PERSONAL SCORE")
    print("=" * 80)

    print(
        comparison[
            [
                "user",
                "day",
                "personal_score_basic",
                "personal_score_weighted",
                "login_evidence_count",
                "login_agent_score",
                "login_agent_status",
            ]
        ].to_string(index=False)
    )


# ================================================================
# 16. SAVE RESULTS
# ================================================================

output_file = (
    OUTPUT_DIR /
    "personal_anomaly_score_experiment.parquet"
)

results_df.to_parquet(
    output_file,
    index=False
)


# Save malicious-day diagnostic too.

malicious_output = (
    OUTPUT_DIR /
    "personal_anomaly_malicious_days.parquet"
)

malicious.to_parquet(
    malicious_output,
    index=False
)


print("\n" + "=" * 80)
print("EXPERIMENT COMPLETE")
print("=" * 80)

print(f"Results saved : {output_file}")
print(f"Malicious diagnostic saved : {malicious_output}")