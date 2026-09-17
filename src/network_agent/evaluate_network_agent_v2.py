from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

RESULTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "network_agent_v2_results.parquet"
)

GT_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "network_ground_truth.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "network_agent_v2_evaluation.csv"
)


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    total = tp + tn + fp + fn

    accuracy = (
        (tp + tn) / total
        if total > 0 else 0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0 else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0 else 0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0 else 0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0 else 0
    )

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0 else 0
    )

    fnr = (
        fn / (fn + tp)
        if (fn + tp) > 0 else 0
    )

    balanced_accuracy = (
        (recall + specificity) / 2
    )

    # Matthews correlation coefficient
    denominator = np.sqrt(
        (tp + fp)
        * (tp + fn)
        * (tn + fp)
        * (tn + fn)
    )

    if denominator > 0:

        mcc = (
            (tp * tn) - (fp * fn)
        ) / denominator

    else:

        mcc = 0

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Specificity": specificity,
        "FPR": fpr,
        "FNR": fnr,
        "Balanced_Accuracy": balanced_accuracy,
        "MCC": mcc
    }


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("NETWORK AGENT V2 - EVALUATION")
print("=" * 70)


# ============================================================
# LOAD RESULTS
# ============================================================

print("\nLoading Network Agent V2 results...")

results = pd.read_parquet(
    RESULTS_FILE,
    engine="fastparquet"
)

print(
    f"Agent rows: {len(results):,}"
)


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("\nLoading Network Ground Truth...")

gt = pd.read_parquet(
    GT_FILE,
    engine="fastparquet"
)

print(
    f"Ground-truth rows: {len(gt):,}"
)


# ============================================================
# NORMALIZE KEYS
# ============================================================

for df in [results, gt]:

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


# ============================================================
# MERGE GROUND TRUTH
# ============================================================

data = results.merge(
    gt[
        [
            "user",
            "day",
            "scenario",
            "network_ground_truth"
        ]
    ],
    on=["user", "day"],
    how="left"
)


data["network_ground_truth"] = (
    data["network_ground_truth"]
    .fillna(0)
    .astype(int)
)


print("\nGround-truth distribution:")

print(
    data["network_ground_truth"]
    .value_counts()
    .sort_index()
)


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD ANALYSIS")
print("=" * 70)


thresholds = list(range(0, 101, 5))

evaluation_rows = []


for threshold in thresholds:

    y_true = data["network_ground_truth"]

    y_pred = (
        data["network_v2_score"]
        >= threshold
    ).astype(int)

    metrics = calculate_metrics(
        y_true,
        y_pred
    )

    alerts = int(y_pred.sum())

    row = {
        "Threshold": threshold,
        "Alerts": alerts,
        **metrics
    }

    evaluation_rows.append(row)


evaluation = pd.DataFrame(
    evaluation_rows
)


print(
    evaluation.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.6f}".format,
            "Precision": "{:.6f}".format,
            "Recall": "{:.6f}".format,
            "F1": "{:.6f}".format,
            "Specificity": "{:.6f}".format,
            "FPR": "{:.6f}".format,
            "FNR": "{:.6f}".format,
            "Balanced_Accuracy": "{:.6f}".format,
            "MCC": "{:.6f}".format
        }
    )
)


# ============================================================
# BEST F1
# ============================================================

best_f1 = evaluation.loc[
    evaluation["F1"].idxmax()
]


print("\n" + "=" * 70)
print("BEST F1 THRESHOLD")
print("=" * 70)

print(
    best_f1.to_string()
)


# ============================================================
# BEST PRECISION
# ============================================================

best_precision = evaluation.loc[
    evaluation["Precision"].idxmax()
]


print("\n" + "=" * 70)
print("BEST PRECISION THRESHOLD")
print("=" * 70)

print(
    best_precision.to_string()
)


# ============================================================
# BEST BALANCED ACCURACY
# ============================================================

best_balanced = evaluation.loc[
    evaluation["Balanced_Accuracy"].idxmax()
]


print("\n" + "=" * 70)
print("BEST BALANCED ACCURACY THRESHOLD")
print("=" * 70)

print(
    best_balanced.to_string()
)


# ============================================================
# BEST MCC
# ============================================================

best_mcc = evaluation.loc[
    evaluation["MCC"].idxmax()
]


print("\n" + "=" * 70)
print("BEST MCC THRESHOLD")
print("=" * 70)

print(
    best_mcc.to_string()
)


# ============================================================
# MALICIOUS NETWORK DAYS
# ============================================================

print("\n" + "=" * 70)
print("MALICIOUS NETWORK DAYS AND V2 SCORES")
print("=" * 70)


malicious = data[
    data["network_ground_truth"] == 1
].copy()


malicious_columns = [
    "user",
    "day",
    "scenario",
    "web_visits",
    "unique_urls",
    "unique_domains",
    "web_visits_clean_zscore",
    "high_web_days_prev7",
    "high_web_days_prev14",
    "network_persistence_ratio",
    "network_v2_raw_score",
    "network_v2_score",
    "network_v2_evidence_count",
    "network_v2_status"
]


print(
    malicious[
        malicious_columns
    ]
    .sort_values(
        ["user", "day"]
    )
    .to_string(index=False)
)


# ============================================================
# SCENARIO COVERAGE
# ============================================================

print("\n" + "=" * 70)
print("SCENARIO COVERAGE")
print("=" * 70)


threshold = float(
    best_f1["Threshold"]
)


data["v2_alert"] = (
    data["network_v2_score"]
    >= threshold
).astype(int)


scenario_rows = []


for scenario, group in data[
    data["network_ground_truth"] == 1
].groupby("scenario"):

    malicious_days = len(group)

    detected = int(
        group["v2_alert"].sum()
    )

    recall = (
        detected / malicious_days
        if malicious_days > 0
        else 0
    )

    scenario_rows.append({
        "Scenario": scenario,
        "Malicious_Days": malicious_days,
        "Detected": detected,
        "Recall": recall
    })


scenario_table = pd.DataFrame(
    scenario_rows
)


print(
    scenario_table.to_string(
        index=False,
        formatters={
            "Recall": "{:.4f}".format
        }
    )
)


# ============================================================
# USER COVERAGE
# ============================================================

print("\n" + "=" * 70)
print("USER COVERAGE")
print("=" * 70)


user_rows = []


for user, group in data[
    data["network_ground_truth"] == 1
].groupby("user"):

    malicious_days = len(group)

    detected = int(
        group["v2_alert"].sum()
    )

    recall = (
        detected / malicious_days
        if malicious_days > 0
        else 0
    )

    user_rows.append({
        "User": user,
        "Malicious_Days": malicious_days,
        "Detected": detected,
        "Recall": recall
    })


user_table = pd.DataFrame(
    user_rows
)


print(
    user_table.to_string(
        index=False,
        formatters={
            "Recall": "{:.4f}".format
        }
    )
)


# ============================================================
# SAVE
# ============================================================

evaluation.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nEvaluation saved to:")
print(OUTPUT_FILE)


print("\n" + "=" * 70)
print("NETWORK AGENT V2 EVALUATION COMPLETE")
print("=" * 70)