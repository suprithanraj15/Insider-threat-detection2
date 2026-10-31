import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
)
def calculate_metrics(y_true, y_pred):
    """
    Calculate classification metrics without sklearn.
    """

    y_true = list(y_true)
    y_pred = list(y_pred)

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for actual, predicted in zip(y_true, y_pred):

        if actual == 1 and predicted == 1:
            tp += 1

        elif actual == 0 and predicted == 0:
            tn += 1

        elif actual == 0 and predicted == 1:
            fp += 1

        elif actual == 1 and predicted == 0:
            fn += 1

    total = tp + tn + fp + fn

    accuracy = (
        (tp + tn) / total
        if total > 0
        else 0.0
    )

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

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    fnr = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else 0.0
    )

    balanced_accuracy = (
        (recall + specificity) / 2
    )

    # Matthews Correlation Coefficient
    denominator = (
        (tp + fp)
        * (tp + fn)
        * (tn + fp)
        * (tn + fn)
    )

    if denominator > 0:
        mcc = (
            (tp * tn) - (fp * fn)
        ) / (denominator ** 0.5)
    else:
        mcc = 0.0

    return {
        "Alerts": tp + fp,
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
        "MCC": mcc,
    }
# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

NETWORK_RESULTS_FILE = os.path.join(
    BASE_DIR,
    "data",
    "outputs",
    "network_agent_v2_results.parquet",
)

GROUND_TRUTH_FILE = os.path.join(
    BASE_DIR,
    "data",
    "outputs",
    "network_ground_truth.parquet",
)

# Thresholds used for evaluation
THRESHOLDS = list(range(0, 101, 5))


# ============================================================
# HELPER FUNCTION
# ============================================================

def calculate_metrics(y_true, y_pred):
    """
    Calculate all evaluation metrics safely.
    """

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    fnr = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else 0
    )

    balanced_accuracy = (
        (recall + specificity) / 2
    )

    mcc = matthews_corrcoef(
        y_true,
        y_pred
    )

    return {
        "Alerts": int(tp + fp),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Specificity": specificity,
        "FPR": fpr,
        "FNR": fnr,
        "Balanced_Accuracy": balanced_accuracy,
        "MCC": mcc,
    }


# ============================================================
# START
# ============================================================

print("=" * 70)
print("NETWORK AGENT V2 - EVALUATION")
print("=" * 70)


# ============================================================
# LOAD NETWORK AGENT RESULTS
# ============================================================

print("\nLoading Network Agent V2 results...")

if not os.path.exists(NETWORK_RESULTS_FILE):
    raise FileNotFoundError(
        f"\nNetwork Agent V2 results not found:\n"
        f"{NETWORK_RESULTS_FILE}"
    )

agent = pd.read_parquet(
    NETWORK_RESULTS_FILE
)

print(f"Agent rows: {len(agent):,}")


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_agent_columns = [
    "user",
    "day",
    "network_v2_score",
]

missing_agent_columns = [
    col
    for col in required_agent_columns
    if col not in agent.columns
]

if missing_agent_columns:
    raise ValueError(
        "\nMissing required columns in Network Agent V2 results:\n"
        + "\n".join(missing_agent_columns)
    )


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("\nLoading Network Ground Truth...")

if not os.path.exists(GROUND_TRUTH_FILE):
    raise FileNotFoundError(
        f"\nNetwork ground truth not found:\n"
        f"{GROUND_TRUTH_FILE}"
    )

ground_truth = pd.read_parquet(
    GROUND_TRUTH_FILE
)

print(f"Ground-truth rows: {len(ground_truth):,}")


# ============================================================
# VALIDATE GROUND TRUTH COLUMNS
# ============================================================

required_gt_columns = [
    "user",
    "day",
]

missing_gt_columns = [
    col
    for col in required_gt_columns
    if col not in ground_truth.columns
]

if missing_gt_columns:
    raise ValueError(
        "\nMissing required columns in network ground truth:\n"
        + "\n".join(missing_gt_columns)
    )


# ============================================================
# FIND GROUND-TRUTH LABEL COLUMN
# ============================================================

possible_label_columns = [
    "network_ground_truth",
    "ground_truth",
    "label",
    "is_malicious",
    "malicious",
]

label_column = None

for col in possible_label_columns:
    if col in ground_truth.columns:
        label_column = col
        break

if label_column is None:
    raise ValueError(
        "\nCould not find the ground-truth label column.\n"
        f"Available columns:\n{list(ground_truth.columns)}"
    )

ground_truth = ground_truth[
    ["user", "day", label_column]
].copy()

ground_truth = ground_truth.rename(
    columns={
        label_column: "network_ground_truth"
    }
)

ground_truth["network_ground_truth"] = (
    ground_truth["network_ground_truth"]
    .fillna(0)
    .astype(int)
)


# ============================================================
# NORMALIZE DATE COLUMNS
# ============================================================

agent["day"] = pd.to_datetime(
    agent["day"]
).dt.date

ground_truth["day"] = pd.to_datetime(
    ground_truth["day"]
).dt.date


# ============================================================
# MERGE AGENT + GROUND TRUTH
# ============================================================

evaluation = agent.merge(
    ground_truth,
    on=["user", "day"],
    how="left",
)

evaluation["network_ground_truth"] = (
    evaluation["network_ground_truth"]
    .fillna(0)
    .astype(int)
)


print("\nGround-truth distribution:")
print(
    evaluation["network_ground_truth"]
    .value_counts()
    .sort_index()
)


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD ANALYSIS")
print("=" * 70)

threshold_results = []

y_true = evaluation[
    "network_ground_truth"
].astype(int)

scores = evaluation[
    "network_v2_score"
].fillna(0)


for threshold in THRESHOLDS:

    y_pred = (
        scores >= threshold
    ).astype(int)

    metrics = calculate_metrics(
        y_true,
        y_pred
    )

    metrics["Threshold"] = threshold

    threshold_results.append(
        metrics
    )


threshold_df = pd.DataFrame(
    threshold_results
)

threshold_df = threshold_df[
    [
        "Threshold",
        "Alerts",
        "TP",
        "TN",
        "FP",
        "FN",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Specificity",
        "FPR",
        "FNR",
        "Balanced_Accuracy",
        "MCC",
    ]
]

print(
    threshold_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# BEST F1
# ============================================================

best_f1 = threshold_df.loc[
    threshold_df["F1"].idxmax()
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

best_precision = threshold_df.loc[
    threshold_df["Precision"].idxmax()
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

best_balanced = threshold_df.loc[
    threshold_df["Balanced_Accuracy"].idxmax()
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

best_mcc = threshold_df.loc[
    threshold_df["MCC"].idxmax()
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

malicious = evaluation[
    evaluation["network_ground_truth"] == 1
].copy()


# ------------------------------------------------------------
# IMPORTANT FIX:
# Do NOT assume that "scenario" exists.
# ------------------------------------------------------------

malicious_columns = [
    "user",
    "day",
    "network_v2_score",
    "network_v2_evidence_count",
    "network_v2_status",
]

# Add scenario only if it actually exists
if "scenario" in malicious.columns:
    malicious_columns.append("scenario")


# Add useful network columns only when available
optional_columns = [
    "web_visits",
    "unique_urls",
    "unique_domains",
    "web_visits_clean_zscore",
    "high_web_days_prev7",
    "network_persistence_ratio",
]

for col in optional_columns:

    if col in malicious.columns:
        malicious_columns.append(col)


# Remove duplicate columns while preserving order
malicious_columns = list(
    dict.fromkeys(malicious_columns)
)


print(
    malicious[
        malicious_columns
    ].sort_values(
        "network_v2_score",
        ascending=False
    ).to_string(
        index=False
    )
)


# ============================================================
# MALICIOUS-DAY DETECTION SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MALICIOUS-DAY DETECTION SUMMARY")
print("=" * 70)

total_malicious = len(malicious)

print(
    f"Total malicious network days : {total_malicious}"
)


for threshold in THRESHOLDS:

    detected = (
        malicious["network_v2_score"]
        >= threshold
    ).sum()

    recall = (
        detected / total_malicious
        if total_malicious > 0
        else 0
    )

    print(
        f"Threshold {threshold:>3}: "
        f"{detected:>3}/{total_malicious} "
        f"detected | Recall = {recall:.6f}"
    )


# ============================================================
# SAVE EVALUATION RESULTS
# ============================================================

evaluation_output = os.path.join(
    BASE_DIR,
    "data",
    "outputs",
    "network_agent_v2_evaluation.csv",
)

threshold_df.to_csv(
    evaluation_output,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("NETWORK AGENT V2 EVALUATION COMPLETE")
print("=" * 70)

print(
    f"\nEvaluation results saved to:\n"
    f"{evaluation_output}"
)