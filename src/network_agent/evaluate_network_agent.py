from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

AGENT_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "network_agent_results.parquet"
)

GT_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "network_ground_truth.parquet"
)


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(y_true, y_pred):

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

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
        2 * precision * recall / (precision + recall)
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

    # Matthews Correlation Coefficient
    denominator = np.sqrt(
        (tp + fp)
        * (tp + fn)
        * (tn + fp)
        * (tn + fn)
    )

    mcc = (
        ((tp * tn) - (fp * fn)) / denominator
        if denominator > 0 else 0
    )

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
# LOAD
# ============================================================

print("=" * 70)
print("NETWORK AGENT V1 - EVALUATION")
print("=" * 70)

print("\nLoading Network Agent results...")

agent = pd.read_parquet(
    AGENT_FILE,
    engine="fastparquet"
)

print(f"Agent rows: {len(agent):,}")


print("\nLoading Network Ground Truth...")

gt = pd.read_parquet(
    GT_FILE,
    engine="fastparquet"
)

print(f"Ground-truth network days: {len(gt):,}")


# ============================================================
# NORMALIZE KEYS
# ============================================================

for df in [agent, gt]:

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
# MERGE GROUND TRUTH
# ============================================================

data = agent.merge(
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
    .to_string()
)


# ============================================================
# THRESHOLD EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD ANALYSIS")
print("=" * 70)

thresholds = [
    0,
    5,
    10,
    15,
    20,
    25,
    30,
    35,
    40,
    45,
    50,
    55,
    60,
    65,
    70,
    75,
    80,
    85,
    90,
    95
]

results = []

for threshold in thresholds:

    y_true = data["network_ground_truth"]

    y_pred = (
        data["network_score"] >= threshold
    ).astype(int)

    metrics = calculate_metrics(
        y_true,
        y_pred
    )

    metrics["Threshold"] = threshold
    metrics["Alerts"] = int(y_pred.sum())

    results.append(metrics)


results_df = pd.DataFrame(results)

results_df = results_df[
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
        "MCC"
    ]
]


# ============================================================
# PRINT RESULTS
# ============================================================

print(
    results_df.to_string(
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

best_f1 = results_df.loc[
    results_df["F1"].idxmax()
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

best_precision = results_df.loc[
    results_df["Precision"].idxmax()
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

best_balanced = results_df.loc[
    results_df["Balanced_Accuracy"].idxmax()
]

print("\n" + "=" * 70)
print("BEST BALANCED ACCURACY THRESHOLD")
print("=" * 70)

print(
    best_balanced.to_string()
)


# ============================================================
# MALICIOUS DAY SCORES
# ============================================================

malicious = data[
    data["network_ground_truth"] == 1
].copy()

print("\n" + "=" * 70)
print("MALICIOUS NETWORK DAYS AND SCORES")
print("=" * 70)

malicious_display = malicious[
    [
        "user",
        "day",
        "scenario",
        "network_score",
        "network_evidence_count",
        "network_status"
    ]
].sort_values(
    ["user", "day"]
)

print(
    malicious_display.to_string(
        index=False
    )
)


# ============================================================
# SCENARIO COVERAGE
# ============================================================

print("\n" + "=" * 70)
print("SCENARIO COVERAGE")
print("=" * 70)

selected_threshold = float(
    best_f1["Threshold"]
)

data["predicted"] = (
    data["network_score"] >= selected_threshold
).astype(int)

scenario_rows = []

for scenario, group in (
    data[
        data["network_ground_truth"] == 1
    ]
    .groupby("scenario")
):

    total = len(group)
    detected = int(group["predicted"].sum())

    scenario_rows.append({
        "Scenario": scenario,
        "Malicious_Days": total,
        "Detected": detected,
        "Recall": (
            detected / total
            if total > 0 else 0
        )
    })

scenario_df = pd.DataFrame(
    scenario_rows
)

print(
    scenario_df.to_string(
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

for user, group in (
    data[
        data["network_ground_truth"] == 1
    ]
    .groupby("user")
):

    total = len(group)
    detected = int(group["predicted"].sum())

    user_rows.append({
        "User": user,
        "Malicious_Days": total,
        "Detected": detected,
        "Recall": (
            detected / total
            if total > 0 else 0
        )
    })

user_df = pd.DataFrame(
    user_rows
)

print(
    user_df.to_string(
        index=False,
        formatters={
            "Recall": "{:.4f}".format
        }
    )
)


# ============================================================
# SAVE EVALUATION
# ============================================================

evaluation_file = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "network_agent_v1_evaluation.csv"
)

results_df.to_csv(
    evaluation_file,
    index=False
)

print("\nEvaluation saved to:")
print(evaluation_file)


print("\n" + "=" * 70)
print("NETWORK AGENT V1 EVALUATION COMPLETE")
print("=" * 70)