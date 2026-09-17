from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "device_agent_results.parquet"
)

GROUND_TRUTH_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "device_ground_truth.parquet"
)


print("=" * 70)
print("CERT r4.1 - DEVICE AGENT EVALUATION")
print("=" * 70)


# ---------------------------------------------------------
# LOAD RESULTS
# ---------------------------------------------------------

print("\nLoading Device Agent results...")

results = pd.read_parquet(
    RESULTS_FILE,
    engine="fastparquet"
)

results["day"] = pd.to_datetime(results["day"]).dt.normalize()

print(f"Device Agent rows: {len(results):,}")


# ---------------------------------------------------------
# LOAD DEVICE GROUND TRUTH
# ---------------------------------------------------------

print("\nLoading device-specific ground truth...")

gt = pd.read_parquet(
    GROUND_TRUTH_FILE,
    engine="fastparquet"
)

gt["day"] = pd.to_datetime(gt["day"]).dt.normalize()

print(f"Ground-truth rows: {len(gt):,}")

print(
    f"Known malicious device user-days: "
    f"{gt['device_ground_truth'].sum():,}"
)


# ---------------------------------------------------------
# MERGE
# ---------------------------------------------------------

df = results.merge(
    gt[
        [
            "user",
            "day",
            "scenario",
            "device_ground_truth"
        ]
    ],
    on=["user", "day"],
    how="left"
)

df["device_ground_truth"] = (
    df["device_ground_truth"]
    .fillna(0)
    .astype(int)
)


# ---------------------------------------------------------
# EVALUATION FUNCTION
# ---------------------------------------------------------

def evaluate_threshold(df, threshold):

    y_true = df["device_ground_truth"].values

    y_pred = (
        df["device_agent_score"].values >= threshold
    ).astype(int)

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

    mcc_denominator = np.sqrt(
        (tp + fp)
        * (tp + fn)
        * (tn + fp)
        * (tn + fn)
    )

    mcc = (
        ((tp * tn) - (fp * fn)) / mcc_denominator
        if mcc_denominator > 0 else 0
    )

    return {
        "threshold": threshold,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "FPR": fpr,
        "FNR": fnr,
        "balanced_accuracy": balanced_accuracy,
        "MCC": mcc
    }


# ---------------------------------------------------------
# THRESHOLD ANALYSIS
# ---------------------------------------------------------

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


evaluation = []

for threshold in thresholds:

    evaluation.append(
        evaluate_threshold(
            df,
            threshold
        )
    )


eval_df = pd.DataFrame(evaluation)


# ---------------------------------------------------------
# DISPLAY
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("DEVICE AGENT THRESHOLD EVALUATION")
print("=" * 70)

print(
    eval_df.to_string(
        index=False,
        formatters={
            "accuracy": "{:.6f}".format,
            "precision": "{:.6f}".format,
            "recall": "{:.6f}".format,
            "f1": "{:.6f}".format,
            "specificity": "{:.6f}".format,
            "FPR": "{:.6f}".format,
            "FNR": "{:.6f}".format,
            "balanced_accuracy": "{:.6f}".format,
            "MCC": "{:.6f}".format
        }
    )
)


# ---------------------------------------------------------
# BEST F1
# ---------------------------------------------------------

best_f1 = eval_df.loc[
    eval_df["f1"].idxmax()
]

print("\n" + "=" * 70)
print("BEST F1 THRESHOLD")
print("=" * 70)

print(best_f1.to_string())


# ---------------------------------------------------------
# BEST PRECISION
# ---------------------------------------------------------

best_precision = eval_df.loc[
    eval_df["precision"].idxmax()
]

print("\n" + "=" * 70)
print("BEST PRECISION THRESHOLD")
print("=" * 70)

print(best_precision.to_string())


# ---------------------------------------------------------
# BEST RECALL
# ---------------------------------------------------------

best_recall = eval_df.loc[
    eval_df["recall"].idxmax()
]

print("\n" + "=" * 70)
print("BEST RECALL THRESHOLD")
print("=" * 70)

print(best_recall.to_string())


# ---------------------------------------------------------
# MALICIOUS DEVICE DAYS
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("MALICIOUS DEVICE DAYS")
print("=" * 70)

malicious = df[
    df["device_ground_truth"] == 1
].copy()

print(
    malicious[
        [
            "user",
            "day",
            "scenario",
            "device_agent_score",
            "device_agent_status",
            "device_explanation"
        ]
    ].sort_values(
        ["user", "day"]
    ).to_string(index=False)
)


# ---------------------------------------------------------
# SAVE EVALUATION
# ---------------------------------------------------------

output_file = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "device_agent_evaluation.csv"
)

eval_df.to_csv(
    output_file,
    index=False
)

print("\nEvaluation saved to:")
print(output_file)

print("\n" + "=" * 70)
print("DEVICE AGENT EVALUATION COMPLETE")
print("=" * 70)