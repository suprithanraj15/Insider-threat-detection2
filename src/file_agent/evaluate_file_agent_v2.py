from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

FILE_AGENT_RESULTS = (
    PROJECT_ROOT / "data" / "outputs" / "file_agent_v2_results.parquet"
)

FILE_GROUND_TRUTH = (
    PROJECT_ROOT / "data" / "outputs" / "file_ground_truth.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "outputs" / "file_agent_v2_evaluation.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FILE AGENT V2 - FILE-SPECIFIC EVALUATION")
print("=" * 70)

print("\nLoading File Agent V2 results...")

agent_df = pd.read_parquet(
    FILE_AGENT_RESULTS,
    engine="fastparquet"
)

print(f"Agent rows: {len(agent_df):,}")


print("\nLoading file-specific ground truth...")

gt_df = pd.read_parquet(
    FILE_GROUND_TRUTH,
    engine="fastparquet"
)

print(f"Ground-truth rows: {len(gt_df):,}")


# ============================================================
# NORMALIZE KEYS
# ============================================================

agent_df["user"] = (
    agent_df["user"]
    .astype(str)
    .str.strip()
    .str.lower()
)

gt_df["user"] = (
    gt_df["user"]
    .astype(str)
    .str.strip()
    .str.lower()
)

agent_df["day"] = pd.to_datetime(
    agent_df["day"],
    errors="coerce"
).dt.normalize()

gt_df["day"] = pd.to_datetime(
    gt_df["day"],
    errors="coerce"
).dt.normalize()


# ============================================================
# BUILD BINARY FILE GROUND TRUTH
# ============================================================

gt_keys = set(
    zip(
        gt_df["user"],
        gt_df["day"]
    )
)


agent_df["file_ground_truth"] = [
    1 if (user, day) in gt_keys else 0
    for user, day in zip(
        agent_df["user"],
        agent_df["day"]
    )
]


# ============================================================
# VERIFY GROUND TRUTH
# ============================================================

print("\n" + "=" * 70)
print("GROUND-TRUTH VERIFICATION")
print("=" * 70)

print(
    f"Malicious FILE user-days: "
    f"{agent_df['file_ground_truth'].sum():,}"
)

print(
    f"Normal user-days: "
    f"{(agent_df['file_ground_truth'] == 0).sum():,}"
)


# ============================================================
# SCORE
# ============================================================

score_column = "file_agent_score"

if score_column not in agent_df.columns:
    raise ValueError(
        f"Missing required column: {score_column}"
    )


agent_df[score_column] = pd.to_numeric(
    agent_df[score_column],
    errors="coerce"
).fillna(0)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_at_threshold(df, threshold):

    y_true = df["file_ground_truth"].values

    y_pred = (
        df[score_column].values >= threshold
    ).astype(int)

    tp = int(
        np.sum(
            (y_true == 1) &
            (y_pred == 1)
        )
    )

    tn = int(
        np.sum(
            (y_true == 0) &
            (y_pred == 0)
        )
    )

    fp = int(
        np.sum(
            (y_true == 0) &
            (y_pred == 1)
        )
    )

    fn = int(
        np.sum(
            (y_true == 1) &
            (y_pred == 0)
        )
    )

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
        2 * precision * recall /
        (precision + recall)
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

    return {
        "threshold": threshold,
        "alerts": int(y_pred.sum()),
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
    }


# ============================================================
# THRESHOLD EVALUATION
# ============================================================

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
    95,
]


results = []

for threshold in thresholds:

    result = evaluate_at_threshold(
        agent_df,
        threshold
    )

    results.append(result)


evaluation_df = pd.DataFrame(results)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD RESULTS")
print("=" * 70)

display_columns = [
    "threshold",
    "alerts",
    "TP",
    "TN",
    "FP",
    "FN",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "FPR",
]

print(
    evaluation_df[
        display_columns
    ].to_string(
        index=False,
        formatters={
            "accuracy": "{:.6f}".format,
            "precision": "{:.6f}".format,
            "recall": "{:.6f}".format,
            "f1": "{:.6f}".format,
            "FPR": "{:.6f}".format,
        }
    )
)


# ============================================================
# BEST F1
# ============================================================

best_f1 = evaluation_df.loc[
    evaluation_df["f1"].idxmax()
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

best_precision = evaluation_df.loc[
    evaluation_df["precision"].idxmax()
]

print("\n" + "=" * 70)
print("BEST PRECISION THRESHOLD")
print("=" * 70)

print(
    best_precision.to_string()
)


# ============================================================
# MALICIOUS FILE USER-DAY
# ============================================================

print("\n" + "=" * 70)
print("MALICIOUS FILE USER-DAY")
print("=" * 70)

malicious_rows = agent_df[
    agent_df["file_ground_truth"] == 1
]

print(
    malicious_rows[
        [
            "user",
            "day",
            score_column,
            "file_agent_status",
            "file_explanation",
        ]
    ].to_string(index=False)
)


# ============================================================
# SAVE
# ============================================================

evaluation_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved evaluation to:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("FILE AGENT V2 EVALUATION COMPLETE")
print("=" * 70)