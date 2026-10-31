import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
)


# ================================================================
# CONFIGURATION
# ================================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "outputs",
    "final_rule_agent_xgb_results.parquet"
)

THRESHOLD = 0.30


# ================================================================
# METRIC CALCULATION
# ================================================================

def calculate_metrics(y_true, y_pred):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

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

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_pred
    )

    mcc = matthews_corrcoef(
        y_true,
        y_pred
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

    return {
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


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("FINAL RULE AGENT - XGBOOST EVALUATION")
    print("=" * 70)

    # ------------------------------------------------------------
    # 1. LOAD FINAL RULE AGENT
    # ------------------------------------------------------------

    print("\nLoading final Rule Agent results...")

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Final Rule Agent file not found:\n{INPUT_FILE}"
        )

    df = pd.read_parquet(INPUT_FILE)

    print(f"Rows : {len(df):,}")
    print(f"Users: {df['user'].nunique():,}")

    # ------------------------------------------------------------
    # 2. CHECK REQUIRED COLUMNS
    # ------------------------------------------------------------

    required_columns = [
        "ground_truth",
        "xgb_probability",
        "xgb_prediction",
        "user",
        "day",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing)
        )

    # ------------------------------------------------------------
    # 3. PREPARE GROUND TRUTH
    # ------------------------------------------------------------

    y_true = (
        pd.to_numeric(
            df["ground_truth"],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )

    # Use the probability and explicitly apply
    # the selected final threshold.
    probabilities = pd.to_numeric(
        df["xgb_probability"],
        errors="coerce"
    ).fillna(0)

    y_pred = (
        probabilities >= THRESHOLD
    ).astype(int)

    # ------------------------------------------------------------
    # 4. GROUND-TRUTH DISTRIBUTION
    # ------------------------------------------------------------

    print("\nGround-truth distribution:")
    print(y_true.value_counts())

    # ------------------------------------------------------------
    # 5. PREDICTION DISTRIBUTION
    # ------------------------------------------------------------

    print("\nPrediction distribution:")
    print(
        y_pred.value_counts()
    )

    # ------------------------------------------------------------
    # 6. FINAL METRICS
    # ------------------------------------------------------------

    metrics = calculate_metrics(
        y_true,
        y_pred
    )

    print("\n" + "=" * 70)
    print("FINAL RULE AGENT RESULTS")
    print("=" * 70)

    print(
        f"\nThreshold: {THRESHOLD:.2f}"
    )

    print("\nConfusion Matrix:")
    print(f"TP : {metrics['TP']}")
    print(f"TN : {metrics['TN']}")
    print(f"FP : {metrics['FP']}")
    print(f"FN : {metrics['FN']}")

    print("\nMetrics:")
    print(
        f"Accuracy           : "
        f"{metrics['Accuracy']:.6f}"
    )

    print(
        f"Precision          : "
        f"{metrics['Precision']:.6f}"
    )

    print(
        f"Recall             : "
        f"{metrics['Recall']:.6f}"
    )

    print(
        f"F1                 : "
        f"{metrics['F1']:.6f}"
    )

    print(
        f"Specificity        : "
        f"{metrics['Specificity']:.6f}"
    )

    print(
        f"FPR                : "
        f"{metrics['FPR']:.6f}"
    )

    print(
        f"FNR                : "
        f"{metrics['FNR']:.6f}"
    )

    print(
        f"Balanced Accuracy  : "
        f"{metrics['Balanced_Accuracy']:.6f}"
    )

    print(
        f"MCC                : "
        f"{metrics['MCC']:.6f}"
    )

    # ------------------------------------------------------------
    # 7. MALICIOUS DAYS
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("ACTUAL MALICIOUS DAYS")
    print("=" * 70)

    malicious = df[
        y_true == 1
    ].copy()

    malicious["evaluation_prediction"] = y_pred[
        y_true == 1
    ]

    malicious["evaluation_probability"] = probabilities[
        y_true == 1
    ]

    columns_to_show = [
        "user",
        "day",
        "ground_truth",
        "xgb_probability",
        "xgb_prediction",
        "final_rule_score",
        "final_threat_level",
    ]

    columns_to_show = [
        col
        for col in columns_to_show
        if col in malicious.columns
    ]

    print(
        malicious[
            columns_to_show
        ].sort_values(
            "xgb_probability",
            ascending=False
        ).to_string(index=False)
    )

    # ------------------------------------------------------------
    # 8. FALSE POSITIVES
    # ------------------------------------------------------------

    false_positives = df[
        (y_true == 0) &
        (y_pred == 1)
    ].copy()

    print("\n" + "=" * 70)
    print("FALSE POSITIVES")
    print("=" * 70)

    print(
        f"False positives: "
        f"{len(false_positives)}"
    )

    if len(false_positives) > 0:

        fp_columns = [
            "user",
            "day",
            "xgb_probability",
            "final_rule_score",
            "final_threat_level",
        ]

        fp_columns = [
            col
            for col in fp_columns
            if col in false_positives.columns
        ]

        print(
            false_positives[
                fp_columns
            ].sort_values(
                "xgb_probability",
                ascending=False
            ).to_string(index=False)
        )

    # ------------------------------------------------------------
    # 9. MISSED MALICIOUS DAYS
    # ------------------------------------------------------------

    false_negatives = df[
        (y_true == 1) &
        (y_pred == 0)
    ].copy()

    print("\n" + "=" * 70)
    print("MISSED MALICIOUS DAYS")
    print("=" * 70)

    print(
        f"False negatives: "
        f"{len(false_negatives)}"
    )

    if len(false_negatives) > 0:

        fn_columns = [
            "user",
            "day",
            "ground_truth",
            "xgb_probability",
            "final_rule_score",
            "final_threat_level",
        ]

        fn_columns = [
            col
            for col in fn_columns
            if col in false_negatives.columns
        ]

        print(
            false_negatives[
                fn_columns
            ].sort_values(
                "xgb_probability",
                ascending=False
            ).to_string(index=False)
        )

    # ------------------------------------------------------------
    # 10. COMPLETE
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL RULE AGENT EVALUATION COMPLETE")
    print("=" * 70)