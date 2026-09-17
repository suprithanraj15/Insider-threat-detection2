from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    balanced_accuracy_score,
    matthews_corrcoef
)
def print_metrics(title, metrics):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    print(f"TP                 : {metrics['TP']}")
    print(f"TN                 : {metrics['TN']}")
    print(f"FP                 : {metrics['FP']}")
    print(f"FN                 : {metrics['FN']}")

    print(f"\nAccuracy           : {metrics['Accuracy']:.4f}")
    print(f"Precision          : {metrics['Precision']:.4f}")
    print(f"Recall             : {metrics['Recall']:.4f}")
    print(f"F1 Score           : {metrics['F1']:.4f}")
    print(
        f"Balanced Accuracy  : "
        f"{metrics['Balanced_Accuracy']:.4f}"
    )
    print(
        f"Specificity        : "
        f"{metrics['Specificity']:.4f}"
    )
    print(
        f"False Positive Rate: "
        f"{metrics['FPR']:.4f}"
    )
    print(
        f"False Negative Rate: "
        f"{metrics['FNR']:.4f}"
    )
    print(
        f"MCC                : "
        f"{metrics['MCC']:.4f}"
    )


def evaluate_login_agent(y_true, y_pred):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    metrics = {}

    metrics["TP"] = tp
    metrics["TN"] = tn
    metrics["FP"] = fp
    metrics["FN"] = fn

    metrics["Accuracy"] = accuracy_score(
        y_true,
        y_pred
    )

    metrics["Precision"] = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    metrics["Recall"] = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    metrics["F1"] = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    metrics["Balanced_Accuracy"] = balanced_accuracy_score(
        y_true,
        y_pred
    )

    metrics["Specificity"] = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    metrics["FPR"] = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    metrics["FNR"] = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else 0
    )

    metrics["MCC"] = matthews_corrcoef(
        y_true,
        y_pred
    )

    return metrics
# ============================================================
# MAIN
# ============================================================

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOGIN_AGENT_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "login_agent_results.parquet"
)

LOGIN_GROUND_TRUTH_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "login_ground_truth.parquet"
)

OUTPUT_THRESHOLD_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "login_agent_threshold_analysis.csv"
)


def main():

    print("=" * 70)
    print("LOGIN AGENT V2 - LOGIN-SPECIFIC EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. LOAD LOGIN AGENT RESULTS
    # --------------------------------------------------------

    agent = pd.read_parquet(
        LOGIN_AGENT_FILE
    )

    print(
        f"\nLogin Agent rows : {len(agent):,}"
    )

    # --------------------------------------------------------
    # 2. LOAD LOGIN-SPECIFIC GROUND TRUTH
    # --------------------------------------------------------

    truth = pd.read_parquet(
        LOGIN_GROUND_TRUTH_FILE
    )

    print(
        f"Ground truth rows: {len(truth):,}"
    )

    # --------------------------------------------------------
    # 3. NORMALIZE USER AND DAY
    # --------------------------------------------------------

    agent["user"] = (
        agent["user"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    truth["user"] = (
        truth["user"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    agent["day"] = pd.to_datetime(
        agent["day"],
        errors="coerce"
    ).dt.normalize()

    truth["day"] = pd.to_datetime(
        truth["day"],
        errors="coerce"
    ).dt.normalize()

    print(
        f"\nInvalid Login Agent dates : "
        f"{agent['day'].isna().sum()}"
    )

    print(
        f"Invalid Ground Truth dates: "
        f"{truth['day'].isna().sum()}"
    )

    # --------------------------------------------------------
    # 4. VERIFY LOGIN GROUND TRUTH
    # --------------------------------------------------------

    print(
        "\nLogin-specific ground truth distribution:"
    )

    print(
        truth["login_ground_truth"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # 5. MERGE
    # --------------------------------------------------------

    merged = agent.merge(
        truth[
            [
                "user",
                "day",
                "login_ground_truth",
                "scenario"
            ]
        ],
        on=[
            "user",
            "day"
        ],
        how="inner"
    )

    print(
        f"\nMerged rows      : {len(merged):,}"
    )

    if len(merged) != len(agent):

        raise ValueError(
            "ERROR: Some Login Agent rows "
            "did not match the login ground truth."
        )

    print(
        "SUCCESS: All Login Agent rows "
        "matched the login ground truth."
    )

    # --------------------------------------------------------
    # 6. CURRENT LOGIN AGENT PREDICTION
    # --------------------------------------------------------

    merged["prediction"] = (
        merged["login_agent_status"]
        .isin(
            [
                "MODERATE_EVIDENCE",
                "STRONG_EVIDENCE"
            ]
        )
        .astype(int)
    )

    y_true = merged[
        "login_ground_truth"
    ]

    y_pred = merged[
        "prediction"
    ]

    metrics = evaluate_login_agent(
        y_true,
        y_pred
    )

    print_metrics(
        "CURRENT LOGIN AGENT V2 RESULTS",
        metrics
    )

    # --------------------------------------------------------
    # 7. PREDICTION DISTRIBUTION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CURRENT LOGIN AGENT PREDICTION DISTRIBUTION")
    print("=" * 70)

    prediction_labels = {
        0: "NO ALERT",
        1: "LOGIN ALERT"
    }

    prediction_distribution = (
        merged["prediction"]
        .map(prediction_labels)
        .value_counts()
    )

    print(
        prediction_distribution.to_string()
    )

    # --------------------------------------------------------
    # 8. LOGIN AGENT STATUS DISTRIBUTION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LOGIN AGENT STATUS DISTRIBUTION")
    print("=" * 70)

    print(
        merged["login_agent_status"]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # 9. THRESHOLD ANALYSIS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LOGIN EVIDENCE THRESHOLD ANALYSIS")
    print("=" * 70)

    threshold_results = []

    max_evidence = int(
        merged["login_evidence_count"].max()
    )

    for threshold in range(
        0,
        max_evidence + 1
    ):

        threshold_pred = (
            merged["login_evidence_count"]
            >= threshold
        ).astype(int)

        threshold_metrics = evaluate_login_agent(
            y_true,
            threshold_pred
        )

        threshold_metrics[
            "Threshold"
        ] = f">= {threshold}"

        threshold_metrics[
            "Alerts"
        ] = int(
            threshold_pred.sum()
        )

        threshold_results.append(
            threshold_metrics
        )

        print(
            f"\nThreshold >= {threshold}"
        )

        print(
            f"  Alerts    : "
            f"{threshold_metrics['Alerts']:,}"
        )

        print(
            f"  TP        : "
            f"{threshold_metrics['TP']}"
        )

        print(
            f"  FP        : "
            f"{threshold_metrics['FP']}"
        )

        print(
            f"  FN        : "
            f"{threshold_metrics['FN']}"
        )

        print(
            f"  Accuracy  : "
            f"{threshold_metrics['Accuracy']:.4f}"
        )

        print(
            f"  Precision : "
            f"{threshold_metrics['Precision']:.4f}"
        )

        print(
            f"  Recall    : "
            f"{threshold_metrics['Recall']:.4f}"
        )

        print(
            f"  F1        : "
            f"{threshold_metrics['F1']:.4f}"
        )

        print(
            f"  FPR       : "
            f"{threshold_metrics['FPR']:.4f}"
        )

        print(
            f"  MCC       : "
            f"{threshold_metrics['MCC']:.4f}"
        )

    # --------------------------------------------------------
    # 10. SAVE THRESHOLD RESULTS
    # --------------------------------------------------------

    threshold_df = pd.DataFrame(
        threshold_results
    )

    threshold_df.to_csv(
        OUTPUT_THRESHOLD_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print("THRESHOLD ANALYSIS SAVED")
    print("=" * 70)

    print(
        OUTPUT_THRESHOLD_FILE
    )

    # --------------------------------------------------------
    # 11. BEST F1 THRESHOLD
    # --------------------------------------------------------

    best_row = threshold_df.loc[
        threshold_df["F1"].idxmax()
    ]

    print("\n" + "=" * 70)
    print("BEST F1 THRESHOLD")
    print("=" * 70)

    print(
        f"Threshold : {best_row['Threshold']}"
    )

    print(
        f"Accuracy  : {best_row['Accuracy']:.4f}"
    )

    print(
        f"Precision : {best_row['Precision']:.4f}"
    )

    print(
        f"Recall    : {best_row['Recall']:.4f}"
    )

    print(
        f"F1        : {best_row['F1']:.4f}"
    )

    print(
        f"MCC       : {best_row['MCC']:.4f}"
    )

    # --------------------------------------------------------
    # 12. SCENARIO-WISE LOGIN DETECTION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SCENARIO-WISE MALICIOUS LOGIN DETECTION")
    print("=" * 70)

    malicious_rows = merged[
        merged["login_ground_truth"] == 1
    ].copy()

    for scenario in sorted(
        malicious_rows["scenario"]
        .dropna()
        .unique()
    ):

        scenario_rows = malicious_rows[
            malicious_rows["scenario"] == scenario
        ]

        detected = int(
            scenario_rows["prediction"].sum()
        )

        total = len(
            scenario_rows
        )

        percentage = (
            detected / total * 100
            if total > 0
            else 0
        )

        print(
            f"Scenario {int(scenario)}: "
            f"{detected}/{total} "
            f"({percentage:.2f}%)"
        )

    # --------------------------------------------------------
    # 13. USER-WISE LOGIN DETECTION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MALICIOUS USER LOGIN DETECTION")
    print("=" * 70)

    for user in sorted(
        malicious_rows["user"].unique()
    ):

        user_rows = malicious_rows[
            malicious_rows["user"] == user
        ]

        detected = int(
            user_rows["prediction"].sum()
        )

        total = len(
            user_rows
        )

        percentage = (
            detected / total * 100
            if total > 0
            else 0
        )

        print(
            f"{user.upper()}: "
            f"{detected}/{total} "
            f"({percentage:.2f}%)"
        )

    # --------------------------------------------------------
    # 14. LOGIN SIGNAL COUNTS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LOGIN SIGNAL COUNTS")
    print("=" * 70)

    signal_columns = [
        column
        for column in merged.columns
        if column.startswith("login_")
        and column.endswith("_signal")
    ]

    for signal in signal_columns:

        print(
            f"{signal:<40}: "
            f"{int(merged[signal].sum()):,}"
        )

    # --------------------------------------------------------
    # 15. FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LOGIN AGENT V2 FINAL SUMMARY")
    print("=" * 70)

    print(
        f"TP                  : {metrics['TP']}"
    )

    print(
        f"TN                  : {metrics['TN']}"
    )

    print(
        f"FP                  : {metrics['FP']}"
    )

    print(
        f"FN                  : {metrics['FN']}"
    )

    print(
        f"\nAccuracy            : "
        f"{metrics['Accuracy']:.4f}"
    )

    print(
        f"Precision           : "
        f"{metrics['Precision']:.4f}"
    )

    print(
        f"Recall              : "
        f"{metrics['Recall']:.4f}"
    )

    print(
        f"F1 Score            : "
        f"{metrics['F1']:.4f}"
    )

    print(
        f"Balanced Accuracy   : "
        f"{metrics['Balanced_Accuracy']:.4f}"
    )

    print(
        f"Specificity         : "
        f"{metrics['Specificity']:.4f}"
    )

    print(
        f"False Positive Rate : "
        f"{metrics['FPR']:.4f}"
    )

    print(
        f"False Negative Rate : "
        f"{metrics['FNR']:.4f}"
    )

    print(
        f"MCC                 : "
        f"{metrics['MCC']:.4f}"
    )

    print("\n" + "=" * 70)
    print(
        "LOGIN AGENT V2 LOGIN-SPECIFIC "
        "EVALUATION COMPLETED"
    )
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
