from pathlib import Path

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


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

AGENT_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "login_agent_results.parquet"
)

GROUND_TRUTH_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "login_ground_truth.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "login_candidate_rule_results.csv"
)


# ============================================================
# METRICS
# ============================================================

def evaluate_rule(y_true, y_pred):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Accuracy": accuracy_score(
            y_true,
            y_pred
        ),
        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "Recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "F1": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "Balanced_Accuracy": balanced_accuracy_score(
            y_true,
            y_pred
        ),
        "FPR": (
            fp / (fp + tn)
            if (fp + tn) > 0
            else 0
        ),
        "MCC": matthews_corrcoef(
            y_true,
            y_pred
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("LOGIN AGENT V2 - CANDIDATE RULE EXPERIMENT")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    agent = pd.read_parquet(
        AGENT_FILE
    )

    truth = pd.read_parquet(
        GROUND_TRUTH_FILE
    )

    # --------------------------------------------------------
    # 2. NORMALIZE KEYS
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

    # --------------------------------------------------------
    # 3. MERGE
    # --------------------------------------------------------

    df = agent.merge(
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
        f"\nMerged rows: {len(df):,}"
    )

    y_true = df[
        "login_ground_truth"
    ].astype(int)

    # --------------------------------------------------------
    # 4. DEFINE SIGNALS
    # --------------------------------------------------------

    zscore = (
        df["login_zscore_signal"] == 1
    )

    strong_zscore = (
        df["login_strong_zscore_signal"] == 1
    )

    after_hours = (
        df["login_after_hours_signal"] == 1
    )

    multiple_pc = (
        df["login_multiple_pc_signal"] == 1
    )

    high_volume = (
        df["login_high_volume_signal"] == 1
    )

    after_hours_ratio = (
        df["login_after_hours_ratio_signal"] == 1
    )

    after_hours_anomaly = (
        df["login_after_hours_anomaly_signal"] == 1
    )

    temporal_change = (
        df["login_temporal_change_signal"] == 1
    )

    behavior_change = (
        df["login_behavior_change_signal"] == 1
    )

    # --------------------------------------------------------
    # 5. CANDIDATE RULES
    # --------------------------------------------------------

    rules = {}

    # Rule 1:
    # Strong statistical anomaly
    rules[
        "R1_Strong_ZScore"
    ] = strong_zscore

    # Rule 2:
    # Z-score + after-hours
    rules[
        "R2_ZScore_AfterHours"
    ] = (
        zscore
        & after_hours
    )

    # Rule 3:
    # Strong z-score OR after-hours anomaly
    rules[
        "R3_StrongAnomaly"
    ] = (
        strong_zscore
        | after_hours_anomaly
    )

    # Rule 4:
    # Multiple PCs + high volume + after-hours
    rules[
        "R4_MultiPC_HighVolume_AfterHours"
    ] = (
        multiple_pc
        & high_volume
        & after_hours
    )

    # Rule 5:
    # Multiple PCs + high volume
    rules[
        "R5_MultiPC_HighVolume"
    ] = (
        multiple_pc
        & high_volume
    )

    # Rule 6:
    # Z-score + multiple PCs
    rules[
        "R6_ZScore_MultiPC"
    ] = (
        zscore
        & multiple_pc
    )

    # Rule 7:
    # Z-score + high volume
    rules[
        "R7_ZScore_HighVolume"
    ] = (
        zscore
        & high_volume
    )

    # Rule 8:
    # After-hours + multiple PCs
    rules[
        "R8_AfterHours_MultiPC"
    ] = (
        after_hours
        & multiple_pc
    )

    # Rule 9:
    # Strong anomaly OR
    # multi-PC + high volume + after-hours
    rules[
        "R9_StrongAnomaly_OR_Context"
    ] = (
        strong_zscore
        | (
            multiple_pc
            & high_volume
            & after_hours
        )
    )

    # Rule 10:
    # Strong anomaly OR
    # z-score + after-hours
    rules[
        "R10_StrongAnomaly_OR_ZScoreAfterHours"
    ] = (
        strong_zscore
        | (
            zscore
            & after_hours
        )
    )

    # Rule 11:
    # Strong anomaly OR
    # z-score + multiple PCs
    rules[
        "R11_StrongAnomaly_OR_ZScoreMultiPC"
    ] = (
        strong_zscore
        | (
            zscore
            & multiple_pc
        )
    )

    # Rule 12:
    # Strong anomaly OR
    # after-hours anomaly
    rules[
        "R12_StrongZ_OR_AfterHoursAnomaly"
    ] = (
        strong_zscore
        | after_hours_anomaly
    )

    # Rule 13:
    # Three contextual signals together
    rules[
        "R13_Context_3Signals"
    ] = (
        after_hours
        & multiple_pc
        & high_volume
    )

    # Rule 14:
    # Strong anomaly OR
    # all four context signals
    rules[
        "R14_StrongAnomaly_OR_AllContext"
    ] = (
        strong_zscore
        | (
            after_hours
            & multiple_pc
            & high_volume
            & after_hours_ratio
        )
    )

    # Rule 15:
    # Existing V2 evidence threshold >= 2
    rules[
        "R15_Current_V2"
    ] = (
        df["login_evidence_count"] >= 2
    )

    # --------------------------------------------------------
    # 6. EVALUATE RULES
    # --------------------------------------------------------

    results = []

    print("\n" + "=" * 80)
    print("CANDIDATE RULE RESULTS")
    print("=" * 80)

    for rule_name, rule_prediction in rules.items():

        y_pred = (
            rule_prediction
            .astype(int)
        )

        metrics = evaluate_rule(
            y_true,
            y_pred
        )

        metrics["Rule"] = rule_name

        metrics["Alerts"] = int(
            y_pred.sum()
        )

        results.append(
            metrics
        )

        print(
            f"\n{rule_name}"
        )

        print(
            f"  Alerts    : "
            f"{metrics['Alerts']:,}"
        )

        print(
            f"  TP        : "
            f"{metrics['TP']}"
        )

        print(
            f"  FP        : "
            f"{metrics['FP']}"
        )

        print(
            f"  FN        : "
            f"{metrics['FN']}"
        )

        print(
            f"  Precision : "
            f"{metrics['Precision']:.6f}"
        )

        print(
            f"  Recall    : "
            f"{metrics['Recall']:.4f}"
        )

        print(
            f"  F1        : "
            f"{metrics['F1']:.6f}"
        )

        print(
            f"  FPR       : "
            f"{metrics['FPR']:.6f}"
        )

        print(
            f"  MCC       : "
            f"{metrics['MCC']:.6f}"
        )

    # --------------------------------------------------------
    # 7. SAVE RESULTS
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df[
        [
            "Rule",
            "Alerts",
            "TP",
            "FP",
            "FN",
            "TN",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "Balanced_Accuracy",
            "FPR",
            "MCC"
        ]
    ]

    results_df = results_df.sort_values(
        [
            "F1",
            "Precision"
        ],
        ascending=False
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # 8. BEST RULES
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("RULES SORTED BY F1")
    print("=" * 80)

    print(
        results_df[
            [
                "Rule",
                "Alerts",
                "TP",
                "FP",
                "FN",
                "Precision",
                "Recall",
                "F1",
                "FPR",
                "MCC"
            ]
        ].to_string(
            index=False
        )
    )

    print("\n" + "=" * 80)
    print("CANDIDATE RULE ANALYSIS SAVED")
    print("=" * 80)

    print(
        OUTPUT_FILE
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()