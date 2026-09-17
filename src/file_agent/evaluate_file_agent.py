from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    balanced_accuracy_score,
    matthews_corrcoef,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"

AGENT_FILE = OUTPUT_DIR / "file_agent_results.parquet"
GT_FILE = OUTPUT_DIR / "user_day_features_labeled.parquet"


print("=" * 80)
print("FILE AGENT EVALUATION")
print("=" * 80)


# ================================================================
# 1. LOAD
# ================================================================

agent = pd.read_parquet(
    AGENT_FILE,
    engine="fastparquet"
)

gt = pd.read_parquet(
    GT_FILE,
    engine="fastparquet"
)


# ================================================================
# 2. NORMALIZE KEYS
# ================================================================

agent["user"] = (
    agent["user"]
    .astype(str)
    .str.lower()
    .str.strip()
)

gt["user"] = (
    gt["user"]
    .astype(str)
    .str.lower()
    .str.strip()
)

agent["day"] = pd.to_datetime(agent["day"]).dt.normalize()
gt["day"] = pd.to_datetime(gt["day"]).dt.normalize()


# ================================================================
# 3. MERGE GROUND TRUTH
# ================================================================

data = agent.merge(
    gt[
        [
            "user",
            "day",
            "ground_truth",
            "scenario",
        ]
    ],
    on=["user", "day"],
    how="left"
)


data["ground_truth"] = (
    data["ground_truth"]
    .fillna(0)
    .astype(int)
)


print(f"\nAgent rows       : {len(agent):,}")
print(f"Ground truth rows: {len(gt):,}")
print(f"Evaluation rows  : {len(data):,}")


# ================================================================
# 4. THRESHOLD EVALUATION
# ================================================================

print("\n" + "=" * 80)
print("THRESHOLD ANALYSIS")
print("=" * 80)


thresholds = [
    0,
    10,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
]


results = []


for threshold in thresholds:

    prediction = (
        data["file_agent_score"] >= threshold
    ).astype(int)

    y_true = data["ground_truth"]

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        prediction,
        labels=[0, 1]
    ).ravel()

    accuracy = accuracy_score(
        y_true,
        prediction
    )

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

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    fpr = 1 - specificity

    balanced_acc = balanced_accuracy_score(
        y_true,
        prediction
    )

    mcc = matthews_corrcoef(
        y_true,
        prediction
    )

    results.append(
        {
            "threshold": threshold,
            "alerts": int(prediction.sum()),
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
            "fpr": fpr,
            "balanced_accuracy": balanced_acc,
            "mcc": mcc,
        }
    )


results_df = pd.DataFrame(results)


print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ================================================================
# 5. BEST F1
# ================================================================

best_f1 = results_df.loc[
    results_df["f1"].idxmax()
]

print("\n" + "=" * 80)
print("BEST F1 THRESHOLD")
print("=" * 80)

print(best_f1.to_string())


# ================================================================
# 6. BEST PRECISION
# ================================================================

best_precision = results_df.loc[
    results_df["precision"].idxmax()
]

print("\n" + "=" * 80)
print("BEST PRECISION THRESHOLD")
print("=" * 80)

print(best_precision.to_string())


# ================================================================
# 7. SCENARIO ANALYSIS
# ================================================================

print("\n" + "=" * 80)
print("SCENARIO ANALYSIS")
print("=" * 80)


# Use threshold with best F1
best_threshold = best_f1["threshold"]


data["prediction"] = (
    data["file_agent_score"] >= best_threshold
).astype(int)


malicious = data[
    data["ground_truth"] == 1
]


for scenario in sorted(
    malicious["scenario"].dropna().unique()
):

    scenario_data = malicious[
        malicious["scenario"] == scenario
    ]

    detected = int(
        scenario_data["prediction"].sum()
    )

    total = len(scenario_data)

    print(
        f"Scenario {int(scenario)}: "
        f"{detected}/{total} malicious user-days detected"
    )


# ================================================================
# 8. MALICIOUS DAYS
# ================================================================

print("\n" + "=" * 80)
print("MALICIOUS USER-DAY RESULTS")
print("=" * 80)


print(
    data[
        data["ground_truth"] == 1
    ][
        [
            "user",
            "day",
            "scenario",
            "file_agent_score",
            "file_agent_status",
            "file_evidence_count",
            "file_explanation",
        ]
    ]
    .sort_values(
        ["scenario", "user", "day"]
    )
    .to_string(index=False)
)


print("\n" + "=" * 80)
print("FILE AGENT EVALUATION COMPLETE")
print("=" * 80)