from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
)

from xgboost import XGBClassifier


# ============================================================
# PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "final_rule_agent_results.parquet"
)


# ============================================================
# LOAD CURRENT RULE AGENT RESULTS
# ============================================================

print("=" * 70)
print("RULE AGENT - XGBOOST EXPERIMENT")
print("=" * 70)

print("\nLoading current Rule Agent results...")

df = pd.read_parquet(INPUT_FILE)

print(f"Rows : {len(df):,}")
print(f"Users: {df['user'].nunique():,}")


# ============================================================
# TARGET
# ============================================================

y = df["ground_truth"].astype(int)

print("\nGround-truth distribution:")
print(y.value_counts().sort_index())


# ============================================================
# FEATURES
# ============================================================

feature_columns = [
    # Primary rule signals
    "rule_high_activity",
    "rule_unusual_logon",
    "rule_unusual_device",
    "rule_unusual_file",
    "rule_unusual_email",
    "rule_unusual_web",
    "rule_behavior_deviation",

    # Supporting rule signals
    "rule_after_hours_login",
    "rule_device_connection_pattern",
    "rule_multiple_logon_pcs",
    "rule_multiple_device_pcs",
    "rule_file_type_activity",

    # Rule counts
    "primary_rule_count",
    "supporting_rule_count",
    "rule_alert_count",

    # Behavioral measurements
    "total_activity",
    "active_channels",
    "device_activity_ratio",
    "file_activity_ratio",
    "file_diversity",
    "email_activity_ratio",
    "avg_email_size",
    "avg_recipients_per_email",
    "web_activity_ratio",
    "url_diversity",
    "domain_diversity",

    # Clean behavioral z-scores
    "total_activity_clean_zscore",
    "logon_events_clean_zscore",
    "device_events_clean_zscore",
    "file_events_clean_zscore",
    "emails_sent_clean_zscore",
    "web_visits_clean_zscore",
    "clean_behavior_deviation",

    # Context features
    "after_hours_logons",
    "connect_events",
    "connect_disconnect_ratio",
    "unique_logon_pcs",
    "unique_device_pcs",
    "unique_file_extensions",
]


# ============================================================
# VERIFY FEATURES
# ============================================================

missing = [
    col for col in feature_columns
    if col not in df.columns
]

if missing:
    raise KeyError(
        "Required feature columns are missing:\n"
        + "\n".join(missing)
    )

print(f"\nFeatures used: {len(feature_columns)}")


# ============================================================
# BUILD FEATURE MATRIX
# ============================================================

X = df[feature_columns].copy()

for col in X.columns:
    if X[col].dtype == bool:
        X[col] = X[col].astype(int)

X = X.replace([np.inf, -np.inf], np.nan)

X = X.fillna(0)


# ============================================================
# STRATIFIED TRAIN / VALIDATION / TEST SPLIT
#
# IMPORTANT:
# The dataset contains only 32 malicious rows.
# Stratification ensures both classes are represented
# in validation and test sets.
# ============================================================

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.25,
    random_state=42,
    stratify=y_train_val,
)


# ============================================================
# SPLIT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DATA SPLIT")
print("=" * 70)

print(f"\nTrain rows      : {len(X_train):,}")
print(f"Validation rows : {len(X_val):,}")
print(f"Test rows       : {len(X_test):,}")

print("\nTrain positives:")
print(int(y_train.sum()))

print("Validation positives:")
print(int(y_val.sum()))

print("Test positives:")
print(int(y_test.sum()))


# ============================================================
# CLASS IMBALANCE
# ============================================================

negative_count = int((y_train == 0).sum())
positive_count = int((y_train == 1).sum())

if positive_count == 0:
    raise ValueError(
        "Training split contains zero malicious samples."
    )

scale_pos_weight = (
    negative_count / positive_count
)

print(
    f"\nXGBoost scale_pos_weight: "
    f"{scale_pos_weight:.4f}"
)


# ============================================================
# XGBOOST MODEL
# ============================================================

model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
)


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("TRAINING XGBOOST")
print("=" * 70)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)

print("Training completed.")


# ============================================================
# TEST PROBABILITIES
# ============================================================

probabilities = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD ANALYSIS")
print("=" * 70)

thresholds = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
]


results = []


for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        predictions,
    )

    mcc = matthews_corrcoef(
        y_test,
        predictions,
    )

    results.append(
        {
            "threshold": threshold,
            "alerts": int(predictions.sum()),
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "balanced_accuracy": balanced_accuracy,
            "mcc": mcc,
        }
    )


results_df = pd.DataFrame(results)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# BEST THRESHOLDS
# ============================================================

print("\n" + "=" * 70)
print("BEST THRESHOLDS")
print("=" * 70)


best_f1 = results_df.loc[
    results_df["f1"].idxmax()
]

best_balanced = results_df.loc[
    results_df["balanced_accuracy"].idxmax()
]

best_mcc = results_df.loc[
    results_df["mcc"].idxmax()
]


print("\nBest F1 threshold:")
print(best_f1.to_string())

print("\nBest Balanced Accuracy threshold:")
print(best_balanced.to_string())

print("\nBest MCC threshold:")
print(best_mcc.to_string())


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame(
    {
        "feature": feature_columns,
        "importance": model.feature_importances_,
    }
).sort_values(
    "importance",
    ascending=False,
)


print("\n" + "=" * 70)
print("TOP XGBOOST FEATURES")
print("=" * 70)

print(
    importance.head(15).to_string(
        index=False
    )
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("XGBOOST RULE AGENT EXPERIMENT COMPLETE")
print("=" * 70)