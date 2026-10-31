from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

from xgboost import XGBClassifier


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT / "data" / "outputs" / "login_agent_results.parquet"
)

GROUND_TRUTH_FILE = (
    PROJECT_ROOT / "data" / "outputs" / "login_ground_truth.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "outputs" / "login_xgboost_experiment.csv"
)

RANDOM_STATE = 42

THRESHOLDS = [
    0.05, 0.10, 0.15, 0.20, 0.25,
    0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75,
    0.80, 0.85, 0.90,
]


# ============================================================
# 2. INPUT FEATURES
#    Do not include the existing agent's final score/status
#    or explanation as model inputs.
# ============================================================

FEATURES = [
    "logon_events",
    "logon_pc_count",
    "logon_logins",
    "logon_logoffs",
    "logon_events_clean_zscore",
    "logon_events_prev7_mean",
    "logon_events_change_7d",
    "temporal_change_count",
    "temporal_change_score",
    "persistent_behavior_flag",
    "after_hours_logons",
    "weekend_logons",
    "unique_logon_pcs",
    "total_logon_events",
    "after_hours_logon_ratio",
    "login_zscore_signal",
    "login_strong_zscore_signal",
    "login_temporal_change_signal",
    "login_strong_temporal_signal",
    "login_after_hours_signal",
    "login_multiple_pc_signal",
    "login_high_volume_signal",
    "login_after_hours_ratio_signal",
    "login_behavior_change_signal",
    "login_after_hours_anomaly_signal",
    "login_evidence_count",
    "login_evidence_score",
]


# ============================================================
# 3. LOAD DATA
# ============================================================

def load_data():
    print("=" * 65)
    print("LOGIN AGENT - XGBOOST EXPERIMENT")
    print("=" * 65)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Login Agent results not found:\n{INPUT_FILE}"
        )

    if not GROUND_TRUTH_FILE.exists():
        raise FileNotFoundError(
            f"Login ground truth not found:\n{GROUND_TRUTH_FILE}"
        )

    # Use fastparquet because PyArrow has caused issues
    # in this Windows environment.
    df = pd.read_parquet(
        INPUT_FILE,
        engine="fastparquet",
    )

    gt = pd.read_parquet(
        GROUND_TRUTH_FILE,
        engine="fastparquet",
    )

    print(f"\nLogin Agent rows: {len(df):,}")
    print(f"Ground-truth rows: {len(gt):,}")

    # Check required join columns.
    required_agent_columns = {"user", "day"}
    required_gt_columns = {
        "user",
        "day",
        "login_ground_truth",
    }

    missing_agent = required_agent_columns - set(df.columns)
    missing_gt = required_gt_columns - set(gt.columns)

    if missing_agent:
        raise ValueError(
            f"Missing Login Agent columns: {sorted(missing_agent)}"
        )

    if missing_gt:
        raise ValueError(
            f"Missing ground-truth columns: {sorted(missing_gt)}"
        )

    # The actual ground-truth label column is named
    # login_ground_truth, not ground_truth.
    gt = gt.rename(
        columns={"login_ground_truth": "ground_truth"}
    )

    return df, gt


# ============================================================
# 4. NORMALIZE KEYS AND MERGE GROUND TRUTH
# ============================================================

def prepare_dataset(df, gt):
    print("\nPreparing user and date keys...")

    # Work on copies. Do not change the original Parquet files.
    df = df.copy()
    gt = gt.copy()

    # FIX: Login Agent user IDs are uppercase, while the
    # ground-truth user IDs are lowercase.
    df["user"] = df["user"].astype("string").str.strip().str.lower()
    gt["user"] = gt["user"].astype("string").str.strip().str.lower()

    # Normalize both date columns.
    df["day"] = pd.to_datetime(df["day"], errors="coerce").dt.normalize()
    gt["day"] = pd.to_datetime(gt["day"], errors="coerce").dt.normalize()

    # Invalid keys should not silently disappear during merging.
    if df[["user", "day"]].isna().any().any():
        raise ValueError(
            "Login Agent data contains missing or invalid user/day keys."
        )

    if gt[["user", "day"]].isna().any().any():
        raise ValueError(
            "Ground-truth data contains missing or invalid user/day keys."
        )

    # A duplicate key can multiply rows during a merge.
    if df.duplicated(["user", "day"]).any():
        raise ValueError(
            "Duplicate user/day keys found in Login Agent results."
        )

    if gt.duplicated(["user", "day"]).any():
        raise ValueError(
            "Duplicate user/day keys found in login ground truth."
        )

    # Keep only the key and target columns from ground truth.
    gt = gt[["user", "day", "ground_truth"]]

    # Confirm that labels are binary and not missing.
    if gt["ground_truth"].isna().any():
        raise ValueError("Ground-truth labels contain missing values.")

    if not gt["ground_truth"].isin([0, 1]).all():
        raise ValueError("Ground-truth labels must contain only 0 and 1.")

    gt["ground_truth"] = gt["ground_truth"].astype("int8")

    print("User IDs normalized to lowercase.")
    print("Dates normalized.")
    print("Joining on user + day...")

    # Validate that the join keys have a one-to-one relationship.
    df = df.merge(
        gt,
        on=["user", "day"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = int((df["_merge"] != "both").sum())

    print(f"Unmatched Login Agent rows: {unmatched:,}")

    if unmatched:
        unmatched_rows = df.loc[
            df["_merge"] != "both", ["user", "day"]
        ].head(10)

        print("\nExample unmatched keys:")
        print(unmatched_rows.to_string(index=False))

        raise ValueError(
            f"{unmatched:,} Login Agent rows did not receive "
            "ground truth. Stopping to avoid incorrect evaluation."
        )

    df = df.drop(columns=["_merge"])

    df["ground_truth"] = df["ground_truth"].astype("int8")

    positives = int(df["ground_truth"].sum())
    negatives = len(df) - positives

    print("\nGround-truth distribution:")
    print(f"Normal user-days:    {negatives:,}")
    print(f"Malicious user-days: {positives:,}")

    if positives < 3:
        raise ValueError(
            "There are too few positive labels for this experiment's "
            "three-way stratified split."
        )

    return df


# ============================================================
# 5. PREPARE FEATURES AND TARGET
# ============================================================

def prepare_features(df):
    missing_features = [
        feature for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "The following expected features are missing:\n"
            + "\n".join(missing_features)
        )

    X = df[FEATURES].copy()
    y = df["ground_truth"].astype("int8").copy()

    # Ensure every model input is numeric.
    for column in X.columns:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    # XGBoost can handle NaN values, but infinite values
    # should be converted to missing values.
    X = X.replace([np.inf, -np.inf], np.nan)

    # Remove features that contain no usable values.
    all_missing = X.columns[X.isna().all()].tolist()

    if all_missing:
        raise ValueError(
            "These features contain no usable numeric values: "
            + ", ".join(all_missing)
        )

    print(f"\nFeatures used: {len(FEATURES)}")
    print(f"Feature rows: {len(X):,}")

    return X, y


# ============================================================
# 6. STRATIFIED TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def split_data(X, y):
    print("\nCreating stratified data splits...")

    # First split: 60% training, 40% temporary data.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.40,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Second split: divide the remaining 40% equally.
    # Final proportions are approximately 60/20/20.
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    print(f"Training rows:   {len(X_train):,}")
    print(f"Validation rows: {len(X_val):,}")
    print(f"Test rows:       {len(X_test):,}")

    print("\nMalicious labels in each split:")
    print(f"Training:   {int(y_train.sum())}")
    print(f"Validation: {int(y_val.sum())}")
    print(f"Test:       {int(y_test.sum())}")

    if y_train.nunique() < 2:
        raise ValueError(
            "Training data must contain both classes."
        )

    if y_val.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError(
            "Validation and test splits must each contain both "
            "classes to calculate the selected metrics. "
            "With only five positive labels, these results can "
            "be sensitive to the split."
        )

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
    )


# ============================================================
# 7. TRAIN XGBOOST
# ============================================================

def train_model(X_train, y_train):
    print("\nTraining XGBoost model...")

    positive_count = int(y_train.sum())
    negative_count = len(y_train) - positive_count

    if positive_count == 0:
        raise ValueError(
            "No malicious examples are available in the training set."
        )

    # Compensate for the severe class imbalance.
    scale_pos_weight = negative_count / positive_count

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )

    model.fit(X_train, y_train)

    print("XGBoost training completed.")

    return model


# ============================================================
# 8. EVALUATE MULTIPLE PROBABILITY THRESHOLDS
# ============================================================

def evaluate_thresholds(model, X_val, y_val, X_test, y_test):
    print("\nGenerating validation and test probabilities...")

    val_probabilities = model.predict_proba(X_val)[:, 1]
    test_probabilities = model.predict_proba(X_test)[:, 1]

    results = []

    for threshold in THRESHOLDS:
        val_predictions = (
            val_probabilities >= threshold
        ).astype("int8")

        test_predictions = (
            test_probabilities >= threshold
        ).astype("int8")

        val_tn, val_fp, val_fn, val_tp = confusion_matrix(
            y_val,
            val_predictions,
            labels=[0, 1],
        ).ravel()

        test_tn, test_fp, test_fn, test_tp = confusion_matrix(
            y_test,
            test_predictions,
            labels=[0, 1],
        ).ravel()

        results.append({
            "threshold": threshold,

            # Validation metrics are useful when choosing a threshold.
            "validation_TP": int(val_tp),
            "validation_FP": int(val_fp),
            "validation_TN": int(val_tn),
            "validation_FN": int(val_fn),
            "validation_accuracy": accuracy_score(
                y_val, val_predictions
            ),
            "validation_precision": precision_score(
                y_val, val_predictions, zero_division=0
            ),
            "validation_recall": recall_score(
                y_val, val_predictions, zero_division=0
            ),
            "validation_f1": f1_score(
                y_val, val_predictions, zero_division=0
            ),

            # Test metrics provide the held-out comparison.
            "test_TP": int(test_tp),
            "test_FP": int(test_fp),
            "test_TN": int(test_tn),
            "test_FN": int(test_fn),
            "test_accuracy": accuracy_score(
                y_test, test_predictions
            ),
            "test_precision": precision_score(
                y_test, test_predictions, zero_division=0
            ),
            "test_recall": recall_score(
                y_test, test_predictions, zero_division=0
            ),
            "test_f1": f1_score(
                y_test, test_predictions, zero_division=0
            ),
            "test_balanced_accuracy": balanced_accuracy_score(
                y_test, test_predictions
            ),
            "test_mcc": matthews_corrcoef(
                y_test, test_predictions
            ),
        })

    results_df = pd.DataFrame(results)

    # Probability-ranking metrics do not depend on a threshold.
    test_auc = roc_auc_score(y_test, test_probabilities)
    test_average_precision = average_precision_score(
        y_test, test_probabilities
    )

    results_df["test_roc_auc"] = test_auc
    results_df["test_average_precision"] = test_average_precision

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 65)
    print("THRESHOLD EVALUATION")
    print("=" * 65)

    display_columns = [
        "threshold",
        "test_TP",
        "test_FP",
        "test_TN",
        "test_FN",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_balanced_accuracy",
        "test_mcc",
    ]

    print(
        results_df[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print(f"\nTest ROC-AUC: {test_auc:.4f}")
    print(f"Test Average Precision: {test_average_precision:.4f}")

    print(f"\nResults saved to:\n{OUTPUT_FILE}")

    return results_df


# ============================================================
# 9. DISPLAY TOP FEATURE IMPORTANCES
# ============================================================

def show_feature_importance(model):
    importance_df = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_,
    })

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False,
    )

    print("\n" + "=" * 65)
    print("TOP 15 FEATURE IMPORTANCES")
    print("=" * 65)

    print(
        importance_df.head(15).to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )


# ============================================================
# 10. MAIN
# ============================================================

def main():
    df, gt = load_data()

    df = prepare_dataset(df, gt)

    X, y = prepare_features(df)

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_data(X, y)

    model = train_model(X_train, y_train)

    evaluate_thresholds(
        model,
        X_val,
        y_val,
        X_test,
        y_test,
    )

    show_feature_importance(model)

    print("\nExperiment finished successfully.")
    print("The existing Login Agent result file was not modified.")


if __name__ == "__main__":
    main()