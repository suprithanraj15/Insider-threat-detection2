
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data" / "outputs" / "final_behavioral_features.parquet"
OUTPUT_FILE = ROOT / "data" / "outputs" / "random_forest_results.parquet"

RANDOM_STATE = 42
TEST_SIZE = 0.20
THRESHOLD = 0.50

# Exclude identifiers, labels, and scenario information.
EXCLUDED_COLUMNS = {
    "user",
    "day",
    "ground_truth",
    "scenario",
}

# Exclude global z-scores and the deviation derived from them.
EXCLUDED_FEATURES = {
    "total_activity_zscore",
    "logon_events_zscore",
    "device_events_zscore",
    "file_events_zscore",
    "emails_sent_zscore",
    "web_visits_zscore",
    "behavior_deviation",
}

# Exclude baseline statistics as inputs; retain past-only clean z-scores.
EXCLUDED_FEATURE_SUFFIXES = (
    "_baseline_mean",
    "_baseline_std",
)


def main():
    print("=" * 65)
    print("RANDOM FOREST AGENT")
    print("=" * 65)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_parquet(INPUT_FILE, engine="fastparquet")

    if "ground_truth" not in df.columns:
        raise ValueError("Required label column 'ground_truth' is missing.")

    if df["ground_truth"].isna().any():
        raise ValueError("ground_truth contains missing values.")

    labels = set(df["ground_truth"].unique())
    if not labels.issubset({0, 1}) or len(labels) != 2:
        raise ValueError(f"Expected binary labels 0 and 1; found {labels}")

    feature_columns = [
        col for col in df.columns
        if col not in EXCLUDED_COLUMNS
        and col not in EXCLUDED_FEATURES
        and not col.endswith(EXCLUDED_FEATURE_SUFFIXES)
    ]

    X = df[feature_columns].copy()
    y = df["ground_truth"].astype(int)

    # Keep only numeric features and reject unexpected nonnumeric columns.
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(f"Non-numeric features require review: {non_numeric}")

    X = X.replace([np.inf, -np.inf], np.nan)

    print(f"Rows: {len(df):,}")
    print(f"Features: {len(feature_columns)}")
    print(f"Normal rows: {(y == 0).sum():,}")
    print(f"Malicious rows: {(y == 1).sum():,}")

    # Stratification ensures each split contains both classes.
    X_train, X_test, y_train, y_test, idx_train, idx_test = (
        train_test_split(
            X,
            y,
            df.index,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            min_samples_leaf=2,
        )),
    ])

    print(f"Training rows: {len(X_train):,}")
    print(f"Training malicious rows: {(y_train == 1).sum():,}")
    print(f"Test rows: {len(X_test):,}")
    print(f"Test malicious rows: {(y_test == 1).sum():,}")

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    print("\nHELD-OUT TEST RESULTS")
    print(f"Threshold: {THRESHOLD:.2f}")
    print("Confusion matrix [ [TN, FP], [FN, TP] ]:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.6f}")
    print(f"Recall:    {recall_score(y_test, predictions, zero_division=0):.6f}")
    print(f"F1:        {f1_score(y_test, predictions, zero_division=0):.6f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, labels=[0, 1], zero_division=0))

    # Save held-out predictions only; do not overwrite other agent outputs.
    results = df.loc[idx_test, ["user", "day", "ground_truth"]].copy()
    results["random_forest_probability"] = probabilities
    results["random_forest_prediction"] = predictions
    results["random_forest_threshold"] = THRESHOLD

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(
        OUTPUT_FILE,
        engine="fastparquet",
        index=False,
    )

    print(f"Output saved: {OUTPUT_FILE}")
    print("Note: This is a stratified random holdout, not a chronological test.")


if __name__ == "__main__":
    main()

    print("\nDone.")