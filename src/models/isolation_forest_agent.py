    
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

# ------------------------------------------------------------
# PATHS AND CONFIGURATION
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"

INPUT_FILE = OUTPUT_DIR / "final_behavioral_features.parquet"
OUTPUT_FILE = OUTPUT_DIR / "isolation_forest_results.parquet"

ENGINE = "fastparquet"

# Provisional contamination rate, NOT estimated from ground truth.
# It controls the approximate fraction of training records
# classified as anomalies by Isolation Forest.
CONTAMINATION = 0.001

TEST_FRACTION = 0.20
RANDOM_STATE = 42

FEATURES = [
    "logon_events",
    "logon_pc_count",
    "logon_logins",
    "logon_logoffs",
    "device_events",
    "device_pc_count",
    "device_connects",
    "device_disconnects",
    "file_events",
    "file_pc_count",
    "unique_files",
    "emails_sent",
    "total_email_size",
    "total_attachments",
    "recipient_count",
    "unique_recipients",
    "web_visits",
    "unique_urls",
    "unique_domains",
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
    "total_activity_clean_zscore",
    "logon_events_clean_zscore",
    "device_events_clean_zscore",
    "file_events_clean_zscore",
    "emails_sent_clean_zscore",
    "web_visits_clean_zscore",
    "clean_behavior_deviation",
]


def main():
    print("=" * 65)
    print("ISOLATION FOREST AGENT")
    print("=" * 65)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_FILE}")

    df = pd.read_parquet(INPUT_FILE, engine=ENGINE)

    required = ["user", "day", "ground_truth"] + FEATURES
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df[required].isna().any().any():
        raise ValueError("Required columns contain missing values.")

    labels = pd.to_numeric(df["ground_truth"], errors="raise")
    if not labels.isin([0, 1]).all():
        raise ValueError("ground_truth must contain only 0 and 1.")

    # Establish chronological order before splitting.
    df["day"] = pd.to_datetime(df["day"], errors="raise")
    df = df.sort_values(["day", "user"]).reset_index(drop=True)

    # Hold out the latest 20% of unique dates.
    unique_days = np.sort(df["day"].unique())
    if len(unique_days) < 2:
        raise ValueError("Not enough unique dates for a chronological split.")

    split_index = int(len(unique_days) * (1 - TEST_FRACTION))
    split_index = min(max(split_index, 1), len(unique_days) - 1)
    split_day = unique_days[split_index]

    train_mask = df["day"] < split_day
    test_mask = ~train_mask

    if not train_mask.any() or not test_mask.any():
        raise ValueError("Chronological split produced an empty partition.")

    # Convert features to numeric and reject invalid values.
    X = df[FEATURES].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Features contain infinite or invalid values.")

    X_train = X.loc[train_mask]
    X_test = X.loc[test_mask]

    y_test = labels.loc[test_mask].astype(int)

    print(f"Total rows: {len(df):,}")
    print(f"Training rows: {int(train_mask.sum()):,}")
    print(f"Test rows: {int(test_mask.sum()):,}")
    print(f"First test date: {pd.Timestamp(split_day).date()}")
    print(f"Features: {len(FEATURES)}")
    print(f"Test malicious days: {int(y_test.sum()):,}")

    # Fit without ground-truth labels.
    model = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train)

    # sklearn: -1 means anomaly; +1 means inlier.
    raw_prediction = model.predict(X_test)
    anomaly_score = -model.score_samples(X_test)
    anomaly_prediction = (raw_prediction == -1).astype(int)

    # Preserve all rows. Training rows have no out-of-sample prediction.
    result = df[["user", "day", "ground_truth"]].copy()
    result["isolation_forest_anomaly_score"] = np.nan
    result["isolation_forest_prediction"] = pd.Series(
        pd.NA, index=result.index, dtype="Int64"
    )
    result["evaluation_split"] = np.where(
        train_mask, "train", "test"
    )

    result.loc[test_mask, "isolation_forest_anomaly_score"] = anomaly_score
    result.loc[test_mask, "isolation_forest_prediction"] = anomaly_prediction

    # Evaluate only the held-out test dates.
    tn, fp, fn, tp = confusion_matrix(
        y_test, anomaly_prediction, labels=[0, 1]
    ).ravel()

    print("\nHELD-OUT TEST RESULTS")
    print(f"TP: {tp}")
    print(f"FP: {fp}")
    print(f"TN: {tn}")
    print(f"FN: {fn}")
    print(
        "Precision:",
        f"{precision_score(y_test, anomaly_prediction, zero_division=0):.6f}",
    )
    print(
        "Recall:",
        f"{recall_score(y_test, anomaly_prediction, zero_division=0):.6f}",
    )
    print(
        "F1:",
        f"{f1_score(y_test, anomaly_prediction, zero_division=0):.6f}",
    )

    # Save a separate output; do not overwrite existing agent results.
    result.to_parquet(OUTPUT_FILE, engine=ENGINE, index=False)

    print(f"\nOutput saved: {OUTPUT_FILE}")
    print(
        "\nLIMITATION: This is a chronological holdout, but existing "
        "baseline-derived input features may use information from "
        "dates beyond the scored day. Verify feature construction "
        "before treating this as leakage-free evaluation."
    )


if __name__ == "__main__":
    main()
