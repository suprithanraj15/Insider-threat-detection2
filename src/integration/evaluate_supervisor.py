from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "outputs" / "final_behavioral_features.parquet"
OUT = ROOT / "data" / "outputs" / "evaluation_user_holdout"
OUT.mkdir(parents=True, exist_ok=True)

TARGET = "ground_truth"
SEED = 42
THRESHOLD = 0.30

FEATURES = [
    "logon_events", "logon_pc_count", "logon_logins", "logon_logoffs",
    "device_events", "device_pc_count", "device_connects", "device_disconnects",
    "file_events", "file_pc_count", "unique_files",
    "emails_sent", "total_email_size", "total_attachments",
    "recipient_count", "unique_recipients",
    "web_visits", "unique_urls", "unique_domains",
    "total_activity", "active_channels",
    "device_activity_ratio", "file_activity_ratio", "file_diversity",
    "email_activity_ratio", "avg_email_size", "avg_recipients_per_email",
    "web_activity_ratio", "url_diversity", "domain_diversity",
    "total_activity_clean_zscore", "logon_events_clean_zscore",
    "device_events_clean_zscore", "file_events_clean_zscore",
    "emails_sent_clean_zscore", "web_visits_clean_zscore",
    "clean_behavior_deviation",
]

def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Input not found: {INPUT}")

    df = pd.read_parquet(INPUT)
    required = {"user", "day", TARGET, *FEATURES}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["day"] = pd.to_datetime(df["day"], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["user", "day"]).sort_values(["user", "day"]).reset_index(drop=True)

    if df["user"].nunique() < 2:
        raise ValueError("Need at least two distinct users for a grouped holdout.")

    # Keep all rows for a user together; never split the same user across sets.
    splitter = GroupShuffleSplit(n_splits=100, test_size=0.20, random_state=SEED)
    train_idx, test_idx = next(splitter.split(df, df[TARGET], groups=df["user"]))
    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()

    # This evaluation is not informative if the holdout has no positive cases.
    summary = {
        "rows_total": int(len(df)),
        "users_total": int(df["user"].nunique()),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_users": int(train["user"].nunique()),
        "test_users": int(test["user"].nunique()),
        "train_positives": int(train[TARGET].sum()),
        "test_positives": int(test[TARGET].sum()),
        "test_negative_rows": int((test[TARGET] == 0).sum()),
        "threshold_xgboost": THRESHOLD,
        "note": (
            "Grouped user holdout. Interpret metrics cautiously because the dataset "
            "has only four malicious users. This does not evaluate the complete Supervisor."
        ),
    }

    if train[TARGET].nunique() < 2:
        raise ValueError("Training split has only one class; cannot train classifiers.")
    if test[TARGET].sum() == 0:
        (OUT / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        raise ValueError(
            f"Test split has zero malicious rows. Summary saved to {OUT / 'split_summary.json'}. "
            "Do not report test precision/recall/F1 from this split."
        )

    X_train = train[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = test[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train = train[TARGET].to_numpy()
    y_test = test[TARGET].to_numpy()

    models = {}

    scale_pos_weight = max(1.0, (y_train == 0).sum() / max(1, (y_train == 1).sum()))
    xgb_model = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="binary:logistic", eval_metric="logloss",
        scale_pos_weight=scale_pos_weight, random_state=SEED, n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train)
    models["XGBoost"] = (xgb_model.predict_proba(X_test)[:, 1], THRESHOLD)

    rf_model = RandomForestClassifier(
        n_estimators=300, class_weight="balanced",
        random_state=SEED, n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)
    models["Random Forest"] = (rf_model.predict_proba(X_test)[:, 1], 0.50)

    # Train the anomaly detector using benign training rows only.
    benign_train = X_train[y_train == 0]
    if len(benign_train) < 2:
        raise ValueError("Not enough benign training rows for Isolation Forest.")
    if_model = IsolationForest(
        n_estimators=200, contamination=0.001, random_state=SEED, n_jobs=-1
    )
    if_model.fit(benign_train)
    # Anomaly decision score is converted to a binary prediction; it is not a probability.
    if_pred = (if_model.predict(X_test) == -1).astype(int)

    rows = []
    prediction_output = test[["user", "day", TARGET]].copy()

    for name, (scores, threshold) in models.items():
        pred = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        rows.append({
            "model": name,
            "threshold": threshold,
            "test_rows": len(y_test),
            "test_positives": int(y_test.sum()),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            "average_precision": average_precision_score(y_test, scores),
        })
        prediction_output[f"{name.lower().replace(' ', '_')}_score"] = scores
        prediction_output[f"{name.lower().replace(' ', '_')}_prediction"] = pred

    tn, fp, fn, tp = confusion_matrix(y_test, if_pred, labels=[0, 1]).ravel()
    rows.append({
        "model": "Isolation Forest",
        "threshold": "anomaly decision",
        "test_rows": len(y_test),
        "test_positives": int(y_test.sum()),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "precision": precision_score(y_test, if_pred, zero_division=0),
        "recall": recall_score(y_test, if_pred, zero_division=0),
        "F1": f1_score(y_test, if_pred, zero_division=0),
        "average_precision": None,
    })
    prediction_output["isolation_forest_prediction"] = if_pred

    pd.DataFrame(rows).to_csv(OUT / "model_metrics.csv", index=False)
    prediction_output.to_parquet(OUT / "heldout_predictions.parquet", index=False)
    (OUT / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nGrouped user holdout completed.")
    print(json.dumps(summary, indent=2))
    print("\nMetrics:")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nSaved outputs to: {OUT}")

if __name__ == "__main__":
    main()
