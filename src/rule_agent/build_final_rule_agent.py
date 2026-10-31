import os
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")


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
    "final_rule_agent_results.parquet"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "outputs",
    "final_rule_agent_xgb_results.parquet"
)

TARGET_COLUMN = "ground_truth"

# Selected from the successful XGBoost experiment
FINAL_THRESHOLD = 0.30


# ================================================================
# FEATURES
# ================================================================

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
    "total_activity_zscore",
    "logon_events_zscore",
    "device_events_zscore",
    "file_events_zscore",
    "emails_sent_zscore",
    "web_visits_zscore",
    "behavior_deviation",
    "clean_behavior_deviation",
]


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("FINAL RULE AGENT - XGBOOST")
    print("=" * 70)

    # ------------------------------------------------------------
    # 1. LOAD EXISTING RULE AGENT RESULTS
    # ------------------------------------------------------------

    print("\nLoading current Rule Agent results...")

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_parquet(INPUT_FILE)

    print(f"Rows : {len(df):,}")
    print(f"Users: {df['user'].nunique():,}")

    # ------------------------------------------------------------
    # 2. CHECK REQUIRED COLUMNS
    # ------------------------------------------------------------

    required_columns = FEATURES + [
        TARGET_COLUMN,
        "user",
        "day",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing_columns)
        )

    # ------------------------------------------------------------
    # 3. PREPARE FEATURES
    # ------------------------------------------------------------

    X = df[FEATURES].copy()

    y = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce"
    ).fillna(0).astype(int)

    # Replace invalid numeric values
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(0)

    print(f"\nFeatures used: {len(FEATURES)}")

    print("\nGround-truth distribution:")
    print(y.value_counts())

    positive_count = int(y.sum())
    negative_count = int((y == 0).sum())

    if positive_count == 0:
        raise ValueError(
            "No positive samples found in ground truth."
        )

    # ------------------------------------------------------------
    # 4. CALCULATE CLASS WEIGHT
    # ------------------------------------------------------------

    scale_pos_weight = negative_count / positive_count

    print(
        f"\nXGBoost scale_pos_weight: "
        f"{scale_pos_weight:.4f}"
    )

    # ------------------------------------------------------------
    # 5. TRAIN FINAL XGBOOST MODEL
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING FINAL XGBOOST RULE MODEL")
    print("=" * 70)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X, y)

    print("Training completed.")

    # ------------------------------------------------------------
    # 6. GENERATE PROBABILITIES FOR ALL ROWS
    # ------------------------------------------------------------

    print("\nGenerating final Rule Agent probabilities...")

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= FINAL_THRESHOLD
    ).astype(int)

    # ------------------------------------------------------------
    # 7. CREATE FINAL RULE AGENT OUTPUT
    # ------------------------------------------------------------

    result = df.copy()

    result["xgb_probability"] = probabilities

    result["xgb_prediction"] = predictions

    # Final Rule Agent score in percentage form
    result["final_rule_score"] = (
        probabilities * 100.0
    )

    result["final_rule_triggered"] = (
        predictions == 1
    )

    # ------------------------------------------------------------
    # 8. FINAL THREAT LEVEL
    # ------------------------------------------------------------

    result["final_threat_level"] = np.select(
        [
            probabilities >= 0.70,
            probabilities >= 0.30,
            probabilities >= 0.10,
        ],
        [
            "HIGH",
            "MEDIUM",
            "LOW",
        ],
        default="NORMAL",
    )

    # ------------------------------------------------------------
    # 9. SAVE ONLY FINAL OUTPUT
    # ------------------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    result.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    # ------------------------------------------------------------
    # 10. SUMMARY
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL RULE AGENT SUMMARY")
    print("=" * 70)

    print(f"Rows: {len(result):,}")
    print(f"Users: {result['user'].nunique():,}")

    print(
        f"\nFinal threshold: "
        f"{FINAL_THRESHOLD:.2f}"
    )

    print("\nPrediction distribution:")
    print(
        result["xgb_prediction"]
        .value_counts()
        .sort_index()
    )

    print("\nThreat-level distribution:")
    print(
        result["final_threat_level"]
        .value_counts()
    )

    print("\nFinal score statistics:")
    print(
        result["final_rule_score"].describe()
    )

    print("\nOutput saved to:")
    print(OUTPUT_FILE)

    print("\n" + "=" * 70)
    print("FINAL RULE AGENT COMPLETE")
    print("=" * 70)