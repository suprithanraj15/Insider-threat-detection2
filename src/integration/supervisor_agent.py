from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"
ENGINE = "fastparquet"

INPUT_FILE = OUTPUT_DIR / "agent_integration_dataset.parquet"
OUTPUT_FILE = OUTPUT_DIR / "supervisor_agent_results.parquet"

ISOLATION_FILE = OUTPUT_DIR / "isolation_forest_results.parquet"
RANDOM_FOREST_FILE = OUTPUT_DIR / "random_forest_results.parquet"
LSTM_FILE = OUTPUT_DIR / "lstm_agent_results.parquet"

# Existing agent weights: preserved.
AGENT_WEIGHTS = {
    "rule_final_rule_score": 0.40,
    "login_login_agent_score": 0.15,
    "file_file_agent_score": 0.15,
    "device_device_agent_score": 0.15,
    "network_network_v2_score": 0.15,
}

# ML contribution weights across available ML models.
ML_WEIGHTS = {
    "isolation_forest": 0.40,
    "random_forest": 0.30,
    "lstm": 0.30,
}

MEDIUM_THRESHOLD = 25.0
HIGH_THRESHOLD = 60.0

# Isolation Forest anomaly score scaling parameters
# Raw anomaly score is -score_samples(X), typical range [0.33, 0.73]
IF_SCORE_MIN = 0.33
IF_SCORE_MAX = 0.75


def load_isolation_forest_output(path):
    """
    Load Isolation Forest output.
    Uses continuous anomaly score scaled to [0, 100] rather than
    a binary 0/100 step function, preserving anomaly graduation.
    Retains raw anomaly score and binary prediction.
    """
    if not path.exists():
        raise FileNotFoundError(f"Isolation Forest output not found: {path}")

    data = pd.read_parquet(path, engine=ENGINE)
    required = {"user", "day", "ground_truth", "isolation_forest_anomaly_score", "isolation_forest_prediction"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Isolation Forest output missing columns: {sorted(missing)}")

    data["day"] = pd.to_datetime(data["day"])
    if data.duplicated(["user", "day"]).any():
        raise ValueError("Isolation Forest contains duplicate user/day rows.")

    # Continuous raw anomaly score (higher = more anomalous)
    raw_score = pd.to_numeric(data["isolation_forest_anomaly_score"], errors="coerce")
    data["isolation_forest_raw_score"] = raw_score

    # Defensible continuous scaling: map [IF_SCORE_MIN, IF_SCORE_MAX] to [0, 100]
    scaled = ((raw_score - IF_SCORE_MIN) / (IF_SCORE_MAX - IF_SCORE_MIN)).clip(0.0, 1.0) * 100.0
    data["isolation_forest_score"] = scaled

    # Keep binary prediction as well
    data["isolation_forest_prediction"] = pd.to_numeric(data["isolation_forest_prediction"], errors="coerce")

    return data[["user", "day", "ground_truth", "isolation_forest_raw_score", "isolation_forest_score", "isolation_forest_prediction"]]


def load_classifier_output(path, prob_col, pred_col, model_name):
    """
    Load supervised classifier output (Random Forest or LSTM).
    Probability is mapped from [0, 1] to [0, 100].
    """
    if not path.exists():
        raise FileNotFoundError(f"{model_name} output not found: {path}")

    data = pd.read_parquet(path, engine=ENGINE)
    required = {"user", "day", "ground_truth", prob_col, pred_col}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{model_name} output missing columns: {sorted(missing)}")

    data["day"] = pd.to_datetime(data["day"])
    if data.duplicated(["user", "day"]).any():
        raise ValueError(f"{model_name} contains duplicate user/day rows.")

    prob = pd.to_numeric(data[prob_col], errors="raise")
    data[f"{model_name}_score"] = prob * 100.0
    data[f"{model_name}_prediction"] = pd.to_numeric(data[pred_col], errors="raise")

    return data[["user", "day", "ground_truth", f"{model_name}_score", f"{model_name}_prediction"]]


def main():
    print("=" * 65)
    print("SUPERVISOR AGENT - MULTI-MODEL INTEGRATION")
    print("=" * 65)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Integration data not found: {INPUT_FILE}")

    df = pd.read_parquet(INPUT_FILE, engine=ENGINE)
    df["day"] = pd.to_datetime(df["day"])

    required = {"user", "day", "ground_truth", *AGENT_WEIGHTS.keys()}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Integration data is missing: {sorted(missing)}")

    if df.duplicated(["user", "day"]).any():
        raise ValueError("Integration data contains duplicate user/day rows.")

    # Calculate baseline 5-agent domain score
    base = df[["user", "day", "ground_truth"]].copy()
    base_score = sum(
        pd.to_numeric(df[col], errors="raise") * weight
        for col, weight in AGENT_WEIGHTS.items()
    )
    base["existing_agent_score"] = base_score.clip(0, 100)

    # Load model outputs with preserved held-out coverage
    isolation = load_isolation_forest_output(ISOLATION_FILE)
    rf = load_classifier_output(RANDOM_FOREST_FILE, "random_forest_probability", "random_forest_prediction", "random_forest")
    lstm = load_classifier_output(LSTM_FILE, "lstm_probability", "lstm_prediction", "lstm")

    result = base.merge(
        isolation, on=["user", "day", "ground_truth"], how="left",
        validate="one_to_one",
    )
    result = result.merge(
        rf, on=["user", "day", "ground_truth"], how="left",
        validate="one_to_one",
    )
    result = result.merge(
        lstm, on=["user", "day", "ground_truth"], how="left",
        validate="one_to_one",
    )

    # Track availability explicitly per model
    result["has_isolation_forest"] = result["isolation_forest_score"].notna()
    result["has_random_forest"] = result["random_forest_score"].notna()
    result["has_lstm"] = result["lstm_score"].notna()
    result["ml_models_available"] = (
        result["has_isolation_forest"].astype(int)
        + result["has_random_forest"].astype(int)
        + result["has_lstm"].astype(int)
    )

    # Weighted ML fusion across available models
    score_columns = [
        ("isolation_forest_score", ML_WEIGHTS["isolation_forest"]),
        ("random_forest_score", ML_WEIGHTS["random_forest"]),
        ("lstm_score", ML_WEIGHTS["lstm"]),
    ]

    weighted_sum = pd.Series(0.0, index=result.index)
    available_weight = pd.Series(0.0, index=result.index)

    for column, weight in score_columns:
        available = result[column].notna()
        weighted_sum.loc[available] += result.loc[available, column] * weight
        available_weight.loc[available] += weight

    result["ml_model_score"] = np.where(
        available_weight > 0,
        weighted_sum / available_weight.replace(0, np.nan),
        np.nan,
    )

    result["scoring_mode"] = np.where(
        result["ml_models_available"] > 0,
        "HYBRID",
        "AGENT_ONLY",
    )

    # Combine domain agent score and ML score:
    # 50/50 when ML evidence exists; 100% agent score when ML is unavailable
    result["supervisor_risk_score"] = np.where(
        result["ml_model_score"].notna(),
        0.50 * result["existing_agent_score"]
        + 0.50 * result["ml_model_score"],
        result["existing_agent_score"],
    ).clip(0, 100)

    result["supervisor_threat_level"] = np.select(
        [
            result["supervisor_risk_score"] >= HIGH_THRESHOLD,
            result["supervisor_risk_score"] >= MEDIUM_THRESHOLD,
        ],
        ["HIGH", "MEDIUM"],
        default="LOW",
    )

    result["supervisor_prediction"] = (
        result["supervisor_risk_score"] >= MEDIUM_THRESHOLD
    ).astype(int)

    print(f"Total Rows: {len(result):,}")
    print(f"Agent-only scoring rows: {int((result['scoring_mode'] == 'AGENT_ONLY').sum()):,}")
    print(f"Hybrid ML scoring rows:  {int((result['scoring_mode'] == 'HYBRID').sum()):,}")
    print(f"  - 1 model available:   {int((result['ml_models_available'] == 1).sum()):,}")
    print(f"  - 2 models available:  {int((result['ml_models_available'] == 2).sum()):,}")
    print(f"  - 3 models available:  {int((result['ml_models_available'] == 3).sum()):,}")
    print("\nThreat-level distribution:")
    print(result["supervisor_threat_level"].value_counts().to_string())

    positives = result[result["ground_truth"] == 1]
    print(f"\nGround Truth Malicious User-Days ({len(positives)} total):")
    print(positives["supervisor_threat_level"].value_counts().to_string())

    result.to_parquet(
        OUTPUT_FILE,
        engine=ENGINE,
        index=False,
    )

    print(f"\nOutput saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
