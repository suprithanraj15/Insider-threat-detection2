
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data" / "outputs" / "final_behavioral_features.parquet"
OUTPUT_FILE = ROOT / "data" / "outputs" / "lstm_agent_results.parquet"

SEQUENCE_LENGTH = 7
TRAIN_FRACTION = 0.80
RANDOM_STATE = 42
THRESHOLD = 0.50
EPOCHS = 15
BATCH_SIZE = 256

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# --------------------------------------------------
# FEATURES
# --------------------------------------------------

EXCLUDED_COLUMNS = {
    "user",
    "day",
    "ground_truth",
    "scenario",
    "total_activity_zscore",
    "logon_events_zscore",
    "device_events_zscore",
    "file_events_zscore",
    "emails_sent_zscore",
    "web_visits_zscore",
    "behavior_deviation",
}

BASELINE_SUFFIXES = (
    "_baseline_mean",
    "_baseline_std",
)

# Features must be numeric and must not include labels or identifiers.


def main():
    print("=" * 65)
    print("LSTM AGENT")
    print("=" * 65)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_parquet(INPUT_FILE, engine="fastparquet")
    df["day"] = pd.to_datetime(df["day"], errors="raise")

    required = {"user", "day", "ground_truth"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df.duplicated(["user", "day"]).any():
        raise ValueError("Duplicate user/day rows detected.")

    df = df.sort_values(["day", "user"]).reset_index(drop=True)

    feature_columns = [
        c for c in df.columns
        if c not in EXCLUDED_COLUMNS
        and not c.endswith(BASELINE_SUFFIXES)
    ]

    non_numeric = df[feature_columns].select_dtypes(
        exclude=[np.number]
    ).columns.tolist()

    if non_numeric:
        raise ValueError(f"Non-numeric input features: {non_numeric}")

    print(f"Rows: {len(df):,}")
    print(f"Users: {df['user'].nunique():,}")
    print(f"Features: {len(feature_columns)}")
    print(f"Malicious rows: {int((df['ground_truth'] == 1).sum()):,}")

  
    # STRATIFIED RANDOM SPLIT
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df["ground_truth"],
    )

    train_df = train_df.sort_values(["user", "day"]).copy()
    test_df = test_df.sort_values(["user", "day"]).copy()

    print(f"Training rows: {len(train_df):,}")
    print(
        "Training malicious rows:",
        int(train_df["ground_truth"].sum()),
    )
    print(f"Test rows: {len(test_df):,}")
    print(
        "Test malicious rows:",
        int(test_df["ground_truth"].sum()),
    )


    # --------------------------------------------------
    # NORMALIZATION — FIT ON TRAINING DATA ONLY
    # --------------------------------------------------

    scaler = StandardScaler()

    train_values = train_df[feature_columns].replace(
        [np.inf, -np.inf], np.nan
    )
    test_values = test_df[feature_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    medians = train_values.median().fillna(0.0)
    train_values = train_values.fillna(medians)
    test_values = test_values.fillna(medians)

    scaler.fit(train_values)

    train_scaled = pd.DataFrame(
        scaler.transform(train_values),
        columns=feature_columns,
        index=train_df.index,
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test_values),
        columns=feature_columns,
        index=test_df.index,
    )

    # --------------------------------------------------
    # CREATE PER-USER SEQUENCES
    # --------------------------------------------------

  
    def make_sequences(part_df, scaled_features):
        X_sequences = []
        y_sequences = []
        row_indices = []

        for _, group in part_df.groupby("user", sort=False):
            indices = group.sort_values("day").index.to_list()

            for pos in range(SEQUENCE_LENGTH - 1, len(indices)):
                window_indices = indices[
                    pos - SEQUENCE_LENGTH + 1:pos + 1
                ]

                X_sequences.append(
                    scaled_features.loc[window_indices].to_numpy(
                        dtype=np.float32
                    )
                )

                current_index = indices[pos]
                y_sequences.append(
                    int(part_df.loc[current_index, "ground_truth"])
                )
                row_indices.append(current_index)

        if not X_sequences:
            return (
                np.empty(
                    (0, SEQUENCE_LENGTH, len(feature_columns)),
                    dtype=np.float32,
                ),
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int64),
            )

        return (
            np.asarray(X_sequences, dtype=np.float32),
            np.asarray(y_sequences, dtype=np.int32),
            np.asarray(row_indices, dtype=np.int64),
        )


    X_train, y_train, train_indices = make_sequences(
        train_df, train_scaled
    )
    X_test, y_test, test_indices = make_sequences(
        test_df, test_scaled
    )

    print(f"Training sequences: {len(X_train):,}")
    print(f"Test sequences: {len(X_test):,}")
    print(f"Malicious training sequences: {int(y_train.sum()):,}")
    print(f"Malicious test sequences: {int(y_test.sum()):,}")

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Not enough per-user history to create sequences.")

    if len(np.unique(y_train)) < 2:
        raise ValueError("Training sequences must contain both classes.")

    if y_test.sum() == 0:
        raise ValueError(
            "No malicious test sequences remain after sequence creation."
        )

    # --------------------------------------------------
    # BUILD AND TRAIN LSTM
    # --------------------------------------------------

    model = tf.keras.Sequential([
        tf.keras.layers.Input(
            shape=(SEQUENCE_LENGTH, len(feature_columns))
        ),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    normal_count = int((y_train == 0).sum())
    malicious_count = int((y_train == 1).sum())

    class_weight = {
        0: 1.0,
        1: normal_count / max(malicious_count, 1),
    }

    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        verbose=1,
        shuffle=False,
    )

    # --------------------------------------------------
    # EVALUATE
    # --------------------------------------------------

    probabilities = model.predict(X_test, verbose=0).ravel()
    predictions = (probabilities >= THRESHOLD).astype(int)

    print("\nCHRONOLOGICAL HELD-OUT RESULTS")
    print(f"Threshold: {THRESHOLD:.2f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.6f}")
    print(f"Recall:    {recall_score(y_test, predictions, zero_division=0):.6f}")
    print(f"F1:        {f1_score(y_test, predictions, zero_division=0):.6f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test, predictions, labels=[0, 1], zero_division=0
        )
    )

    # --------------------------------------------------
    # SAVE HELD-OUT PREDICTIONS
    # --------------------------------------------------

    results = df.loc[
        test_indices, ["user", "day", "ground_truth"]
    ].copy()

    results["lstm_probability"] = probabilities
    results["lstm_prediction"] = predictions
    results["lstm_threshold"] = THRESHOLD

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(
        OUTPUT_FILE,
        engine="fastparquet",
        index=False,
    )

    print(f"Output saved: {OUTPUT_FILE}")
    print("Evaluation uses a chronological date split.")


if __name__ == "__main__":
    main()
