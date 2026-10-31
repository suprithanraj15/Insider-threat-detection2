
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "outputs" / "behavioral_features.parquet"
OUTPUT_FILE = PROJECT_ROOT / "data" / "outputs" / "final_behavioral_features.parquet"

PARQUET_ENGINE = "fastparquet"
MIN_HISTORY = 3

BASELINE_COLUMNS = [
    "total_activity",
    "logon_events",
    "device_events",
    "file_events",
    "emails_sent",
    "web_visits",
]


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def main():
    print("Loading behavioral features...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_parquet(
        INPUT_FILE,
        engine=PARQUET_ENGINE,
    )

    required_columns = {"user", "day", *BASELINE_COLUMNS}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    df["day"] = pd.to_datetime(df["day"], errors="raise")

    print(f"Rows loaded: {len(df):,}")
    print(f"Users: {df['user'].nunique():,}")


    # --------------------------------------------------
    # SORT CHRONOLOGICALLY
    # --------------------------------------------------

    print("Sorting user activity chronologically...")

    df = df.sort_values(
        ["user", "day"]
    ).reset_index(drop=True)

    # Each user should have no more than one row per day.
    duplicates = df.duplicated(
        subset=["user", "day"]
    )

    if duplicates.any():
        raise ValueError(
            f"Found {int(duplicates.sum()):,} duplicate user/day rows. "
            "Resolve these before calculating historical baselines."
        )


    # --------------------------------------------------
    # BUILD PAST-ONLY USER BASELINES
    # --------------------------------------------------

    print("Building past-only user baselines...")

    grouped = df.groupby("user", sort=False)

    clean_zscore_columns = []

    for col in BASELINE_COLUMNS:
        print(f"Processing: {col}")

        # Exclude the current day from its own baseline.
        history = grouped[col].shift(1)

        # Calculate expanding statistics using only previous rows.
        historical_mean = (
            history.groupby(df["user"], sort=False)
            .expanding(min_periods=MIN_HISTORY)
            .mean()
            .reset_index(level=0, drop=True)
            .reindex(df.index)
        )

        historical_std = (
            history.groupby(df["user"], sort=False)
            .expanding(min_periods=MIN_HISTORY)
            .std()
            .reset_index(level=0, drop=True)
            .reindex(df.index)
        )

        df[f"{col}_baseline_mean"] = historical_mean
        df[f"{col}_baseline_std"] = historical_std

        # Avoid division by zero or invalid standard deviations.
        valid_std = historical_std.where(historical_std > 0)

        zscore_col = f"{col}_clean_zscore"

        zscores = (
            (df[col] - historical_mean) / valid_std
        )

        df[zscore_col] = (
            zscores
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

        clean_zscore_columns.append(zscore_col)


    # --------------------------------------------------
    # OVERALL BEHAVIOR DEVIATION
    # --------------------------------------------------

    print("Calculating clean behavior deviation...")

    df["clean_behavior_deviation"] = (
        df[clean_zscore_columns]
        .abs()
        .mean(axis=1)
    )


    # --------------------------------------------------
    # VALIDATE AND SAVE
    # --------------------------------------------------

    print("Validating output...")

    if len(df) == 0:
        raise ValueError("The input dataset is empty.")

    if df[["user", "day"]].isna().any().any():
        raise ValueError(
            "Missing user/day values detected."
        )

    if not df["clean_behavior_deviation"].notna().all():
        raise ValueError(
            "Invalid clean_behavior_deviation values detected."
        )

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")
    print(f"Users: {df['user'].nunique():,}")

    if "ground_truth" in df.columns:
        malicious_days = int(
            (df["ground_truth"] == 1).sum()
        )
        print(f"Ground-truth malicious days: {malicious_days:,}")

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        OUTPUT_FILE,
        engine=PARQUET_ENGINE,
        index=False,
    )

    print(f"\nSaved: {OUTPUT_FILE}")
    print("Baseline feature generation completed.")


if __name__ == "__main__":
    main()
