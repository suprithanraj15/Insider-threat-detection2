from pathlib import Path

import pandas as pd
import numpy as np

# Project paths
OUTPUT_DIR = Path("data/outputs")
ENGINE = "fastparquet"

# Approved agent outputs and the columns required for integration
AGENT_FILES = {
    "rule": (
        "final_rule_agent_xgb_results.parquet",
        ["final_rule_score", "xgb_probability", "final_threat_level"],
    ),
    "login": (
        "login_agent_results.parquet",
        ["login_agent_score", "login_agent_status", "login_explanation"],
    ),
    "file": (
        "file_agent_v2_results.parquet",
        ["file_agent_score", "file_agent_status", "file_explanation"],
    ),
    "device": (
        "device_agent_results.parquet",
        ["device_agent_score", "device_agent_status", "device_explanation"],
    ),
    "network": (
        "network_agent_v2_results.parquet",
        ["network_v2_score", "network_v2_status"],
    ),
}

GROUND_TRUTH_FILE = "user_day_features_labeled.parquet"
OUTPUT_FILE = "agent_integration_dataset.parquet"


def normalize_day(series):
    """Normalize dates, including integer nanosecond timestamps."""
    first = series.dropna().iloc[0]

    if pd.api.types.is_numeric_dtype(series.dtype) or isinstance(
        first, (int, np.integer)
    ):
        dates = pd.to_datetime(series, unit="ns", errors="raise")
    else:
        dates = pd.to_datetime(series, errors="raise")

    return dates.dt.normalize()


def prepare_keys(df, name):
    """Normalize user/day keys and reject missing or duplicate keys."""
    if not {"user", "day"}.issubset(df.columns):
        raise ValueError(f"{name}: missing user/day columns")

    result = pd.DataFrame({
        "user": df["user"].astype("string").str.strip().str.upper(),
        "day": normalize_day(df["day"]),
    })

    if result.isna().any().any():
        raise ValueError(f"{name}: missing user/day keys")

    if result.duplicated(["user", "day"]).any():
        raise ValueError(f"{name}: duplicate user/day keys")

    return result


def main():
    print("Loading global ground truth...")

    gt = pd.read_parquet(
        OUTPUT_DIR / GROUND_TRUTH_FILE,
        engine=ENGINE,
    )

    if "ground_truth" not in gt.columns:
        raise ValueError("Global ground-truth file has no ground_truth column")

    gt_keys = prepare_keys(gt, "Ground truth")
    integrated = gt_keys.copy()

    # The target is sourced only from the global ground-truth file.
    integrated["ground_truth"] = gt["ground_truth"].to_numpy()

    if integrated["ground_truth"].isna().any():
        raise ValueError("Ground truth contains missing labels")

    print(f"Ground-truth rows: {len(integrated):,}")

    for agent_name, (filename, columns) in AGENT_FILES.items():
        path = OUTPUT_DIR / filename

        print(f"\nLoading {agent_name} agent: {filename}")

        agent = pd.read_parquet(path, engine=ENGINE)

        missing_columns = [
            col for col in columns if col not in agent.columns
        ]
        if missing_columns:
            raise ValueError(
                f"{agent_name}: missing required columns {missing_columns}"
            )

        keys = prepare_keys(agent, f"{agent_name} agent")

        selected = keys.copy()
        for col in columns:
            selected[f"{agent_name}_{col}"] = agent[col].to_numpy()

        # Validate exact key coverage before joining.
        if len(selected) != len(integrated):
            raise ValueError(
                f"{agent_name}: row count differs from integration dataset"
            )

        key_check = integrated[["user", "day"]].merge(
            selected[["user", "day"]],
            on=["user", "day"],
            how="outer",
            indicator=True,
            validate="one_to_one",
        )

        unmatched = int((key_check["_merge"] != "both").sum())
        if unmatched:
            raise ValueError(
                f"{agent_name}: {unmatched} unmatched user/day keys"
            )

        # Keep the integration one row per user-day.
        integrated = integrated.merge(
            selected,
            on=["user", "day"],
            how="left",
            validate="one_to_one",
        )

        print(f"{agent_name} merge: PASS")

    # Check score columns for missing and non-numeric values.
    score_columns = [
        "rule_final_rule_score",
        "rule_xgb_probability",
        "login_login_agent_score",
        "file_file_agent_score",
        "device_device_agent_score",
        "network_network_v2_score",
    ]

    for col in score_columns:
        values = pd.to_numeric(integrated[col], errors="coerce")

        if values.isna().any():
            raise ValueError(
                f"{col}: missing or non-numeric values detected"
            )

        integrated[col] = values

    if integrated.duplicated(["user", "day"]).any():
        raise ValueError("Final integration dataset has duplicate user-days")

    if len(integrated) != len(gt):
        raise ValueError("Final row count does not match ground truth")

    # Save to a new file; approved agent outputs remain unchanged.
    output_path = OUTPUT_DIR / OUTPUT_FILE
    integrated.to_parquet(
        output_path,
        engine=ENGINE,
        index=False,
    )

    print("\n" + "=" * 60)
    print("INTEGRATION COMPLETED")
    print("=" * 60)
    print(f"Output: {output_path}")
    print(f"Rows: {len(integrated):,}")
    print(f"Columns: {len(integrated.columns)}")
    print(f"Unique users: {integrated['user'].nunique():,}")
    print(
        "Ground-truth distribution:\n"
        + integrated["ground_truth"].value_counts().sort_index().to_string()
    )
    print(
        "Duplicate user/day keys:",
        integrated.duplicated(["user", "day"]).sum(),
    )
    print(
        "Missing values in selected scores:",
        int(integrated[score_columns].isna().sum().sum()),
    )
    print("\nFirst five rows:")
    print(integrated.head().to_string(index=False))


if __name__ == "__main__":
    main()  