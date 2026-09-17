from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

AGENT_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "login_agent_results.parquet"
)

GROUND_TRUTH_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "user_day_features_labeled.parquet"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("LOGIN AGENT V2 - MALICIOUS DAY DIAGNOSTIC")
    print("=" * 80)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    agent = pd.read_parquet(AGENT_FILE)

    truth = pd.read_parquet(GROUND_TRUTH_FILE)

    print(f"\nAgent rows : {len(agent):,}")
    print(f"Truth rows : {len(truth):,}")

    # --------------------------------------------------------
    # Normalize keys
    # --------------------------------------------------------

    agent["user"] = (
        agent["user"]
        .astype(str)
        .str.strip()
    )

    truth["user"] = (
        truth["user"]
        .astype(str)
        .str.strip()
    )

    agent["day"] = pd.to_datetime(
        agent["day"],
        errors="coerce"
    ).dt.normalize()

    truth["day"] = pd.to_datetime(
        truth["day"],
        errors="coerce"
    ).dt.normalize()

    # --------------------------------------------------------
    # Select ground truth
    # --------------------------------------------------------

    truth = truth[
        [
            "user",
            "day",
            "ground_truth",
            "scenario"
        ]
    ].copy()

    truth = truth.rename(
        columns={
            "ground_truth": "gt_label",
            "scenario": "gt_scenario"
        }
    )

    truth["gt_label"] = (
        pd.to_numeric(
            truth["gt_label"],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    df = agent.merge(
        truth,
        on=["user", "day"],
        how="inner"
    )

    print(f"Merged rows: {len(df):,}")

    if len(df) != len(agent):

        raise ValueError(
            "Merge mismatch. "
            "Agent and ground truth keys do not fully match."
        )

    # --------------------------------------------------------
    # Current V2 prediction
    # --------------------------------------------------------

    df["v2_prediction"] = (
        df["login_agent_status"]
        .isin(
            [
                "MODERATE_EVIDENCE",
                "STRONG_EVIDENCE"
            ]
        )
        .astype(int)
    )

    # --------------------------------------------------------
    # Malicious days only
    # --------------------------------------------------------

    malicious = df[
        df["gt_label"] == 1
    ].copy()

    malicious = malicious.sort_values(
        [
            "gt_scenario",
            "user",
            "day"
        ]
    )

    print("\n" + "=" * 80)
    print(
        f"TOTAL MALICIOUS USER-DAYS: "
        f"{len(malicious)}"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # Columns to inspect
    # --------------------------------------------------------

    columns = [
        "user",
        "day",
        "gt_scenario",

        "v2_prediction",

        "login_agent_status",
        "login_evidence_count",
        "login_agent_score",

        "logon_events",
        "logon_events_clean_zscore",

        "logon_events_prev7_mean",
        "logon_events_change_7d",

        "after_hours_logons",
        "after_hours_logon_ratio",

        "unique_logon_pcs",
        "total_logon_events",

        "login_zscore_signal",
        "login_strong_zscore_signal",
        "login_temporal_change_signal",
        "login_strong_temporal_signal",
        "login_after_hours_signal",
        "login_multiple_pc_signal",
        "login_high_volume_signal",
        "login_after_hours_ratio_signal",
        "login_behavior_change_signal",
        "login_after_hours_anomaly_signal"
    ]

    existing_columns = [
        column
        for column in columns
        if column in malicious.columns
    ]

    print("\nMALICIOUS DAY DETAILS:\n")

    print(
        malicious[
            existing_columns
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Detected / missed
    # --------------------------------------------------------

    detected = malicious[
        malicious["v2_prediction"] == 1
    ]

    missed = malicious[
        malicious["v2_prediction"] == 0
    ]

    print("\n" + "=" * 80)
    print("V2 DETECTION SUMMARY")
    print("=" * 80)

    print(
        f"Detected malicious days : "
        f"{len(detected)}/{len(malicious)}"
    )

    print(
        f"Missed malicious days   : "
        f"{len(missed)}/{len(malicious)}"
    )

    # --------------------------------------------------------
    # Scenario summary
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("SCENARIO SUMMARY")
    print("=" * 80)

    scenario_summary = (
        malicious
        .groupby("gt_scenario")
        .agg(
            total_days=("gt_label", "size"),
            detected_days=("v2_prediction", "sum")
        )
        .reset_index()
    )

    scenario_summary["recall"] = (
        scenario_summary["detected_days"]
        / scenario_summary["total_days"]
    )

    print(
        scenario_summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Signal coverage on malicious days
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("SIGNAL COVERAGE ON MALICIOUS DAYS")
    print("=" * 80)

    signal_columns = [
        "login_zscore_signal",
        "login_strong_zscore_signal",
        "login_temporal_change_signal",
        "login_strong_temporal_signal",
        "login_after_hours_signal",
        "login_multiple_pc_signal",
        "login_high_volume_signal",
        "login_after_hours_ratio_signal",
        "login_behavior_change_signal",
        "login_after_hours_anomaly_signal"
    ]

    for column in signal_columns:

        if column not in malicious.columns:
            continue

        count = int(
            malicious[column]
            .sum()
        )

        percentage = (
            count / len(malicious)
            if len(malicious) > 0
            else 0
        )

        print(
            f"{column:40s}: "
            f"{count:2d}/"
            f"{len(malicious):2d} "
            f"({percentage:.2%})"
        )

    # --------------------------------------------------------
    # Missed malicious days
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("MISSED MALICIOUS DAYS")
    print("=" * 80)

    if len(missed) == 0:

        print(
            "No malicious days were missed."
        )

    else:

        print(
            missed[
                existing_columns
            ].to_string(index=False)
        )

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()