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
    / "login_ground_truth.parquet"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("LOGIN AGENT V2 - FALSE POSITIVE DIAGNOSTIC")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    agent = pd.read_parquet(
        AGENT_FILE
    )

    truth = pd.read_parquet(
        GROUND_TRUTH_FILE
    )

    print(
        f"\nAgent rows : {len(agent):,}"
    )

    print(
        f"Truth rows: {len(truth):,}"
    )

    # --------------------------------------------------------
    # 2. NORMALIZE KEYS
    # --------------------------------------------------------

    agent["user"] = (
        agent["user"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    truth["user"] = (
        truth["user"]
        .astype(str)
        .str.strip()
        .str.lower()
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
    # 3. MERGE
    # --------------------------------------------------------

    df = agent.merge(
        truth[
            [
                "user",
                "day",
                "login_ground_truth",
                "scenario"
            ]
        ],
        on=[
            "user",
            "day"
        ],
        how="inner"
    )

    print(
        f"Merged rows: {len(df):,}"
    )

    # --------------------------------------------------------
    # 4. CURRENT PREDICTION
    # --------------------------------------------------------

    df["prediction"] = (
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
    # 5. SEPARATE TP / FP / FN
    # --------------------------------------------------------

    tp = df[
        (df["prediction"] == 1)
        &
        (df["login_ground_truth"] == 1)
    ].copy()

    fp = df[
        (df["prediction"] == 1)
        &
        (df["login_ground_truth"] == 0)
    ].copy()

    fn = df[
        (df["prediction"] == 0)
        &
        (df["login_ground_truth"] == 1)
    ].copy()

    print("\n" + "=" * 80)
    print("CURRENT CONFUSION GROUPS")
    print("=" * 80)

    print(
        f"True Positives : {len(tp):,}"
    )

    print(
        f"False Positives: {len(fp):,}"
    )

    print(
        f"False Negatives: {len(fn):,}"
    )

    # --------------------------------------------------------
    # 6. SIGNAL COLUMNS
    # --------------------------------------------------------

    signal_columns = [
        column
        for column in df.columns
        if column.startswith("login_")
        and column.endswith("_signal")
    ]

    print("\n" + "=" * 80)
    print("SIGNAL COMPARISON")
    print("=" * 80)

    comparison = []

    for signal in signal_columns:

        tp_rate = (
            tp[signal].mean()
            if len(tp) > 0
            else 0
        )

        fp_rate = (
            fp[signal].mean()
            if len(fp) > 0
            else 0
        )

        fn_rate = (
            fn[signal].mean()
            if len(fn) > 0
            else 0
        )

        comparison.append(
            {
                "signal": signal,
                "TP_rate": tp_rate,
                "FP_rate": fp_rate,
                "FN_rate": fn_rate,
                "TP_count": int(tp[signal].sum()),
                "FP_count": int(fp[signal].sum()),
                "FN_count": int(fn[signal].sum()),
            }
        )

    comparison_df = pd.DataFrame(
        comparison
    )

    comparison_df[
        "discrimination_ratio"
    ] = (
        comparison_df["TP_rate"]
        /
        comparison_df["FP_rate"].replace(
            0,
            1e-9
        )
    )

    comparison_df = comparison_df.sort_values(
        "discrimination_ratio",
        ascending=False
    )

    print(
        comparison_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 7. SCORE DISTRIBUTION
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("SCORE DISTRIBUTION")
    print("=" * 80)

    print(
        "\nTrue positives:"
    )

    print(
        tp[
            [
                "user",
                "day",
                "login_evidence_count",
                "login_agent_score",
                "login_agent_status"
            ]
        ]
        .sort_values(
            "login_agent_score",
            ascending=False
        )
        .to_string(
            index=False
        )
    )

    print(
        "\nFalse negatives:"
    )

    print(
        fn[
            [
                "user",
                "day",
                "login_evidence_count",
                "login_agent_score",
                "login_agent_status"
            ]
        ]
        .sort_values(
            "login_agent_score",
            ascending=False
        )
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 8. FALSE POSITIVE SCORE DISTRIBUTION
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("FALSE POSITIVE SCORE DISTRIBUTION")
    print("=" * 80)

    print(
        fp[
            [
                "login_evidence_count",
                "login_agent_score",
                "login_agent_status"
            ]
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # 9. FALSE POSITIVE SIGNAL COMBINATIONS
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("TOP FALSE POSITIVE SIGNAL COMBINATIONS")
    print("=" * 80)

    if len(fp) > 0:

        fp_combinations = (
            fp[signal_columns]
            .astype(int)
            .astype(str)
            .agg(
                ",".join,
                axis=1
            )
            .value_counts()
            .head(20)
        )

        print(
            fp_combinations.to_string()
        )

    # --------------------------------------------------------
    # 10. MALICIOUS LOGIN DAYS WITH ALL SIGNALS
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("ALL MALICIOUS LOGIN DAYS")
    print("=" * 80)

    display_columns = [
        "user",
        "day",
        "scenario",
        "login_evidence_count",
        "login_agent_score",
        "login_agent_status"
    ] + signal_columns

    print(
        df[
            df["login_ground_truth"] == 1
        ][display_columns]
        .sort_values(
            [
                "user",
                "day"
            ]
        )
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 11. SAVE DIAGNOSTIC
    # --------------------------------------------------------

    output_file = (
        PROJECT_ROOT
        / "data"
        / "outputs"
        / "login_false_positive_diagnostic.csv"
    )

    comparison_df.to_csv(
        output_file,
        index=False
    )

    print("\n" + "=" * 80)
    print("DIAGNOSTIC SAVED")
    print("=" * 80)

    print(
        output_file
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()