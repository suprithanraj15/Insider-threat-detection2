from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BEHAVIORAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "final_behavioral_features.parquet"
)

TEMPORAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "temporal_behavioral_features.parquet"
)

RULE_CONTEXT_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "rule_context_features.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "login_agent_results.parquet"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("LOGIN AGENT V2")
    print("Behavioral + Temporal Login Analysis")
    print("=" * 70)

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    print("\nLoading behavioral features...")

    behavioral = pd.read_parquet(
        BEHAVIORAL_FILE
    )

    print(
        f"Behavioral rows: {len(behavioral):,}"
    )

    print("\nLoading temporal features...")

    temporal = pd.read_parquet(
        TEMPORAL_FILE
    )

    print(
        f"Temporal rows: {len(temporal):,}"
    )

    print("\nLoading rule context...")

    context = pd.read_parquet(
        RULE_CONTEXT_FILE
    )

    print(
        f"Context rows: {len(context):,}"
    )

    # ========================================================
    # 2. NORMALIZE KEYS
    # ========================================================

    for data in [
        behavioral,
        temporal,
        context
    ]:

        data["user"] = (
            data["user"]
            .astype(str)
            .str.strip()
        )

        data["day"] = pd.to_datetime(
            data["day"],
            errors="coerce"
        ).dt.normalize()

    # ========================================================
    # 3. SELECT BEHAVIORAL FEATURES
    # ========================================================

    behavioral_columns = [
        "user",
        "day",
        "logon_events",
        "logon_pc_count",
        "logon_logins",
        "logon_logoffs",
        "logon_events_clean_zscore"
    ]

    behavioral = behavioral[
        [
            column
            for column in behavioral_columns
            if column in behavioral.columns
        ]
    ].copy()

    # ========================================================
    # 4. SELECT ACTUAL TEMPORAL FEATURES
    # ========================================================
    #
    # IMPORTANT:
    # These are the REAL column names in our file.
    #
    # logon_events_prev7_mean
    # logon_events_change_7d
    # ========================================================

    temporal_columns = [
        "user",
        "day",
        "logon_events_prev7_mean",
        "logon_events_change_7d",
        "temporal_change_count",
        "temporal_change_score",
        "persistent_behavior_flag"
    ]

    temporal = temporal[
        [
            column
            for column in temporal_columns
            if column in temporal.columns
        ]
    ].copy()

    # ========================================================
    # 5. SELECT CONTEXT FEATURES
    # ========================================================

    context_columns = [
        "user",
        "day",
        "after_hours_logons",
        "weekend_logons",
        "unique_logon_pcs",
        "total_logon_events",
        "after_hours_logon_ratio"
    ]

    context = context[
        [
            column
            for column in context_columns
            if column in context.columns
        ]
    ].copy()

    # ========================================================
    # 6. MERGE
    # ========================================================

    df = behavioral.merge(
        temporal,
        on=["user", "day"],
        how="left"
    )

    df = df.merge(
        context,
        on=["user", "day"],
        how="left"
    )

    print(
        f"\nMerged rows: {len(df):,}"
    )

    # ========================================================
    # 7. CHECK TEMPORAL FEATURES
    # ========================================================

    required_temporal = [
        "logon_events_prev7_mean",
        "logon_events_change_7d"
    ]

    print("\nTemporal feature availability:")

    for column in required_temporal:

        if column in df.columns:

            non_zero = (
                df[column]
                .fillna(0)
                .ne(0)
                .sum()
            )

            print(
                f"{column:35s}: "
                f"available | non-zero: {non_zero:,}"
            )

        else:

            print(
                f"{column:35s}: MISSING"
            )

    # ========================================================
    # 8. NUMERIC CONVERSION
    # ========================================================

    numeric_columns = [
        "logon_events",
        "logon_pc_count",
        "logon_logins",
        "logon_logoffs",
        "logon_events_clean_zscore",
        "logon_events_prev7_mean",
        "logon_events_change_7d",
        "temporal_change_count",
        "temporal_change_score",
        "persistent_behavior_flag",
        "after_hours_logons",
        "weekend_logons",
        "unique_logon_pcs",
        "total_logon_events",
        "after_hours_logon_ratio"
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            ).fillna(0)

    # ========================================================
    # 9. LOGIN SIGNAL 1
    # ========================================================
    #
    # Individual behavioral anomaly.
    #

    df["login_zscore_signal"] = (
        df["logon_events_clean_zscore"] >= 2.5
    ).astype(int)

    # ========================================================
    # 10. LOGIN SIGNAL 2
    # ========================================================
    #
    # Strong individual anomaly.
    #

    df["login_strong_zscore_signal"] = (
        df["logon_events_clean_zscore"] >= 3.0
    ).astype(int)

    # ========================================================
    # 11. LOGIN SIGNAL 3
    # ========================================================
    #
    # Sudden change compared with previous 7-day baseline.
    #
    # ACTUAL COLUMN:
    # logon_events_change_7d
    #

    df["login_temporal_change_signal"] = (
        df["logon_events_change_7d"].abs() >= 1.0
    ).astype(int)

    # ========================================================
    # 12. LOGIN SIGNAL 4
    # ========================================================
    #
    # Strong temporal change.
    #

    df["login_strong_temporal_signal"] = (
        df["logon_events_change_7d"].abs() >= 2.0
    ).astype(int)

    # ========================================================
    # 13. LOGIN SIGNAL 5
    # ========================================================
    #
    # After-hours login.
    #
    # Supporting evidence only.
    #

    df["login_after_hours_signal"] = (
        df["after_hours_logons"] > 0
    ).astype(int)

    # ========================================================
    # 14. LOGIN SIGNAL 6
    # ========================================================
    #
    # Multiple PCs.
    #
    # Supporting evidence only.
    #

    df["login_multiple_pc_signal"] = (
        df["unique_logon_pcs"] >= 2
    ).astype(int)

    # ========================================================
    # 15. LOGIN SIGNAL 7
    # ========================================================

    df["login_high_volume_signal"] = (
        df["total_logon_events"] >= 5
    ).astype(int)

    # ========================================================
    # 16. LOGIN SIGNAL 8
    # ========================================================

    df["login_after_hours_ratio_signal"] = (
        df["after_hours_logon_ratio"] >= 0.5
    ).astype(int)

    # ========================================================
    # 17. STRONG COMBINATION:
    # BEHAVIOR + TEMPORAL CHANGE
    # ========================================================

    df["login_behavior_change_signal"] = (
        (
            df["login_zscore_signal"] == 1
        )
        &
        (
            df["login_temporal_change_signal"] == 1
        )
    ).astype(int)

    # ========================================================
    # 18. AFTER-HOURS + ANOMALY
    # ========================================================

    df["login_after_hours_anomaly_signal"] = (
        (
            df["login_after_hours_signal"] == 1
        )
        &
        (
            df["login_zscore_signal"] == 1
        )
    ).astype(int)

    # ========================================================
    # 19. EVIDENCE SCORE
    # ========================================================

    df["login_evidence_score"] = (

        df["login_zscore_signal"] * 2.0

        +

        df["login_strong_zscore_signal"] * 1.0

        +

        df["login_temporal_change_signal"] * 2.0

        +

        df["login_strong_temporal_signal"] * 1.0

        +

        df["login_after_hours_signal"] * 0.5

        +

        df["login_multiple_pc_signal"] * 0.5

        +

        df["login_high_volume_signal"] * 1.0

        +

        df["login_after_hours_ratio_signal"] * 1.0

        +

        df["login_behavior_change_signal"] * 2.0

        +

        df["login_after_hours_anomaly_signal"] * 1.5
    )

    # ========================================================
    # 20. PRIMARY EVIDENCE COUNT
    # ========================================================
    #
    # Only stronger login indicators.
    #

    df["login_evidence_count"] = (

        df["login_zscore_signal"]

        +

        df["login_temporal_change_signal"]

        +

        df["login_high_volume_signal"]

        +

        df["login_after_hours_ratio_signal"]

        +

        df["login_behavior_change_signal"]

        +

        df["login_after_hours_anomaly_signal"]
    )

    # ========================================================
    # 21. AGENT STATUS
    # ========================================================

    def determine_status(row):

        score = row["login_evidence_score"]

        count = row["login_evidence_count"]

        if (
            score >= 7
            or count >= 3
        ):

            return "STRONG_EVIDENCE"

        elif (
            score >= 4
            or count >= 2
        ):

            return "MODERATE_EVIDENCE"

        elif (
            score > 0
            or row["login_multiple_pc_signal"] == 1
            or row["login_after_hours_signal"] == 1
        ):

            return "WEAK_EVIDENCE"

        else:

            return "NO_LOGIN_EVIDENCE"

    df["login_agent_status"] = df.apply(
        determine_status,
        axis=1
    )

    # ========================================================
    # 22. NORMALIZED AGENT SCORE
    # ========================================================

    max_score = (
        df["login_evidence_score"]
        .max()
    )

    if max_score > 0:

        df["login_agent_score"] = (
            df["login_evidence_score"]
            / max_score
            * 100
        )

    else:

        df["login_agent_score"] = 0.0

    # ========================================================
    # 23. EXPLANATION
    # ========================================================

    def build_explanation(row):

        reasons = []

        if row["login_zscore_signal"]:

            reasons.append(
                "unusual login volume"
            )

        if row["login_strong_zscore_signal"]:

            reasons.append(
                "strong login deviation"
            )

        if row["login_temporal_change_signal"]:

            reasons.append(
                "login behavior changed from previous 7-day baseline"
            )

        if row["login_strong_temporal_signal"]:

            reasons.append(
                "strong temporal login change"
            )

        if row["login_after_hours_signal"]:

            reasons.append(
                "after-hours login"
            )

        if row["login_multiple_pc_signal"]:

            reasons.append(
                "multiple login PCs"
            )

        if row["login_high_volume_signal"]:

            reasons.append(
                "high login activity"
            )

        if row["login_after_hours_ratio_signal"]:

            reasons.append(
                "high after-hours login ratio"
            )

        if row["login_behavior_change_signal"]:

            reasons.append(
                "login anomaly combined with temporal change"
            )

        if row["login_after_hours_anomaly_signal"]:

            reasons.append(
                "after-hours login combined with behavioral anomaly"
            )

        if not reasons:

            return "No significant login evidence"

        return "; ".join(reasons)

    df["login_explanation"] = df.apply(
        build_explanation,
        axis=1
    )

    # ========================================================
    # 24. OUTPUT COLUMNS
    # ========================================================

    output_columns = [
        "user",
        "day",

        "logon_events",
        "logon_pc_count",
        "logon_logins",
        "logon_logoffs",

        "logon_events_clean_zscore",

        "logon_events_prev7_mean",
        "logon_events_change_7d",

        "temporal_change_count",
        "temporal_change_score",
        "persistent_behavior_flag",

        "after_hours_logons",
        "weekend_logons",
        "unique_logon_pcs",
        "total_logon_events",
        "after_hours_logon_ratio",

        "login_zscore_signal",
        "login_strong_zscore_signal",
        "login_temporal_change_signal",
        "login_strong_temporal_signal",
        "login_after_hours_signal",
        "login_multiple_pc_signal",
        "login_high_volume_signal",
        "login_after_hours_ratio_signal",
        "login_behavior_change_signal",
        "login_after_hours_anomaly_signal",

        "login_evidence_count",
        "login_evidence_score",
        "login_agent_score",
        "login_agent_status",
        "login_explanation"
    ]

    output_columns = [
        column
        for column in output_columns
        if column in df.columns
    ]

    output = df[
        output_columns
    ].copy()

    # ========================================================
    # 25. SAVE
    # ========================================================

    output.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # 26. SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("LOGIN AGENT V2 SUMMARY")
    print("=" * 70)

    print(
        f"\nRows processed : "
        f"{len(output):,}"
    )

    print("\nLogin Agent status:")

    print(
        output[
            "login_agent_status"
        ].value_counts()
    )

    print("\nEvidence count:")

    print(
        output[
            "login_evidence_count"
        ]
        .value_counts()
        .sort_index()
    )

    print("\nSignal counts:")

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

        print(
            f"{column:40s}: "
            f"{int(output[column].sum()):,}"
        )

    print(
        "\nAverage login evidence score: "
        f"{output['login_evidence_score'].mean():.4f}"
    )

    print(
        "Maximum login evidence score: "
        f"{output['login_evidence_score'].max():.4f}"
    )

    print(
        "\nOutput:"
    )

    print(
        OUTPUT_FILE
    )

    print("\n" + "=" * 70)
    print("LOGIN AGENT V2 COMPLETED")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()