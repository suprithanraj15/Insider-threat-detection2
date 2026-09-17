from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOGON_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "logon.parquet"
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
    print("LOGIN RAW BEHAVIOR ANALYSIS")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    print("\nLoading logon data...")

    logon = pd.read_parquet(
        LOGON_FILE
    )

    print(
        f"Logon rows : {len(logon):,}"
    )

    print("\nLoading ground truth...")

    truth = pd.read_parquet(
        GROUND_TRUTH_FILE
    )

    print(
        f"Truth rows : {len(truth):,}"
    )

    # --------------------------------------------------------
    # 2. NORMALIZE LOGON DATA
    # --------------------------------------------------------

    logon["user"] = (
        logon["user"]
        .astype(str)
        .str.strip()
    )

    logon["pc"] = (
        logon["pc"]
        .astype(str)
        .str.strip()
    )

    logon["date"] = pd.to_datetime(
        logon["date"],
        errors="coerce"
    )

    logon["day"] = (
        logon["date"]
        .dt.normalize()
    )

    # --------------------------------------------------------
    # 3. NORMALIZE GROUND TRUTH
    # --------------------------------------------------------

    truth["user"] = (
        truth["user"]
        .astype(str)
        .str.strip()
    )

    truth["day"] = pd.to_datetime(
        truth["day"],
        errors="coerce"
    ).dt.normalize()

    # --------------------------------------------------------
    # 4. SELECT MALICIOUS USER-DAYS
    # --------------------------------------------------------

    malicious = truth[
        truth["ground_truth"] == 1
    ][
        [
            "user",
            "day",
            "scenario"
        ]
    ].copy()

    print(
        f"\nMalicious user-days: "
        f"{len(malicious)}"
    )

    # --------------------------------------------------------
    # 5. MERGE RAW LOGON EVENTS WITH MALICIOUS DAYS
    # --------------------------------------------------------

    events = logon.merge(
        malicious,
        on=[
            "user",
            "day"
        ],
        how="inner"
    )

    print(
        f"Raw logon events belonging "
        f"to malicious days: {len(events):,}"
    )

    # --------------------------------------------------------
    # 6. LOGIN HOUR
    # --------------------------------------------------------

    events["hour"] = (
        events["date"]
        .dt.hour
    )

    # --------------------------------------------------------
    # 7. MALICIOUS LOGIN PC USAGE
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("MALICIOUS LOGIN PC USAGE")
    print("=" * 80)

    pc_usage = (
        events
        .groupby(
            [
                "user",
                "pc"
            ]
        )
        .size()
        .reset_index(
            name="events"
        )
        .sort_values(
            [
                "user",
                "events"
            ],
            ascending=[
                True,
                False
            ]
        )
    )

    if len(pc_usage) > 0:

        print(
            pc_usage.to_string(
                index=False
            )
        )

    else:

        print(
            "No malicious PC usage records found."
        )

    # --------------------------------------------------------
    # 8. MALICIOUS LOGIN HOURS
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("MALICIOUS LOGIN HOURS")
    print("=" * 80)

    hour_distribution = (
        events
        .groupby("hour")
        .size()
        .reset_index(
            name="events"
        )
        .sort_values("hour")
    )

    if len(hour_distribution) > 0:

        print(
            hour_distribution.to_string(
                index=False
            )
        )

    else:

        print(
            "No malicious login hours found."
        )

    # --------------------------------------------------------
    # 9. USER-PC COMBINATIONS
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("MALICIOUS USER-PC COMBINATIONS")
    print("=" * 80)

    user_pc = (
        events
        .groupby(
            [
                "user",
                "pc"
            ]
        )
        .agg(
            events=("pc", "size"),
            first_seen=("date", "min"),
            last_seen=("date", "max")
        )
        .reset_index()
    )

    if len(user_pc) > 0:

        print(
            user_pc.to_string(
                index=False
            )
        )

    else:

        print(
            "No user-PC combinations found."
        )

    # --------------------------------------------------------
    # 10. LOGIN ACTIVITY TYPES
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("MALICIOUS LOGIN ACTIVITY TYPES")
    print("=" * 80)

    print(
        events["activity"]
        .value_counts()
    )

    # --------------------------------------------------------
    # 11. PER MALICIOUS USER-DAY LOGIN DETAILS
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("MALICIOUS USER-DAY LOGIN DETAILS")
    print("=" * 80)

    daily = (
        events
        .groupby(
            [
                "user",
                "day",
                "scenario"
            ]
        )
        .agg(
            login_events=("date", "size"),
            unique_pcs=("pc", "nunique"),
            first_login=("date", "min"),
            last_login=("date", "max")
        )
        .reset_index()
    )

    daily["first_login_hour"] = (
        daily["first_login"]
        .dt.hour
    )

    daily["last_login_hour"] = (
        daily["last_login"]
        .dt.hour
    )

    if len(daily) > 0:

        print(
            daily.to_string(
                index=False
            )
        )

    else:

        print(
            "No malicious user-day login details found."
        )

    # --------------------------------------------------------
    # 12. MULTIPLE-PC MALICIOUS DAYS
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("MALICIOUS DAYS WITH MULTIPLE PCS")
    print("=" * 80)

    multi_pc = daily[
        daily["unique_pcs"] > 1
    ]

    if len(multi_pc) > 0:

        print(
            multi_pc[
                [
                    "user",
                    "day",
                    "scenario",
                    "login_events",
                    "unique_pcs",
                    "first_login_hour",
                    "last_login_hour"
                ]
            ].to_string(
                index=False
            )
        )

    else:

        print(
            "No malicious days used multiple PCs."
        )

    # --------------------------------------------------------
    # 13. SAVE ANALYSIS
    # --------------------------------------------------------

    output_file = (
        PROJECT_ROOT
        / "data"
        / "outputs"
        / "login_raw_malicious_analysis.csv"
    )

    daily.to_csv(
        output_file,
        index=False
    )

    print("\n" + "=" * 80)
    print("ANALYSIS SAVED")
    print("=" * 80)

    print(output_file)

    print("\n" + "=" * 80)
    print("LOGIN RAW BEHAVIOR ANALYSIS COMPLETED")
    print("=" * 80)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()