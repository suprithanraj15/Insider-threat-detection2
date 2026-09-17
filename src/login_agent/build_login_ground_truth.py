from pathlib import Path
import csv
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ANSWERS_DIR = PROJECT_ROOT / "CERT_Dataset" / "answers"

GROUND_TRUTH_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "user_day_features_labeled.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "login_ground_truth.parquet"
)


# ============================================================
# EXPECTED CERT ANSWER FILE COUNTS
# ============================================================

EXPECTED_COUNTS = {
    "r4.1-1.csv": 10,
    "r4.1-2.csv": 246,
    "r4.1-3.csv": 17,
}


# ============================================================
# READ CERT ANSWER FILE
# ============================================================

def read_answer_file(answer_file):

    records = []

    with open(
        answer_file,
        "r",
        encoding="utf-8",
        errors="replace",
        newline=""
    ) as f:

        reader = csv.reader(f)

        for line_number, row in enumerate(reader, start=1):

            if not row:
                continue

            # CERT answer files are headerless.
            # Required fields:
            # 0 = event type
            # 1 = event id
            # 2 = date
            # 3 = user
            # 4 = pc
            # 5 = activity

            if len(row) < 6:
                raise ValueError(
                    f"Malformed row in {answer_file.name} "
                    f"at line {line_number}: "
                    f"expected at least 6 fields, "
                    f"found {len(row)}"
                )

            records.append(
                {
                    "event_type": row[0].strip(),
                    "event_id": row[1].strip(),
                    "date": row[2].strip(),
                    "user": row[3].strip(),
                    "pc": row[4].strip(),
                    "activity": row[5].strip(),
                    "answer_file": answer_file.name,
                    "line_number": line_number,
                }
            )

    return pd.DataFrame(records)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("BUILD LOGIN-SPECIFIC GROUND TRUTH")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. CHECK ANSWERS DIRECTORY
    # --------------------------------------------------------

    print("\nChecking CERT answers directory...")

    if not ANSWERS_DIR.exists():
        raise FileNotFoundError(
            f"Answers directory not found:\n{ANSWERS_DIR}"
        )

    print(
        f"Answers directory: {ANSWERS_DIR}"
    )

    # --------------------------------------------------------
    # 2. LOAD EXISTING GROUND TRUTH
    # --------------------------------------------------------

    print("\nLoading existing ground truth...")

    truth = pd.read_parquet(
        GROUND_TRUTH_FILE
    )

    print(
        f"Ground truth rows: {len(truth):,}"
    )

    truth["user"] = (
        truth["user"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    truth["day"] = pd.to_datetime(
        truth["day"],
        errors="coerce"
    ).dt.normalize()

    # --------------------------------------------------------
    # 3. GET KNOWN MALICIOUS USER-DAYS
    # --------------------------------------------------------

    malicious_days = truth[
        truth["ground_truth"] == 1
    ][
        [
            "user",
            "day",
            "scenario"
        ]
    ].copy()

    print(
        f"Known malicious user-days: "
        f"{len(malicious_days)}"
    )

    # --------------------------------------------------------
    # 4. FIND ANSWER FILES
    # --------------------------------------------------------

    print(
        "\nReading CERT answer files..."
    )

    answer_files = sorted(
        ANSWERS_DIR.glob("r4.1-*.csv")
    )

    print(
        f"Answer files found: {len(answer_files)}"
    )

    if len(answer_files) == 0:
        raise FileNotFoundError(
            f"No r4.1 answer files found in:\n"
            f"{ANSWERS_DIR}"
        )

    # --------------------------------------------------------
    # 5. READ ALL ANSWER FILES
    # --------------------------------------------------------

    all_answer_records = []

    for answer_file in answer_files:

        print(
            f"\nReading: {answer_file.name}"
        )

        answer = read_answer_file(
            answer_file
        )

        print(
            f"Rows read: {len(answer):,}"
        )

        # ----------------------------------------------------
        # VERIFY EXPECTED ROW COUNT
        # ----------------------------------------------------

        expected_count = EXPECTED_COUNTS.get(
            answer_file.name
        )

        if expected_count is not None:

            if len(answer) != expected_count:
                raise ValueError(
                    f"Row count mismatch for "
                    f"{answer_file.name}: "
                    f"expected {expected_count}, "
                    f"found {len(answer)}"
                )

            print(
                f"Row count verified: "
                f"{expected_count}"
            )

        # ----------------------------------------------------
        # EVENT TYPE DISTRIBUTION
        # ----------------------------------------------------

        print(
            "Event types:"
        )

        print(
            answer["event_type"]
            .value_counts()
            .to_string()
        )

        # ----------------------------------------------------
        # ACTIVITY TYPE DISTRIBUTION
        # ----------------------------------------------------

        print(
            "Activity types:"
        )

        print(
            answer["activity"]
            .value_counts()
            .to_string()
        )

        all_answer_records.append(
            answer
        )

    # --------------------------------------------------------
    # 6. COMBINE ALL ANSWER RECORDS
    # --------------------------------------------------------

    all_answers = pd.concat(
        all_answer_records,
        ignore_index=True
    )

    print("\n" + "=" * 80)
    print("ALL CERT ANSWER RECORDS")
    print("=" * 80)

    print(
        f"Total answer records: "
        f"{len(all_answers):,}"
    )

    # --------------------------------------------------------
    # 7. NORMALIZE EVENT TYPE / ACTIVITY
    # --------------------------------------------------------

    all_answers["event_type"] = (
        all_answers["event_type"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    all_answers["activity"] = (
        all_answers["activity"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # --------------------------------------------------------
    # 8. EXTRACT MALICIOUS LOGON RECORDS
    # --------------------------------------------------------

    logon_answers = all_answers[
        (
            all_answers["event_type"] == "logon"
        )
        |
        (
            all_answers["activity"] == "logon"
        )
    ].copy()

    print(
        f"\nMalicious LOGON answer records: "
        f"{len(logon_answers):,}"
    )

    if len(logon_answers) == 0:
        raise ValueError(
            "No Logon records were found "
            "in the CERT answer files."
        )

    # --------------------------------------------------------
    # 9. NORMALIZE USER
    # --------------------------------------------------------

    logon_answers["user"] = (
        logon_answers["user"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # --------------------------------------------------------
    # 10. PARSE DATE
    # --------------------------------------------------------

    logon_answers["date"] = pd.to_datetime(
        logon_answers["date"],
        errors="coerce"
    )

    invalid_dates = (
        logon_answers["date"]
        .isna()
        .sum()
    )

    if invalid_dates > 0:
        raise ValueError(
            f"{invalid_dates} Logon answer records "
            f"have invalid dates."
        )

    logon_answers["day"] = (
        logon_answers["date"]
        .dt.normalize()
    )

    # --------------------------------------------------------
    # 11. CONVERT LOGON EVENTS TO USER-DAYS
    # --------------------------------------------------------

    malicious_logon_days = (
        logon_answers[
            [
                "user",
                "day"
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    malicious_logon_days[
        "login_ground_truth"
    ] = 1

    print(
        f"Unique malicious login user-days: "
        f"{len(malicious_logon_days):,}"
    )

    # --------------------------------------------------------
    # 12. MAP SCENARIO FROM GLOBAL GROUND TRUTH
    # --------------------------------------------------------

    malicious_logon_days = (
        malicious_logon_days.merge(
            malicious_days,
            on=[
                "user",
                "day"
            ],
            how="left"
        )
    )

    # --------------------------------------------------------
    # 13. VERIFY LOGIN DAYS EXIST IN GLOBAL GT
    # --------------------------------------------------------

    missing_from_global_gt = (
        malicious_logon_days["scenario"]
        .isna()
        .sum()
    )

    if missing_from_global_gt > 0:

        print(
            "\nWARNING:"
        )

        print(
            f"{missing_from_global_gt} malicious "
            f"login user-days were not found "
            f"in the global ground truth."
        )

        print(
            malicious_logon_days[
                malicious_logon_days["scenario"].isna()
            ][
                [
                    "user",
                    "day"
                ]
            ].to_string(
                index=False
            )
        )

        raise ValueError(
            "Login ground truth contains user-days "
            "missing from the global ground truth."
        )

    print(
        "All malicious login user-days "
        "matched the global ground truth."
    )

    # --------------------------------------------------------
    # 14. CREATE ALL USER-DAYS
    # --------------------------------------------------------

    all_user_days = truth[
        [
            "user",
            "day"
        ]
    ].drop_duplicates().copy()

    # Every user-day is normal for login
    # unless it appears in malicious login days.
    all_user_days[
        "login_ground_truth"
    ] = 0

    # --------------------------------------------------------
    # 15. MERGE MALICIOUS LOGIN DAYS
    # --------------------------------------------------------

    result = all_user_days.merge(
        malicious_logon_days[
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
        how="left",
        suffixes=(
            "_all",
            "_login"
        )
    )

    # --------------------------------------------------------
    # 16. CREATE FINAL LOGIN LABEL
    # --------------------------------------------------------

    result["login_ground_truth"] = (
        result["login_ground_truth_login"]
        .fillna(
            result["login_ground_truth_all"]
        )
        .astype(int)
    )

    # --------------------------------------------------------
    # 17. KEEP FINAL COLUMNS
    # --------------------------------------------------------

    result = result[
        [
            "user",
            "day",
            "login_ground_truth",
            "scenario"
        ]
    ].copy()

    # --------------------------------------------------------
    # 18. VERIFY USER-DAY COUNT
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)

    print(
        f"Original user-days : "
        f"{len(all_user_days):,}"
    )

    print(
        f"Result user-days   : "
        f"{len(result):,}"
    )

    if len(result) != len(all_user_days):
        raise ValueError(
            "ERROR: User-day row count changed."
        )

    print(
        "User-day count verified."
    )

    # --------------------------------------------------------
    # 19. LOGIN GROUND TRUTH DISTRIBUTION
    # --------------------------------------------------------

    print(
        "\nLogin ground truth distribution:"
    )

    distribution = (
        result["login_ground_truth"]
        .value_counts()
        .sort_index()
    )

    print(
        distribution.to_string()
    )

    # --------------------------------------------------------
    # 20. MALICIOUS LOGIN DAYS
    # --------------------------------------------------------

    malicious_result = result[
        result["login_ground_truth"] == 1
    ].copy()

    print("\n" + "=" * 80)
    print("MALICIOUS LOGIN DAYS BY USER")
    print("=" * 80)

    print(
        malicious_result
        .groupby("user")
        .size()
        .sort_values(
            ascending=False
        )
        .to_string()
    )

    # --------------------------------------------------------
    # 21. MALICIOUS LOGIN DAYS BY SCENARIO
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("MALICIOUS LOGIN DAYS BY SCENARIO")
    print("=" * 80)

    print(
        malicious_result
        .groupby("scenario")
        .size()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # 22. SHOW MALICIOUS LOGIN USER-DAYS
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("MALICIOUS LOGIN USER-DAYS")
    print("=" * 80)

    print(
        malicious_result
        .sort_values(
            [
                "scenario",
                "user",
                "day"
            ]
        )
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 23. SAVE
    # --------------------------------------------------------

    result.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 80)
    print("LOGIN GROUND TRUTH SAVED")
    print("=" * 80)

    print(
        OUTPUT_FILE
    )

    print("\n" + "=" * 80)
    print("BUILD COMPLETED SUCCESSFULLY")
    print("=" * 80)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()