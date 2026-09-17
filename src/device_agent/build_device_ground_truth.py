from pathlib import Path
import csv
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

ANSWERS_DIR = PROJECT_ROOT / "CERT_Dataset" / "answers"

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "device_ground_truth.parquet"
)


EXPECTED_ROWS = {
    "r4.1-1.csv": 10,
    "r4.1-2.csv": 246,
    "r4.1-3.csv": 17,
}


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("CERT r4.1 - DEVICE AGENT GROUND TRUTH")
print("=" * 70)


# ============================================================
# READ ANSWER FILES
# ============================================================

all_records = []

for filename, expected_count in EXPECTED_ROWS.items():

    filepath = ANSWERS_DIR / filename

    print(f"\nReading: {filename}")

    if not filepath.exists():
        raise FileNotFoundError(
            f"Answer file not found: {filepath}"
        )

    rows = []

    with open(
        filepath,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.reader(f)

        for row in reader:
            if row:
                rows.append(row)

    print(f"Rows found    : {len(rows)}")
    print(f"Rows expected : {expected_count}")

    if len(rows) != expected_count:
        raise ValueError(
            f"Row count mismatch in {filename}: "
            f"expected {expected_count}, "
            f"found {len(rows)}"
        )

    scenario = int(
        filename.split("-")[1].split(".")[0]
    )

    for row in rows:

        # CERT r4.1 answer-key structure:
        #
        # row[0] = event type
        # row[1] = event ID
        # row[2] = timestamp
        # row[3] = user
        # row[4] = PC
        # row[5] = activity

        if len(row) < 6:
            continue

        event_type = row[0].strip().lower()

        if event_type != "device":
            continue

        event_id = row[1].strip()
        timestamp = row[2].strip()
        user = row[3].strip().lower()
        pc = row[4].strip()
        activity = row[5].strip().lower()

        all_records.append(
            {
                "scenario": scenario,
                "event_type": event_type,
                "event_id": event_id,
                "timestamp": timestamp,
                "user": user,
                "pc": pc,
                "activity": activity,
            }
        )


# ============================================================
# ANSWER KEY SUMMARY
# ============================================================

answer_df = pd.DataFrame(all_records)

print("\n" + "=" * 70)
print("ANSWER KEY SUMMARY")
print("=" * 70)

print(
    f"Total malicious DEVICE records : "
    f"{len(answer_df)}"
)

print("\nScenario distribution:")

print(
    answer_df["scenario"]
    .value_counts()
    .sort_index()
    .to_string()
)


print("\nDevice activity distribution:")

print(
    answer_df["activity"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# PARSE TIMESTAMP
# ============================================================

answer_df["timestamp_parsed"] = pd.to_datetime(
    answer_df["timestamp"],
    format="%m/%d/%Y %H:%M:%S",
    errors="coerce"
)


invalid = answer_df[
    answer_df["timestamp_parsed"].isna()
]


if len(invalid) > 0:

    print(
        f"\nWARNING: {len(invalid)} "
        f"DEVICE records have invalid timestamps."
    )

    print(
        invalid[
            [
                "scenario",
                "event_id",
                "timestamp",
                "user",
                "pc",
                "activity"
            ]
        ].to_string(index=False)
    )

else:

    print(
        "\nAll DEVICE timestamps "
        "parsed successfully."
    )


# ============================================================
# KEEP VALID RECORDS
# ============================================================

answer_df = answer_df[
    answer_df["timestamp_parsed"].notna()
].copy()


# ============================================================
# BUILD USER-DAY GROUND TRUTH
# ============================================================

answer_df["day"] = (
    answer_df["timestamp_parsed"]
    .dt.normalize()
)


# ============================================================
# UNIQUE MALICIOUS DEVICE USER-DAYS
# ============================================================

device_ground_truth = (
    answer_df[
        [
            "user",
            "day",
            "scenario"
        ]
    ]
    .drop_duplicates()
    .sort_values(
        ["user", "day"]
    )
    .reset_index(drop=True)
)


device_ground_truth[
    "device_ground_truth"
] = 1


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DEVICE-SPECIFIC GROUND TRUTH")
print("=" * 70)

print(
    f"Malicious DEVICE records       : "
    f"{len(answer_df)}"
)

print(
    f"Unique malicious DEVICE user-days: "
    f"{len(device_ground_truth)}"
)

print(
    f"Unique malicious DEVICE users   : "
    f"{device_ground_truth['user'].nunique()}"
)


print("\nScenario distribution:")

print(
    device_ground_truth[
        "scenario"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


print("\nUsers:")

print(
    device_ground_truth[
        "user"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


print("\nMalicious DEVICE user-days:")

print(
    device_ground_truth.to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

device_ground_truth.to_parquet(
    OUTPUT_FILE,
    engine="fastparquet",
    index=False
)


print("\nSaved to:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("DEVICE GROUND-TRUTH BUILD COMPLETE")
print("=" * 70)