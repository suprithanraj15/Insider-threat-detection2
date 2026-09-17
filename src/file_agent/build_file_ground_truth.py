from pathlib import Path
import csv
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

ANSWERS_DIR = PROJECT_ROOT / "CERT_Dataset" / "answers"
OUTPUT_FILE = PROJECT_ROOT / "data" / "outputs" / "file_ground_truth.parquet"


# Expected answer-key row counts
EXPECTED_ROWS = {
    "r4.1-1.csv": 10,
    "r4.1-2.csv": 246,
    "r4.1-3.csv": 17,
}


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("CERT r4.1 - FILE AGENT GROUND TRUTH")
print("=" * 70)


# ============================================================
# READ ANSWER KEY
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
            f"expected {expected_count}, found {len(rows)}"
        )

    scenario = int(filename.split("-")[1].split(".")[0])

    for row in rows:

        # ----------------------------------------------------
        # CERT ANSWER-KEY STRUCTURE
        #
        # row[0] = event type
        # row[1] = event ID
        # row[2] = timestamp
        # row[3] = user
        # row[4] = PC
        # row[5+] = event-specific fields
        # ----------------------------------------------------

        if len(row) < 4:
            continue

        event_type = row[0].strip().lower()
        event_id = row[1].strip()
        timestamp = row[2].strip()
        user = row[3].strip().lower()

        all_records.append(
            {
                "scenario": scenario,
                "event_type": event_type,
                "event_id": event_id,
                "timestamp": timestamp,
                "user": user,
            }
        )


# ============================================================
# ANSWER KEY SUMMARY
# ============================================================

answer_df = pd.DataFrame(all_records)

print("\n" + "=" * 70)
print("ANSWER KEY SUMMARY")
print("=" * 70)

print(f"Total answer-key records : {len(answer_df)}")

file_records = answer_df[
    answer_df["event_type"] == "file"
].copy()

print(f"Malicious FILE records   : {len(file_records)}")


# ============================================================
# PARSE FILE TIMESTAMPS
# ============================================================

file_records["timestamp_parsed"] = pd.to_datetime(
    file_records["timestamp"],
    format="%m/%d/%Y %H:%M:%S",
    errors="coerce"
)

invalid_timestamps = file_records[
    file_records["timestamp_parsed"].isna()
]

if len(invalid_timestamps) > 0:

    print(
        f"\nWARNING: {len(invalid_timestamps)} "
        f"FILE records have invalid timestamps."
    )

    print("\nInvalid FILE records:")

    print(
        invalid_timestamps[
            ["scenario", "event_id", "timestamp", "user"]
        ].to_string(index=False)
    )

else:

    print("\nAll FILE timestamps parsed successfully.")


# ============================================================
# KEEP ONLY VALID FILE RECORDS
# ============================================================

file_records = file_records[
    file_records["timestamp_parsed"].notna()
].copy()


# ============================================================
# BUILD USER-DAY GROUND TRUTH
# ============================================================

file_records["day"] = (
    file_records["timestamp_parsed"]
    .dt.normalize()
)

file_records["user"] = (
    file_records["user"]
    .str.strip()
    .str.lower()
)


# ============================================================
# REMOVE DUPLICATE USER-DAY LABELS
# ============================================================

file_ground_truth = (
    file_records[
        ["user", "day", "scenario"]
    ]
    .drop_duplicates()
    .sort_values(["user", "day"])
    .reset_index(drop=True)
)


# Add binary label
file_ground_truth["file_ground_truth"] = 1


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FILE-SPECIFIC GROUND TRUTH")
print("=" * 70)

print(
    f"Malicious FILE records        : "
    f"{len(file_records)}"
)

print(
    f"Unique malicious FILE user-days: "
    f"{len(file_ground_truth)}"
)

print(
    f"Unique malicious FILE users    : "
    f"{file_ground_truth['user'].nunique()}"
)


print("\nScenario distribution:")

if len(file_ground_truth) > 0:

    print(
        file_ground_truth["scenario"]
        .value_counts()
        .sort_index()
        .to_string()
    )

else:

    print("No FILE ground-truth records found.")


print("\nUsers:")

if len(file_ground_truth) > 0:

    print(
        file_ground_truth["user"]
        .value_counts()
        .sort_index()
        .to_string()
    )

else:

    print("No FILE ground-truth users found.")


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

file_ground_truth.to_parquet(
    OUTPUT_FILE,
    engine="fastparquet",
    index=False
)


print("\nSaved to:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("GROUND-TRUTH BUILD COMPLETE")
print("=" * 70)