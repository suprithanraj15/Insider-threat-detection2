from pathlib import Path
import csv
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ANSWERS_DIR = PROJECT_ROOT / "CERT_Dataset" / "answers"

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "network_ground_truth.parquet"
)


print("=" * 70)
print("CERT r4.1 - NETWORK GROUND TRUTH")
print("=" * 70)


# ---------------------------------------------------------
# READ ANSWER FILES
# ---------------------------------------------------------

records = []

answer_files = sorted(
    ANSWERS_DIR.glob("r4.1-*.csv")
)

print(f"\nAnswer files found: {len(answer_files)}")


for answer_file in answer_files:

    print(f"\nReading: {answer_file.name}")

    with open(
        answer_file,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.reader(f)

        file_http_count = 0

        for row_number, row in enumerate(reader, start=1):

            if not row:
                continue

            # HTTP answer record
            if row[0].strip().lower() != "http":
                continue

            # CERT HTTP format:
            # 0 = event type
            # 1 = event id
            # 2 = timestamp
            # 3 = user
            # 4 = PC
            # 5 = URL
            # 6 = content

            if len(row) < 7:
                print(
                    f"WARNING: malformed HTTP row "
                    f"{row_number} in {answer_file.name}"
                )
                continue

            timestamp = pd.to_datetime(
                row[2],
                errors="coerce"
            )

            if pd.isna(timestamp):
                print(
                    f"WARNING: invalid timestamp "
                    f"at row {row_number} in {answer_file.name}"
                )
                continue

            records.append(
                {
                    "scenario": answer_file.stem,
                    "user": row[3].strip().lower(),
                    "pc": row[4].strip().lower(),
                    "timestamp": timestamp,
                    "day": timestamp.normalize(),
                    "url": row[5].strip(),
                    "content": row[6].strip()
                }
            )

            file_http_count += 1

        print(
            f"HTTP records extracted: {file_http_count}"
        )


# ---------------------------------------------------------
# CREATE DATAFRAME
# ---------------------------------------------------------

df = pd.DataFrame(records)


print("\n" + "=" * 70)
print("HTTP GROUND TRUTH SUMMARY")
print("=" * 70)

print(
    f"Total malicious HTTP records: {len(df):,}"
)


# ---------------------------------------------------------
# SCENARIO DISTRIBUTION
# ---------------------------------------------------------

print("\nHTTP records by scenario:")

print(
    df["scenario"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ---------------------------------------------------------
# USER DISTRIBUTION
# ---------------------------------------------------------

print("\nHTTP records by user:")

print(
    df["user"]
    .value_counts()
    .to_string()
)


# ---------------------------------------------------------
# UNIQUE MALICIOUS USER-DAYS
# ---------------------------------------------------------

user_day_gt = (
    df[
        [
            "user",
            "day",
            "scenario"
        ]
    ]
    .drop_duplicates()
)

user_day_gt["network_ground_truth"] = 1


print(
    "\nUnique malicious network user-days: "
    f"{len(user_day_gt):,}"
)


print("\nMalicious network user-days by scenario:")

print(
    user_day_gt["scenario"]
    .value_counts()
    .sort_index()
    .to_string()
)


print("\nMalicious network user-days:")

print(
    user_day_gt
    .sort_values(["user", "day"])
    .to_string(index=False)
)


# ---------------------------------------------------------
# DATE VALIDATION
# ---------------------------------------------------------

print("\nTimestamp validation:")

print(
    f"Valid timestamps: {df['timestamp'].notna().sum():,}"
)

print(
    f"Invalid timestamps: {df['timestamp'].isna().sum():,}"
)


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

user_day_gt.to_parquet(
    OUTPUT_FILE,
    engine="fastparquet",
    index=False
)


print("\n" + "=" * 70)
print("NETWORK GROUND TRUTH CREATED")
print("=" * 70)

print("\nSaved to:")
print(OUTPUT_FILE)

print(
    f"\nRows saved: {len(user_day_gt):,}"
)

print("\n" + "=" * 70)
print("COMPLETE")
print("=" * 70)