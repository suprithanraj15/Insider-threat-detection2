from pathlib import Path
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_EMAIL = PROJECT_ROOT / "CERT_Dataset" / "r4.1" / "email.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "email_daily_features.parquet"

CHUNK_SIZE = 100_000


def process_email():

    print("=" * 60)
    print("EMAIL DATA PROCESSING")
    print("=" * 60)

    daily_features = []

    chunk_number = 0

    for chunk in tqdm(
        pd.read_csv(
            RAW_EMAIL,
            chunksize=CHUNK_SIZE,
            low_memory=False
        ),
        desc="Processing email chunks"
    ):

        chunk_number += 1

        # Clean column names
        chunk.columns = chunk.columns.str.strip().str.lower()

        # Convert date
        chunk["date"] = pd.to_datetime(
            chunk["date"],
            errors="coerce"
        )

        # Remove invalid dates
        chunk = chunk.dropna(subset=["date"])

        # Create day
        chunk["day"] = chunk["date"].dt.date

        # Attachment count
        chunk["attachments"] = pd.to_numeric(
            chunk["attachments"],
            errors="coerce"
        ).fillna(0)

        # Email size
        chunk["size"] = pd.to_numeric(
            chunk["size"],
            errors="coerce"
        ).fillna(0)

        # Number of recipients
        recipient_columns = ["to", "cc", "bcc"]

        for column in recipient_columns:
            chunk[column] = chunk[column].fillna("")

        chunk["recipient_count"] = (
            (chunk["to"] != "").astype(int)
            + (chunk["cc"] != "").astype(int)
            + (chunk["bcc"] != "").astype(int)
        )

        # Daily aggregation by user
        features = (
            chunk.groupby(["user", "day"])
            .agg(
                emails_sent=("id", "count"),
                total_email_size=("size", "sum"),
                total_attachments=("attachments", "sum"),
                recipient_count=("recipient_count", "sum"),
                unique_recipients=("to", "nunique"),
            )
            .reset_index()
        )

        daily_features.append(features)

        print(
            f"Chunk {chunk_number} processed "
            f"({len(chunk):,} rows)"
        )

    print("\nCombining chunks...")

    result = pd.concat(
        daily_features,
        ignore_index=True
    )

    # Aggregate again because the same user/day
    # can occur in multiple chunks.
    result = (
        result.groupby(["user", "day"])
        .agg(
            emails_sent=("emails_sent", "sum"),
            total_email_size=("total_email_size", "sum"),
            total_attachments=("total_attachments", "sum"),
            recipient_count=("recipient_count", "sum"),
            unique_recipients=("unique_recipients", "sum"),
        )
        .reset_index()
    )

    result.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 60)
    print("EMAIL PROCESSING COMPLETED")
    print("=" * 60)

    print(f"Rows: {len(result):,}")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    process_email()