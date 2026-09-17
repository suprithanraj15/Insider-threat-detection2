from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_HTTP = PROJECT_ROOT / "CERT_Dataset" / "r4.1" / "http.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "http_daily_features.parquet"

CHUNK_SIZE = 100_000


def extract_domain(url):
    """Extract domain from URL safely."""
    try:
        return urlparse(str(url)).netloc.lower()
    except Exception:
        return ""


def process_http():

    print("=" * 60)
    print("HTTP DATA PROCESSING")
    print("=" * 60)

    daily_features = []

    chunk_number = 0

    for chunk in tqdm(
        pd.read_csv(
            RAW_HTTP,
            chunksize=CHUNK_SIZE,
            low_memory=False
        ),
        desc="Processing HTTP chunks"
    ):

        chunk_number += 1

        # Standardize column names
        chunk.columns = chunk.columns.str.strip().str.lower()

        # Convert date
        chunk["date"] = pd.to_datetime(
            chunk["date"],
            errors="coerce"
        )

        # Remove invalid dates
        chunk = chunk.dropna(subset=["date"])

        # Create daily key
        chunk["day"] = chunk["date"].dt.date

        # Clean URLs
        chunk["url"] = chunk["url"].fillna("").astype(str)

        # Extract domain
        chunk["domain"] = chunk["url"].apply(extract_domain)

        # Daily user-level aggregation
        features = (
            chunk.groupby(["user", "day"])
            .agg(
                web_visits=("id", "count"),
                unique_urls=("url", "nunique"),
                unique_domains=("domain", "nunique"),
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

    # Same user/day may appear in multiple chunks.
    result = (
        result.groupby(["user", "day"])
        .agg(
            web_visits=("web_visits", "sum"),
            unique_urls=("unique_urls", "sum"),
            unique_domains=("unique_domains", "sum"),
        )
        .reset_index()
    )

    result.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 60)
    print("HTTP PROCESSING COMPLETED")
    print("=" * 60)

    print(f"Rows: {len(result):,}")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    process_http()