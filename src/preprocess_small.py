from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA = PROJECT_ROOT / "CERT_Dataset" / "r4.1"
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"

PROCESSED_DATA.mkdir(parents=True, exist_ok=True)


def process_file(filename):
    input_path = RAW_DATA / filename
    output_path = PROCESSED_DATA / filename.replace(".csv", ".parquet")

    print("\n" + "=" * 60)
    print(f"Processing: {filename}")
    print("=" * 60)

    df = pd.read_csv(input_path)

    # Clean column names
    df.columns = df.columns.str.strip().str.lower()

    # Convert date column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

    # Remove completely empty rows
    df = df.dropna(how="all")

    # Remove duplicate IDs
    if "id" in df.columns:
        before = len(df)

        df = df.drop_duplicates(
            subset=["id"],
            keep="first"
        )

        print(f"Duplicate rows removed: {before - len(df):,}")

    # Save as Parquet
    df.to_parquet(
        output_path,
        index=False
    )

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Saved: {output_path}")


files = [
    "logon.csv",
    "device.csv",
    "file.csv",
    "psychometric.csv",
]


if __name__ == "__main__":

    for filename in files:
        process_file(filename)

    print("\n" + "=" * 60)
    print("SMALL FILE PREPROCESSING COMPLETED")
    print("=" * 60)