from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


LOGON_FILE = PROCESSED / "logon.parquet"
DEVICE_FILE = PROCESSED / "device.parquet"
FILE_FILE = PROCESSED / "file.parquet"
EMAIL_FILE = PROCESSED / "email_daily_features.parquet"
HTTP_FILE = PROCESSED / "http_daily_features.parquet"
PSYCHOMETRIC_FILE = PROCESSED / "psychometric.parquet"


def prepare_logon():

    df = pd.read_parquet(LOGON_FILE)

    df["day"] = df["date"].dt.date

    result = (
        df.groupby(["user", "day"])
        .agg(
            logon_events=("id", "count"),
            logon_pc_count=("pc", "nunique"),
            logon_logins=("activity", lambda x: (x == "Logon").sum()),
            logon_logoffs=("activity", lambda x: (x == "Logoff").sum()),
        )
        .reset_index()
    )

    return result


def prepare_device():

    df = pd.read_parquet(DEVICE_FILE)

    df["day"] = df["date"].dt.date

    result = (
        df.groupby(["user", "day"])
        .agg(
            device_events=("id", "count"),
            device_pc_count=("pc", "nunique"),
            device_connects=("activity", lambda x: (x == "Connect").sum()),
            device_disconnects=("activity", lambda x: (x == "Disconnect").sum()),
        )
        .reset_index()
    )

    return result


def prepare_file():

    df = pd.read_parquet(FILE_FILE)

    df["day"] = df["date"].dt.date

    result = (
        df.groupby(["user", "day"])
        .agg(
            file_events=("id", "count"),
            file_pc_count=("pc", "nunique"),
            unique_files=("filename", "nunique"),
        )
        .reset_index()
    )

    return result


def prepare_email():

    df = pd.read_parquet(EMAIL_FILE)

    return df


def prepare_http():

    df = pd.read_parquet(HTTP_FILE)

    return df


def prepare_psychometric():

    df = pd.read_parquet(PSYCHOMETRIC_FILE)

    df = df.rename(
        columns={
            "user_id": "user"
        }
    )

    return df[
        ["user", "o", "c", "e", "a", "n"]
    ]


def main():

    print("=" * 60)
    print("BUILDING USER-DAY FEATURE TABLE")
    print("=" * 60)

    print("\n1. Processing logon...")
    logon = prepare_logon()
    print(f"Logon rows: {len(logon):,}")

    print("\n2. Processing device...")
    device = prepare_device()
    print(f"Device rows: {len(device):,}")

    print("\n3. Processing file...")
    file = prepare_file()
    print(f"File rows: {len(file):,}")

    print("\n4. Loading email features...")
    email = prepare_email()
    print(f"Email rows: {len(email):,}")

    print("\n5. Loading HTTP features...")
    http = prepare_http()
    print(f"HTTP rows: {len(http):,}")

    print("\n6. Loading psychometric data...")
    psychometric = prepare_psychometric()
    print(f"Psychometric users: {len(psychometric):,}")

    print("\n7. Merging activity features...")

    merged = logon.merge(
        device,
        on=["user", "day"],
        how="outer"
    )

    merged = merged.merge(
        file,
        on=["user", "day"],
        how="outer"
    )

    merged = merged.merge(
        email,
        on=["user", "day"],
        how="outer"
    )

    merged = merged.merge(
        http,
        on=["user", "day"],
        how="outer"
    )

    print(f"Rows after activity merge: {len(merged):,}")

    print("\n8. Filling missing activity values...")

    numeric_columns = merged.select_dtypes(
        include="number"
    ).columns

    merged[numeric_columns] = merged[numeric_columns].fillna(0)

    print("\n9. Adding psychometric information...")

    merged = merged.merge(
        psychometric,
        on="user",
        how="left"
    )

    print("\n10. Sorting...")

    merged = merged.sort_values(
        ["user", "day"]
    ).reset_index(drop=True)

    output_file = OUTPUT_DIR / "user_day_features.parquet"

    merged.to_parquet(
        output_file,
        index=False
    )

    print("\n" + "=" * 60)
    print("USER-DAY FEATURE TABLE CREATED")
    print("=" * 60)

    print(f"Rows: {len(merged):,}")
    print(f"Columns: {len(merged.columns)}")
    print(f"Users: {merged['user'].nunique():,}")
    print(f"Saved: {output_file}")

    print("\nColumns:")
    print(list(merged.columns))


if __name__ == "__main__":
    main()