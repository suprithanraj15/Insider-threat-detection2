from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"

GROUND_TRUTH_FILE = OUTPUT_DIR / "final_behavioral_features.parquet"


print("=" * 70)
print("RULE AGENT - EVENT RULE FEATURES")
print("=" * 70)


# ------------------------------------------------------------
# 1. Load ground truth
# ------------------------------------------------------------

gt = pd.read_parquet(GROUND_TRUTH_FILE)

gt["day"] = pd.to_datetime(gt["day"]).dt.normalize()

base = gt[
    ["user", "day", "ground_truth", "scenario"]
].copy()

print(f"\nTotal user-days: {len(base):,}")
print(f"Malicious user-days: {(base['ground_truth'] == 1).sum():,}")
print(f"Normal user-days: {(base['ground_truth'] == 0).sum():,}")


# ------------------------------------------------------------
# Helper function
# ------------------------------------------------------------

def load_event_file(filename):
    path = PROCESSED_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    df = pd.read_parquet(path)

    if "day" not in df.columns:
        if "date" in df.columns:
            df["day"] = pd.to_datetime(df["date"]).dt.normalize()
        else:
            raise ValueError(
                f"Neither 'day' nor 'date' column found in {filename}.\n"
                f"Columns: {df.columns.tolist()}"
            )
    else:
        df["day"] = pd.to_datetime(df["day"]).dt.normalize()

    return df


# ------------------------------------------------------------
# 2. LOGON EVENT FEATURES
# ------------------------------------------------------------

print("\nLoading logon data...")

logon = load_event_file("logon.parquet")

logon_features = (
    logon.groupby(["user", "day"])
    .agg(
        event_logon_count=("id", "count"),
        event_logon_pc_count=("pc", "nunique")
    )
    .reset_index()
)


# ------------------------------------------------------------
# 3. DEVICE EVENT FEATURES
# ------------------------------------------------------------

print("Loading device data...")

device = load_event_file("device.parquet")

device_features = (
    device.groupby(["user", "day"])
    .agg(
        event_device_count=("id", "count"),
        event_device_pc_count=("pc", "nunique")
    )
    .reset_index()
)


# ------------------------------------------------------------
# 4. FILE EVENT FEATURES
# ------------------------------------------------------------

print("Loading file data...")

file_df = load_event_file("file.parquet")

file_features = (
    file_df.groupby(["user", "day"])
    .agg(
        event_file_count=("id", "count"),
        event_unique_files=("filename", "nunique")
    )
    .reset_index()
)


# ------------------------------------------------------------
# 5. EMAIL EVENT FEATURES
# ------------------------------------------------------------

print("Loading email daily features")

email = load_event_file("email_daily_features.parquet")

email_features = email[
    [
        "user",
        "day",
        "emails_sent",
        "total_attachments",
        "recipient_count",
        "unique_recipients"
    ]
].copy()

email_features = email_features.rename(
    columns={
        "emails_sent": "event_email_count",
        "total_attachments": "event_email_attachments",
        "recipient_count": "event_email_recipients",
        "unique_recipients": "event_unique_recipients"
    }
)


# ------------------------------------------------------------
# 6. HTTP DAILY FEATURES
# ------------------------------------------------------------

print("Loading HTTP data...")

http = load_event_file("http_daily_features.parquet")

http_features = http[
    [
        "user",
        "day",
        "web_visits",
        "unique_urls",
        "unique_domains"
    ]
].copy()

http_features = http_features.rename(
    columns={
        "web_visits": "event_http_count",
        "unique_urls": "event_unique_urls",
        "unique_domains": "event_unique_domains"
    }
)


# ------------------------------------------------------------
# 7. Merge all event features
# ------------------------------------------------------------

print("\nMerging event features...")

features = base.copy()

feature_tables = [
    logon_features,
    device_features,
    file_features,
    email_features,
    http_features
]

for table in feature_tables:

    features = features.merge(
        table,
        on=["user", "day"],
        how="left"
    )


# ------------------------------------------------------------
# 8. Fill missing values
# ------------------------------------------------------------

feature_columns = [
    "event_logon_count",
    "event_logon_pc_count",

    "event_device_count",
    "event_device_pc_count",

    "event_file_count",
    "event_unique_files",

    "event_email_count",
    "event_email_attachments",
    "event_email_recipients",
    "event_unique_recipients",

    "event_http_count",
    "event_unique_urls",
    "event_unique_domains"
]

for column in feature_columns:
    features[column] = features[column].fillna(0)


# ------------------------------------------------------------
# 9. Event presence indicators
# ------------------------------------------------------------

features["has_logon_event"] = (
    features["event_logon_count"] > 0
)

features["has_device_event"] = (
    features["event_device_count"] > 0
)

features["has_file_event"] = (
    features["event_file_count"] > 0
)

features["has_email_event"] = (
    features["event_email_count"] > 0
)

features["has_http_event"] = (
    features["event_http_count"] > 0
)


# ------------------------------------------------------------
# 10. Save
# ------------------------------------------------------------

output_file = OUTPUT_DIR / "event_rule_features.parquet"

features.to_parquet(
    output_file,
    index=False
)


# ------------------------------------------------------------
# 11. Verification
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("EVENT FEATURE SUMMARY")
print("=" * 70)

print(f"Rows: {len(features):,}")
print(f"Columns: {len(features.columns)}")

malicious = features[
    features["ground_truth"] == 1
]

print("\nMalicious user-day event presence:")

presence_columns = [
    "has_logon_event",
    "has_device_event",
    "has_file_event",
    "has_email_event",
    "has_http_event"
]

for column in presence_columns:

    count = int(malicious[column].sum())

    print(
        f"{column:25s}: {count}/32"
    )


print("\n" + "=" * 70)
print("EVENT RULE FEATURES COMPLETED")
print("=" * 70)

print("\nSaved to:")
print(output_file)
