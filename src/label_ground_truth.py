from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_FILE = PROJECT_ROOT / "data" / "outputs" / "user_day_features.parquet"
ANSWERS_DIR = PROJECT_ROOT / "CERT_Dataset" / "answers"
OUTPUT_FILE = PROJECT_ROOT / "data" / "outputs" / "user_day_features_labeled.parquet"


# ============================================================
# CERT r4.1 GROUND-TRUTH INCIDENTS
# ============================================================

INCIDENT_FILES = {
    "r4.1-1.csv": {
        "user": "ABB0427",
        "scenario": 1,
    },
    "r4.1-2.csv": {
        "user": "HFC0492",
        "scenario": 2,
    },
    "r4.1-3.csv": {
        "user": "KTW0365",
        "scenario": 3,
    },
}


# ============================================================
# LOAD USER-DAY FEATURES
# ============================================================

print("=" * 60)
print("BUILDING CERT r4.1 GROUND-TRUTH LABELS")
print("=" * 60)

print("\n1. Loading user-day feature table...")

df = pd.read_parquet(FEATURE_FILE)

df["day"] = pd.to_datetime(df["day"]).dt.date
df["ground_truth"] = 0
df["scenario"] = 0

print(f"Rows: {len(df):,}")
print(f"Users: {df['user'].nunique():,}")


# ============================================================
# PROCESS EACH r4.1 ANSWER FILE
# ============================================================

print("\n2. Reading r4.1 answer observables...")

all_observables = []

for filename, info in INCIDENT_FILES.items():

    filepath = ANSWERS_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Missing answer file: {filepath}")

    print(f"   Reading {filename}...")

    records = []

    # Answer files are intentionally variable-width.
    # We only need the first 4 fields:
    # type, id, date, user
    with open(filepath, "r", encoding="utf-8") as f:

        for line in f:

            line = line.rstrip("\n\r")

            if not line.strip():
                continue

            parts = line.split(",", 4)

            if len(parts) < 4:
                continue

            activity_type = parts[0]
            event_id = parts[1]
            event_date = parts[2]
            user = parts[3]

            records.append(
                {
                    "activity_type": activity_type,
                    "event_id": event_id,
                    "event_date": pd.to_datetime(event_date),
                    "user": user,
                    "scenario": info["scenario"],
                }
            )

    incident_df = pd.DataFrame(records)

    print(f"      Observables: {len(incident_df):,}")

    all_observables.append(incident_df)


# ============================================================
# COMBINE OBSERVABLES
# ============================================================

observables = pd.concat(all_observables, ignore_index=True)

observables["day"] = observables["event_date"].dt.date

print("\n3. Ground-truth observable summary...")

print(
    observables.groupby(["user", "scenario"])
    .size()
    .reset_index(name="observable_events")
    .to_string(index=False)
)


# ============================================================
# MAP MALICIOUS OBSERVABLES TO USER-DAY
# ============================================================

print("\n4. Mapping malicious observables to user-day records...")

malicious_days = (
    observables[["user", "day", "scenario"]]
    .drop_duplicates()
)

# A user-day becomes ground truth positive
# only when an actual CMU malicious observable
# exists for that user on that day.

df = df.merge(
    malicious_days,
    on=["user", "day"],
    how="left",
    suffixes=("", "_answer"),
)

df["ground_truth"] = df["scenario_answer"].notna().astype(int)

df["scenario"] = (
    df["scenario_answer"]
    .fillna(0)
    .astype(int)
)

df.drop(columns=["scenario_answer"], inplace=True)


# ============================================================
# SUMMARY
# ============================================================

print("\n5. Ground-truth labeling summary...")

print(f"Total user-days: {len(df):,}")
print(f"Malicious user-days: {df['ground_truth'].sum():,}")
print(
    f"Normal user-days: {(df['ground_truth'] == 0).sum():,}"
)

print("\nMalicious user-days by scenario:")

print(
    df[df["ground_truth"] == 1]
    .groupby(["scenario"])
    .size()
    .rename("user_days")
    .to_string()
)


# ============================================================
# SAVE
# ============================================================

df = df.sort_values(["user", "day"])

df.to_parquet(
    OUTPUT_FILE,
    index=False,
)

print("\n" + "=" * 60)
print("GROUND-TRUTH LABELING COMPLETE")
print("=" * 60)

print(f"Saved: {OUTPUT_FILE}")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print("\nNew columns:")
print("  ground_truth")
print("  scenario")