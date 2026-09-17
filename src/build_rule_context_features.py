from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOGON_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "logon.parquet"
)

DEVICE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "device.parquet"
)

FILE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "file.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "rule_context_features.parquet"
)


print("=" * 70)
print("BUILDING RULE CONTEXT FEATURES")
print("=" * 70)


# ============================================================
# 1. LOGON CONTEXT
# ============================================================

print("\n1. Processing logon context...")

logon = pd.read_parquet(LOGON_FILE)

logon["date"] = pd.to_datetime(logon["date"])

logon["day"] = logon["date"].dt.date

logon["hour"] = logon["date"].dt.hour

logon["is_weekend"] = (
    logon["date"].dt.dayofweek >= 5
).astype(int)


# After-hours = before 7 AM or after/equal 7 PM
logon["after_hours"] = (
    (logon["hour"] < 7)
    | (logon["hour"] >= 19)
).astype(int)


logon_context = (
    logon
    .groupby(["user", "day"])
    .agg(
        after_hours_logons=("after_hours", "sum"),
        weekend_logons=("is_weekend", "sum"),
        unique_logon_pcs=("pc", "nunique"),
        total_logon_events=("id", "count")
    )
    .reset_index()
)


# ============================================================
# 2. DEVICE CONTEXT
# ============================================================

print("2. Processing device context...")

device = pd.read_parquet(DEVICE_FILE)

device["date"] = pd.to_datetime(device["date"])

device["day"] = device["date"].dt.date

device["connect_events"] = (
    device["activity"]
    .eq("Connect")
    .astype(int)
)

device["disconnect_events"] = (
    device["activity"]
    .eq("Disconnect")
    .astype(int)
)


device_context = (
    device
    .groupby(["user", "day"])
    .agg(
        connect_events=("connect_events", "sum"),
        disconnect_events=("disconnect_events", "sum"),
        unique_device_pcs=("pc", "nunique")
    )
    .reset_index()
)


# ============================================================
# 3. FILE CONTEXT
# ============================================================

print("3. Processing file context...")

file_df = pd.read_parquet(FILE_FILE)

file_df["date"] = pd.to_datetime(file_df["date"])

file_df["day"] = file_df["date"].dt.date

file_df["filename"] = (
    file_df["filename"]
    .fillna("")
    .astype(str)
)


file_df["extension"] = (
    file_df["filename"]
    .str.lower()
    .str.extract(r"(\.[a-z0-9]+)$", expand=False)
    .fillna("unknown")
)


file_df["is_executable"] = (
    file_df["extension"] == ".exe"
).astype(int)


file_df["is_archive"] = (
    file_df["extension"].isin(
        [".zip", ".rar", ".7z"]
    )
).astype(int)


file_context = (
    file_df
    .groupby(["user", "day"])
    .agg(
        unique_file_extensions=("extension", "nunique"),
        executable_files=("is_executable", "sum"),
        archive_files=("is_archive", "sum")
    )
    .reset_index()
)


# ============================================================
# 4. Merge context features
# ============================================================

print("4. Combining context features...")

logon_context["day"] = pd.to_datetime(
    logon_context["day"]
)

device_context["day"] = pd.to_datetime(
    device_context["day"]
)

file_context["day"] = pd.to_datetime(
    file_context["day"]
)


context = logon_context.merge(
    device_context,
    on=["user", "day"],
    how="outer"
)


context = context.merge(
    file_context,
    on=["user", "day"],
    how="outer"
)


# ============================================================
# 5. Clean missing values
# ============================================================

numeric_columns = context.select_dtypes(
    include=[np.number]
).columns

context[numeric_columns] = (
    context[numeric_columns]
    .fillna(0)
)


# ============================================================
# 6. Derived context features
# ============================================================

context["connect_disconnect_ratio"] = (
    context["connect_events"]
    / (context["disconnect_events"] + 1)
)


context["after_hours_logon_ratio"] = (
    context["after_hours_logons"]
    / (context["total_logon_events"] + 1)
)


# ============================================================
# 7. Save
# ============================================================

context = context.sort_values(
    ["user", "day"]
).reset_index(drop=True)


context.to_parquet(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 70)
print("RULE CONTEXT FEATURES CREATED")
print("=" * 70)

print(f"Rows: {len(context):,}")
print(f"Users: {context['user'].nunique():,}")
print(f"Columns: {len(context.columns):,}")

print("\nFeatures:")

for column in context.columns:
    if column not in ["user", "day"]:
        print(f"  - {column}")

print("\nSaved:")
print(OUTPUT_FILE)
