from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

FILES = {
    "logon": PROJECT_ROOT / "data" / "processed" / "logon.parquet",
    "device": PROJECT_ROOT / "data" / "processed" / "device.parquet",
    "file": PROJECT_ROOT / "data" / "processed" / "file.parquet",
}


print("=" * 70)
print("CERT ACTIVITY CONTEXT INSPECTION")
print("=" * 70)


# ------------------------------------------------------------
# LOGON
# ------------------------------------------------------------

print("\n1. LOGON ACTIVITY VALUES")
print("-" * 50)

logon = pd.read_parquet(FILES["logon"])

print("Columns:")
print(logon.columns.tolist())

if "activity" in logon.columns:
    print("\nActivity values:")
    print(logon["activity"].value_counts(dropna=False).to_string())


# ------------------------------------------------------------
# DEVICE
# ------------------------------------------------------------

print("\n2. DEVICE ACTIVITY VALUES")
print("-" * 50)

device = pd.read_parquet(FILES["device"])

print("Columns:")
print(device.columns.tolist())

if "activity" in device.columns:
    print("\nActivity values:")
    print(device["activity"].value_counts(dropna=False).to_string())


# ------------------------------------------------------------
# FILE
# ------------------------------------------------------------

print("\n3. FILE DATA")
print("-" * 50)

file_df = pd.read_parquet(FILES["file"])

print("Columns:")
print(file_df.columns.tolist())

print("\nRows:")
print(f"{len(file_df):,}")

if "filename" in file_df.columns:
    print("\nSample filenames:")
    print(
        file_df["filename"]
        .dropna()
        .head(20)
        .to_string(index=False)
    )


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("INSPECTION COMPLETED")
print("=" * 70)