from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

HTTP_FILE = PROJECT_ROOT / "data" / "processed" / "http_daily_features.parquet"
GT_FILE = PROJECT_ROOT / "data" / "outputs" / "network_ground_truth.parquet"

print("=" * 70)
print("NETWORK GROUND-TRUTH JOIN DIAGNOSTIC")
print("=" * 70)

print("\nLoading files...")

http = pd.read_parquet(HTTP_FILE, engine="fastparquet")
gt = pd.read_parquet(GT_FILE, engine="fastparquet")

print(f"HTTP rows: {len(http):,}")
print(f"GT rows  : {len(gt):,}")

print("\nHTTP dtypes:")
print(http[["user", "day"]].dtypes)

print("\nGT dtypes:")
print(gt[["user", "day"]].dtypes)

print("\n" + "=" * 70)
print("HTTP SAMPLE")
print("=" * 70)
print(http[["user", "day"]].head(10).to_string(index=False))

print("\n" + "=" * 70)
print("GROUND TRUTH SAMPLE")
print("=" * 70)
print(gt[["user", "day", "scenario"]].head(10).to_string(index=False))

print("\n" + "=" * 70)
print("RAW VALUES FROM GROUND TRUTH")
print("=" * 70)

for _, row in gt.head(10).iterrows():
    print(
        f"user={repr(row['user'])}, "
        f"day={repr(row['day'])}, "
        f"scenario={repr(row['scenario'])}"
    )

print("\n" + "=" * 70)
print("NORMALIZATION TEST")
print("=" * 70)

http_test = http.copy()
gt_test = gt.copy()

http_test["user"] = http_test["user"].astype(str).str.strip().str.lower()
gt_test["user"] = gt_test["user"].astype(str).str.strip().str.lower()

http_test["day"] = pd.to_datetime(
    http_test["day"], errors="coerce"
).dt.strftime("%Y-%m-%d")

gt_test["day"] = pd.to_datetime(
    gt_test["day"], errors="coerce"
).dt.strftime("%Y-%m-%d")

print("\nNormalized HTTP sample:")
print(http_test[["user", "day"]].head(5).to_string(index=False))

print("\nNormalized GT sample:")
print(gt_test[["user", "day"]].head(10).to_string(index=False))

print("\n" + "=" * 70)
print("JOIN TEST")
print("=" * 70)

matched = gt_test.merge(
    http_test[["user", "day"]],
    on=["user", "day"],
    how="inner"
)

print(f"Ground-truth rows      : {len(gt_test):,}")
print(f"Matched user-days      : {len(matched):,}")
print(f"Unmatched ground-truth : {len(gt_test) - len(matched):,}")

if len(matched) > 0:
    print("\nMATCHED DAYS:")
    print(matched.drop_duplicates().to_string(index=False))

print("\n" + "=" * 70)
print("JOIN DIAGNOSTIC COMPLETE")
print("=" * 70)