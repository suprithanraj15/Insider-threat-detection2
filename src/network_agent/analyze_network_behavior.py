from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

BEHAVIORAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "final_behavioral_features.parquet"
)

TEMPORAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "temporal_behavioral_features.parquet"
)

GT_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "network_ground_truth.parquet"
)


# ============================================================
# LOAD FILES
# ============================================================

print("=" * 70)
print("CERT r4.1 - NETWORK BEHAVIORAL ANALYSIS")
print("=" * 70)

print("\nLoading behavioral features...")

behavioral = pd.read_parquet(
    BEHAVIORAL_FILE,
    engine="fastparquet"
)

print(f"Behavioral rows: {len(behavioral):,}")


print("\nLoading temporal features...")

temporal = pd.read_parquet(
    TEMPORAL_FILE,
    engine="fastparquet"
)

print(f"Temporal rows: {len(temporal):,}")


print("\nLoading network ground truth...")

gt = pd.read_parquet(
    GT_FILE,
    engine="fastparquet"
)

print(f"Malicious network days: {len(gt):,}")


# ============================================================
# NORMALIZE USER
# ============================================================

for df in [behavioral, temporal, gt]:

    df["user"] = (
        df["user"]
        .astype(str)
        .str.strip()
        .str.lower()
    )


# ============================================================
# NORMALIZE DATES
# ============================================================

behavioral["day"] = pd.to_datetime(
    behavioral["day"],
    errors="coerce"
).dt.strftime("%Y-%m-%d")


temporal["day"] = pd.to_datetime(
    temporal["day"],
    errors="coerce"
).dt.strftime("%Y-%m-%d")


gt["day"] = pd.to_datetime(
    gt["day"],
    errors="coerce"
).dt.strftime("%Y-%m-%d")


# ============================================================
# DISPLAY AVAILABLE NETWORK-RELATED COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("NETWORK-RELATED BEHAVIORAL COLUMNS")
print("=" * 70)

behavioral_network_columns = [
    col
    for col in behavioral.columns
    if any(
        keyword in col.lower()
        for keyword in [
            "web",
            "url",
            "domain",
            "behavior",
            "zscore"
        ]
    )
]

for col in behavioral_network_columns:
    print(f"  {col}")


print("\n" + "=" * 70)
print("NETWORK-RELATED TEMPORAL COLUMNS")
print("=" * 70)

temporal_network_columns = [
    col
    for col in temporal.columns
    if any(
        keyword in col.lower()
        for keyword in [
            "web",
            "url",
            "domain",
            "temporal",
            "change",
            "persistent"
        ]
    )
]

for col in temporal_network_columns:
    print(f"  {col}")


# ============================================================
# SELECT IMPORTANT FEATURES
# ============================================================

behavioral_features = [
    "web_visits",
    "unique_urls",
    "unique_domains",
    "web_visits_clean_zscore",
    "clean_behavior_deviation"
]

temporal_features = [
    "web_visits_prev7_mean",
    "web_visits_change_7d",
    "temporal_change_count",
    "temporal_change_score",
    "persistent_behavior_flag"
]


# ============================================================
# CHECK AVAILABLE FEATURES
# ============================================================

print("\n" + "=" * 70)
print("FEATURE AVAILABILITY")
print("=" * 70)

for col in behavioral_features:

    if col in behavioral.columns:
        print(f"[OK] Behavioral: {col}")
    else:
        print(f"[MISSING] Behavioral: {col}")


for col in temporal_features:

    if col in temporal.columns:
        print(f"[OK] Temporal: {col}")
    else:
        print(f"[MISSING] Temporal: {col}")


# ============================================================
# MERGE DATA
# ============================================================

print("\n" + "=" * 70)
print("MERGING NETWORK BEHAVIORAL FEATURES")
print("=" * 70)

selected_behavioral = [
    "user",
    "day"
] + [
    col
    for col in behavioral_features
    if col in behavioral.columns
]

selected_temporal = [
    "user",
    "day"
] + [
    col
    for col in temporal_features
    if col in temporal.columns
]


data = behavioral[selected_behavioral].merge(
    temporal[selected_temporal],
    on=["user", "day"],
    how="inner"
)


data = data.merge(
    gt[
        [
            "user",
            "day",
            "scenario",
            "network_ground_truth"
        ]
    ],
    on=["user", "day"],
    how="left"
)


data["network_ground_truth"] = (
    data["network_ground_truth"]
    .fillna(0)
    .astype(int)
)


malicious = data[
    data["network_ground_truth"] == 1
].copy()

normal = data[
    data["network_ground_truth"] == 0
].copy()


print(f"Total rows       : {len(data):,}")
print(f"Normal rows      : {len(normal):,}")
print(f"Malicious rows   : {len(malicious):,}")


# ============================================================
# MALICIOUS NETWORK DAYS
# ============================================================

print("\n" + "=" * 70)
print("MALICIOUS NETWORK BEHAVIOR")
print("=" * 70)

display_columns = [
    "user",
    "day",
    "scenario"
]

display_columns += [
    col
    for col in behavioral_features
    if col in data.columns
]

display_columns += [
    col
    for col in temporal_features
    if col in data.columns
]

print(
    malicious[
        display_columns
    ]
    .sort_values(["user", "day"])
    .to_string(index=False)
)


# ============================================================
# NORMAL VS MALICIOUS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("NORMAL VS MALICIOUS BEHAVIOR")
print("=" * 70)

analysis_features = [
    col
    for col in (
        behavioral_features
        + temporal_features
    )
    if col in data.columns
]

for feature in analysis_features:

    normal_values = pd.to_numeric(
        normal[feature],
        errors="coerce"
    ).dropna()

    malicious_values = pd.to_numeric(
        malicious[feature],
        errors="coerce"
    ).dropna()

    if len(normal_values) == 0:
        continue

    print(f"\n--- {feature} ---")

    print(
        f"Normal    : "
        f"mean={normal_values.mean():.4f}, "
        f"median={normal_values.median():.4f}, "
        f"max={normal_values.max():.4f}"
    )

    print(
        f"Malicious : "
        f"mean={malicious_values.mean():.4f}, "
        f"median={malicious_values.median():.4f}, "
        f"max={malicious_values.max():.4f}"
    )


# ============================================================
# NETWORK BEHAVIORAL ANALYSIS COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("NETWORK BEHAVIORAL ANALYSIS COMPLETE")
print("=" * 70)