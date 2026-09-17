from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

HTTP_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "http_daily_features.parquet"
)

GT_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "network_ground_truth.parquet"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_http_day(series):
    """
    HTTP daily features contain the day as an integer-like
    nanosecond timestamp.

    Example:
        1262563200000000000
        -> 2010-01-04
    """

    numeric_day = pd.to_numeric(series, errors="coerce")

    return pd.to_datetime(
        numeric_day,
        unit="ns",
        errors="coerce"
    ).dt.strftime("%Y-%m-%d")


def normalize_gt_day(series):
    """
    Ground-truth day is already stored as datetime.
    """

    return pd.to_datetime(
        series,
        errors="coerce"
    ).dt.strftime("%Y-%m-%d")


def percentile_rank(normal_values, value):
    """
    Calculate the percentage of normal observations
    less than or equal to the supplied value.
    """

    if len(normal_values) == 0:
        return float("nan")

    return (
        (normal_values <= value).sum()
        / len(normal_values)
    ) * 100


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("CERT r4.1 - NETWORK FEATURE ANALYSIS")
print("=" * 70)


# ============================================================
# LOAD HTTP FEATURES
# ============================================================

print("\nLoading HTTP daily features...")

http = pd.read_parquet(
    HTTP_FILE,
    engine="fastparquet"
)

print(f"HTTP rows: {len(http):,}")

print("\nHTTP columns:")
print(http.columns.tolist())


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("\nLoading network ground truth...")

gt = pd.read_parquet(
    GT_FILE,
    engine="fastparquet"
)

print(f"Malicious network days: {len(gt):,}")


# ============================================================
# NORMALIZE USER
# ============================================================

http["user"] = (
    http["user"]
    .astype(str)
    .str.strip()
    .str.lower()
)

gt["user"] = (
    gt["user"]
    .astype(str)
    .str.strip()
    .str.lower()
)


# ============================================================
# NORMALIZE DAY
# ============================================================

print("\nNormalizing dates...")

# IMPORTANT:
# HTTP day is stored as nanoseconds
http["day"] = normalize_http_day(http["day"])

# Ground-truth day is already datetime
gt["day"] = normalize_gt_day(gt["day"])


# ============================================================
# CHECK INVALID DATES
# ============================================================

http_invalid_days = http["day"].isna().sum()
gt_invalid_days = gt["day"].isna().sum()

print(f"Invalid HTTP dates: {http_invalid_days:,}")
print(f"Invalid GT dates  : {gt_invalid_days:,}")


# ============================================================
# SELECT NETWORK FEATURES
# ============================================================

network_features = [
    "web_visits",
    "unique_urls",
    "unique_domains"
]


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_http_columns = [
    "user",
    "day"
] + network_features

required_gt_columns = [
    "user",
    "day",
    "scenario"
]

missing_http = [
    col
    for col in required_http_columns
    if col not in http.columns
]

missing_gt = [
    col
    for col in required_gt_columns
    if col not in gt.columns
]

if missing_http:
    raise ValueError(
        f"Missing HTTP columns: {missing_http}"
    )

if missing_gt:
    raise ValueError(
        f"Missing ground-truth columns: {missing_gt}"
    )


# ============================================================
# MERGE HTTP FEATURES WITH GROUND TRUTH
# ============================================================

print("\n" + "=" * 70)
print("GROUND-TRUTH JOIN")
print("=" * 70)

merged = http.merge(
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

merged["network_ground_truth"] = (
    merged["network_ground_truth"]
    .fillna(0)
    .astype(int)
)

malicious = merged[
    merged["network_ground_truth"] == 1
].copy()

normal = merged[
    merged["network_ground_truth"] == 0
].copy()

print(f"HTTP rows             : {len(http):,}")
print(f"Ground-truth days     : {len(gt):,}")
print(f"Matched malicious days: {len(malicious):,}")
print(f"Normal rows            : {len(normal):,}")


# ============================================================
# VERIFY JOIN
# ============================================================

if len(malicious) == 0:
    raise RuntimeError(
        "\nERROR: No malicious network days matched.\n"
        "The user/day normalization is still incorrect."
    )

print("\nJOIN SUCCESSFUL.")


# ============================================================
# NETWORK FEATURE DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("NETWORK FEATURE DISTRIBUTION")
print("=" * 70)

for feature in network_features:

    print(f"\n--- {feature} ---")

    normal_values = pd.to_numeric(
        normal[feature],
        errors="coerce"
    ).dropna()

    malicious_values = pd.to_numeric(
        malicious[feature],
        errors="coerce"
    ).dropna()

    print(
        f"Normal     : "
        f"mean={normal_values.mean():.4f}, "
        f"median={normal_values.median():.4f}, "
        f"max={normal_values.max():.4f}"
    )

    print(
        f"Malicious  : "
        f"mean={malicious_values.mean():.4f}, "
        f"median={malicious_values.median():.4f}, "
        f"max={malicious_values.max():.4f}"
    )


# ============================================================
# MALICIOUS NETWORK DAYS
# ============================================================

print("\n" + "=" * 70)
print("MALICIOUS NETWORK DAYS")
print("=" * 70)

malicious_days = malicious[
    [
        "user",
        "day",
        "scenario",
        "web_visits",
        "unique_urls",
        "unique_domains"
    ]
].copy()

malicious_days = (
    malicious_days
    .sort_values(["user", "day"])
    .drop_duplicates(
        subset=["user", "day"]
    )
)

print(
    malicious_days.to_string(
        index=False
    )
)


# ============================================================
# MALICIOUS-DAY PERCENTILES
# ============================================================

print("\n" + "=" * 70)
print("MALICIOUS-DAY PERCENTILES IN GLOBAL POPULATION")
print("=" * 70)

for feature in network_features:

    print(f"\n{feature}")

    normal_values = pd.to_numeric(
        normal[feature],
        errors="coerce"
    ).dropna()

    for _, row in malicious_days.iterrows():

        value = pd.to_numeric(
            row[feature],
            errors="coerce"
        )

        if pd.isna(value):
            percentile = float("nan")
        else:
            percentile = percentile_rank(
                normal_values,
                value
            )

        print(
            f"{row['user']} "
            f"{row['day']} : "
            f"value={value:.4f}, "
            f"percentile={percentile:.2f}"
        )


# ============================================================
# ZERO / NON-ZERO NETWORK ACTIVITY
# ============================================================

print("\n" + "=" * 70)
print("ZERO / NON-ZERO NETWORK ACTIVITY")
print("=" * 70)

for feature in network_features:

    normal_values = pd.to_numeric(
        normal[feature],
        errors="coerce"
    )

    malicious_values = pd.to_numeric(
        malicious[feature],
        errors="coerce"
    )

    normal_zero = (
        normal_values == 0
    ).mean()

    malicious_zero = (
        malicious_values == 0
    ).mean()

    print(
        f"{feature}: "
        f"normal_zero={normal_zero:.4f}, "
        f"malicious_zero={malicious_zero:.4f}"
    )


# ============================================================
# ADDITIONAL NETWORK STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("NETWORK FEATURE DIFFERENCE")
print("=" * 70)

for feature in network_features:

    normal_values = pd.to_numeric(
        normal[feature],
        errors="coerce"
    ).dropna()

    malicious_values = pd.to_numeric(
        malicious[feature],
        errors="coerce"
    ).dropna()

    normal_mean = normal_values.mean()
    malicious_mean = malicious_values.mean()

    if normal_mean != 0:
        ratio = (
            malicious_mean
            / normal_mean
        )
    else:
        ratio = float("inf")

    print(
        f"{feature}: "
        f"malicious/normal mean ratio="
        f"{ratio:.4f}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("NETWORK ANALYSIS SUMMARY")
print("=" * 70)

print(
    f"Total HTTP user-days       : {len(http):,}"
)

print(
    f"Ground-truth network days  : {len(gt):,}"
)

print(
    f"Matched malicious days     : "
    f"{len(malicious_days):,}"
)

print(
    f"Unique malicious users     : "
    f"{malicious_days['user'].nunique():,}"
)

print(
    f"Scenarios represented      : "
    f"{malicious_days['scenario'].nunique():,}"
)

print("\nNetwork features analyzed:")
for feature in network_features:
    print(f"  - {feature}")

print("\n" + "=" * 70)
print("NETWORK FEATURE ANALYSIS COMPLETE")
print("=" * 70)