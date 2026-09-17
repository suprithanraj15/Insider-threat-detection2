from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

PERSISTENCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "network_persistence_features.parquet"
)

GT_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "network_ground_truth.parquet"
)


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("NETWORK PERSISTENCE FEATURE DIAGNOSTIC")
print("=" * 70)

print("\nLoading persistence features...")

data = pd.read_parquet(
    PERSISTENCE_FILE,
    engine="fastparquet"
)

print(f"Persistence rows: {len(data):,}")


print("\nLoading network ground truth...")

gt = pd.read_parquet(
    GT_FILE,
    engine="fastparquet"
)

print(f"Ground-truth rows: {len(gt):,}")


# ============================================================
# NORMALIZE KEYS
# ============================================================

for df in [data, gt]:

    df["user"] = (
        df["user"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["day"] = pd.to_datetime(
        df["day"],
        errors="coerce"
    ).dt.strftime("%Y-%m-%d")


# ============================================================
# MERGE
# ============================================================

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


print("\n" + "=" * 70)
print("GROUND-TRUTH JOIN")
print("=" * 70)

print(
    f"Total rows      : {len(data):,}"
)

print(
    f"Normal rows     : {len(normal):,}"
)

print(
    f"Malicious rows  : {len(malicious):,}"
)


# ============================================================
# SIGNAL LIST
# ============================================================

signal_columns = [
    "high_web_activity_flag",
    "high_url_diversity_flag",
    "high_domain_diversity_flag",
    "network_intensity_flag",
    "current_high_web_with_past_activity",
    "current_high_web_with_strong_persistence",
    "network_diversity_signal",
    "personal_web_increase_flag",
    "strong_personal_web_increase_flag",
    "personal_diversity_increase_flag",
    "network_persistent_behavior_flag"
]


# ============================================================
# SIGNAL DISCRIMINATION
# ============================================================

print("\n" + "=" * 70)
print("PERSISTENCE SIGNAL DISCRIMINATION")
print("=" * 70)

results = []

for signal in signal_columns:

    malicious_count = int(
        malicious[signal].sum()
    )

    normal_count = int(
        normal[signal].sum()
    )

    malicious_rate = (
        malicious_count / len(malicious)
        if len(malicious) > 0
        else 0
    )

    normal_rate = (
        normal_count / len(normal)
        if len(normal) > 0
        else 0
    )

    if normal_rate > 0:
        lift = (
            malicious_rate
            / normal_rate
        )
    else:
        lift = np.inf

    results.append({
        "Signal": signal,
        "Malicious_Count": malicious_count,
        "Malicious_Rate": malicious_rate,
        "Normal_Count": normal_count,
        "Normal_Rate": normal_rate,
        "Lift": lift
    })


diagnostic = pd.DataFrame(results)

diagnostic = diagnostic.sort_values(
    "Lift",
    ascending=False
)


print(
    diagnostic.to_string(
        index=False,
        formatters={
            "Malicious_Rate": "{:.6f}".format,
            "Normal_Rate": "{:.6f}".format,
            "Lift": "{:.4f}".format
        }
    )
)


# ============================================================
# PERSISTENCE VALUE DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("PERSISTENCE VALUE DISTRIBUTION")
print("=" * 70)

value_columns = [
    "high_web_days_prev7",
    "high_web_days_prev14",
    "network_persistence_ratio",
    "network_persistence_ratio_14d",
    "network_high_activity_streak",
    "web_activity_ratio_to_baseline",
    "url_diversity_ratio_to_baseline",
    "domain_diversity_ratio_to_baseline"
]


for feature in value_columns:

    if feature not in data.columns:
        continue

    normal_values = pd.to_numeric(
        normal[feature],
        errors="coerce"
    ).dropna()

    malicious_values = pd.to_numeric(
        malicious[feature],
        errors="coerce"
    ).dropna()

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
# MALICIOUS NETWORK DAYS
# ============================================================

print("\n" + "=" * 70)
print("MALICIOUS NETWORK DAYS - PERSISTENCE FEATURES")
print("=" * 70)

display_columns = [
    "user",
    "day",
    "scenario",
    "web_visits",
    "unique_urls",
    "unique_domains",
    "web_visits_prev7_mean",
    "web_activity_ratio_to_baseline",
    "high_web_days_prev7",
    "high_web_days_prev14",
    "network_persistence_ratio",
    "network_persistence_ratio_14d",
    "network_high_activity_streak",
    "network_persistent_behavior_flag",
    "personal_web_increase_flag",
    "strong_personal_web_increase_flag",
    "personal_diversity_increase_flag"
]


print(
    malicious[
        display_columns
    ]
    .sort_values(["user", "day"])
    .to_string(index=False)
)


# ============================================================
# MALICIOUS SIGNAL COVERAGE
# ============================================================

print("\n" + "=" * 70)
print("MALICIOUS-DAY SIGNAL COVERAGE")
print("=" * 70)

for signal in signal_columns:

    detected = int(
        malicious[signal].sum()
    )

    total = len(malicious)

    coverage = (
        detected / total
        if total > 0
        else 0
    )

    print(
        f"{signal}: "
        f"{detected}/{total} "
        f"({coverage * 100:.2f}%)"
    )


# ============================================================
# HFC0492 ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("HFC0492 PERSISTENCE ANALYSIS")
print("=" * 70)

hfc = malicious[
    malicious["user"] == "hfc0492"
].copy()

if len(hfc) > 0:

    hfc_columns = [
        "day",
        "web_visits",
        "web_visits_prev7_mean",
        "web_activity_ratio_to_baseline",
        "high_web_days_prev7",
        "high_web_days_prev14",
        "network_persistence_ratio",
        "network_high_activity_streak",
        "network_persistent_behavior_flag"
    ]

    print(
        hfc[
            hfc_columns
        ]
        .sort_values("day")
        .to_string(index=False)
    )


# ============================================================
# SAVE DIAGNOSTIC
# ============================================================

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "network_persistence_diagnostic.csv"
)

diagnostic.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nDiagnostic saved to:")
print(OUTPUT_FILE)


print("\n" + "=" * 70)
print("NETWORK PERSISTENCE DIAGNOSTIC COMPLETE")
print("=" * 70)