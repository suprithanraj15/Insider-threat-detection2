from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"

BEHAVIORAL_FILE = OUTPUT_DIR / "final_behavioral_features.parquet"
TEMPORAL_FILE = OUTPUT_DIR / "temporal_behavioral_features.parquet"
CONTEXT_FILE = OUTPUT_DIR / "rule_context_features.parquet"
LOGIN_GT_FILE = OUTPUT_DIR / "login_ground_truth.parquet"


def percentile_rank(normal_values, value):
    normal_values = pd.Series(normal_values).dropna()

    if len(normal_values) == 0:
        return np.nan

    return (normal_values <= value).mean() * 100


print("=" * 80)
print("PERSONAL LOGIN BASELINE DIAGNOSTIC")
print("=" * 80)

behavioral = pd.read_parquet(BEHAVIORAL_FILE)
temporal = pd.read_parquet(TEMPORAL_FILE)
context = pd.read_parquet(CONTEXT_FILE)
login_gt = pd.read_parquet(LOGIN_GT_FILE)

for df in [behavioral, temporal, context, login_gt]:
    df["user"] = df["user"].astype(str).str.lower().str.strip()
    df["day"] = pd.to_datetime(df["day"]).dt.normalize()

data = (
    behavioral
    .merge(
        temporal[
            [
                "user",
                "day",
                "logon_events_prev7_mean",
                "logon_events_change_7d",
                "temporal_change_count",
                "temporal_change_score",
            ]
        ],
        on=["user", "day"],
        how="left",
    )
    .merge(
        context[
            [
                "user",
                "day",
                "after_hours_logons",
                "unique_logon_pcs",
                "total_logon_events",
                "after_hours_logon_ratio",
            ]
        ],
        on=["user", "day"],
        how="left",
    )
    .merge(
        login_gt[["user", "day", "login_ground_truth"]],
        on=["user", "day"],
        how="left",
    )
)

malicious = data[data["login_ground_truth"] == 1].copy()

print(f"\nTotal rows       : {len(data):,}")
print(f"Login GT rows    : {len(malicious)}")

features = [
    "logon_events",
    "logon_pc_count",
    "after_hours_logons",
    "unique_logon_pcs",
    "total_logon_events",
    "after_hours_logon_ratio",
    "logon_events_prev7_mean",
    "logon_events_change_7d",
    "temporal_change_count",
    "temporal_change_score",
    "logon_events_clean_zscore",
]

print("\n" + "=" * 80)
print("MALICIOUS LOGIN DAYS VS PERSONAL NORMAL BASELINE")
print("=" * 80)

results = []

for _, row in malicious.iterrows():

    user = row["user"]
    day = row["day"]

    # Only normal login days for this SAME user.
    normal = data[
        (data["user"] == user)
        & (data["login_ground_truth"] == 0)
    ]

    print(f"\nUSER: {user}   DAY: {day.date()}")
    print("-" * 80)

    for feature in features:

        if feature not in data.columns:
            continue

        normal_values = normal[feature].dropna()

        if len(normal_values) == 0:
            continue

        value = row[feature]

        if pd.isna(value):
            continue

        mean = normal_values.mean()
        median = normal_values.median()
        q95 = normal_values.quantile(0.95)
        q99 = normal_values.quantile(0.99)

        percentile = percentile_rank(normal_values, value)

        results.append({
            "user": user,
            "day": day,
            "feature": feature,
            "malicious_value": value,
            "normal_mean": mean,
            "normal_median": median,
            "normal_q95": q95,
            "normal_q99": q99,
            "personal_percentile": percentile,
        })

        print(
            f"{feature:30s} "
            f"value={value:8.3f}  "
            f"median={median:8.3f}  "
            f"q95={q95:8.3f}  "
            f"q99={q99:8.3f}  "
            f"percentile={percentile:6.2f}%"
        )


result_df = pd.DataFrame(results)

output_file = OUTPUT_DIR / "login_personal_baseline_diagnostic.parquet"
result_df.to_parquet(output_file, index=False)

print("\n" + "=" * 80)
print("SAVED")
print("=" * 80)
print(output_file)