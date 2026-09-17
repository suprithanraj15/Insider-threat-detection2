from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"

LOGIN_AGENT_FILE = OUTPUT_DIR / "login_agent_results.parquet"
LOGIN_GT_FILE = OUTPUT_DIR / "login_ground_truth.parquet"


print("=" * 80)
print("LOGIN AGENT SCORE DISTRIBUTION ANALYSIS")
print("=" * 80)


# ================================================================
# 1. LOAD
# ================================================================

agent = pd.read_parquet(LOGIN_AGENT_FILE)
gt = pd.read_parquet(LOGIN_GT_FILE)


def normalize(df):
    df = df.copy()

    df["user"] = (
        df["user"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df["day"] = (
        pd.to_datetime(df["day"])
        .dt.normalize()
    )

    return df


agent = normalize(agent)
gt = normalize(gt)


# ================================================================
# 2. MERGE
# ================================================================

data = agent.merge(
    gt[
        [
            "user",
            "day",
            "login_ground_truth"
        ]
    ],
    on=["user", "day"],
    how="inner"
)

data["login_ground_truth"] = (
    data["login_ground_truth"]
    .fillna(0)
    .astype(int)
)


print(f"\nAgent rows : {len(agent):,}")
print(f"GT rows    : {len(gt):,}")
print(f"Merged rows: {len(data):,}")


# ================================================================
# 3. SCORE COLUMN CHECK
# ================================================================

score_column = "login_agent_score"

if score_column not in data.columns:

    raise ValueError(
        f"\nERROR: '{score_column}' not found.\n"
        f"Available columns:\n{list(data.columns)}"
    )


# ================================================================
# 4. NORMAL / MALICIOUS
# ================================================================

normal = (
    data[
        data["login_ground_truth"] == 0
    ][score_column]
    .dropna()
)

malicious = (
    data[
        data["login_ground_truth"] == 1
    ][score_column]
    .dropna()
)


print("\n" + "=" * 80)
print("SCORE DISTRIBUTION")
print("=" * 80)

print(f"\nNormal days    : {len(normal):,}")
print(f"Malicious days : {len(malicious):,}")


def print_distribution(name, values):

    print(f"\n{name}")
    print("-" * 60)

    print(f"Min    : {values.min():.6f}")
    print(f"Median : {values.median():.6f}")
    print(f"Mean   : {values.mean():.6f}")
    print(f"Max    : {values.max():.6f}")

    print("\nPercentiles:")

    for p in [
        90,
        95,
        97,
        98,
        99,
        99.5,
        99.9,
        99.99
    ]:

        print(
            f"P{p:<5} : "
            f"{np.percentile(values, p):.6f}"
        )


print_distribution("NORMAL", normal)
print_distribution("MALICIOUS", malicious)


# ================================================================
# 5. MALICIOUS LOGIN-DAY DETAILS
# ================================================================

print("\n" + "=" * 80)
print("MALICIOUS LOGIN-DAY SCORES")
print("=" * 80)


malicious_rows = (
    data[
        data["login_ground_truth"] == 1
    ]
    [
        [
            "user",
            "day",
            "login_agent_score",
            "login_evidence_count",
            "login_agent_status",
        ]
    ]
    .sort_values(
        ["user", "day"]
    )
    .copy()
)


print(
    malicious_rows.to_string(
        index=False
    )
)


# ================================================================
# 6. RANK OF EACH MALICIOUS SCORE
# ================================================================

print("\n" + "=" * 80)
print("MALICIOUS SCORE RANK AMONG NORMAL DAYS")
print("=" * 80)


rank_results = []


for _, row in malicious_rows.iterrows():

    score = row["login_agent_score"]

    normal_days_above = int(
        (normal > score).sum()
    )

    normal_days_at_or_above = int(
        (normal >= score).sum()
    )

    normal_percentile = (
        (normal <= score).mean() * 100
    )


    rank_results.append(
        {
            "user": row["user"],
            "day": row["day"],
            "score": score,
            "normal_days_above": normal_days_above,
            "normal_days_at_or_above": normal_days_at_or_above,
            "normal_percentile": normal_percentile,
        }
    )


rank_df = pd.DataFrame(rank_results)


print(
    rank_df.to_string(
        index=False
    )
)


# ================================================================
# 7. THRESHOLD ANALYSIS
# ================================================================

print("\n" + "=" * 80)
print("TARGETED SCORE THRESHOLD ANALYSIS")
print("=" * 80)


def evaluate(threshold):

    predictions = (
        data[score_column] >= threshold
    )

    actual = (
        data["login_ground_truth"] == 1
    )

    tp = int(
        (predictions & actual).sum()
    )

    fp = int(
        (predictions & ~actual).sum()
    )

    tn = int(
        (~predictions & ~actual).sum()
    )

    fn = int(
        (~predictions & actual).sum()
    )


    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )


    return {
        "threshold": threshold,
        "alerts": int(predictions.sum()),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
    }


# Use the actual distinct malicious scores.
thresholds = sorted(
    malicious.unique()
)


threshold_results = []


for threshold in thresholds:

    threshold_results.append(
        evaluate(threshold)
    )


threshold_df = pd.DataFrame(
    threshold_results
)


print(
    threshold_df[
        [
            "threshold",
            "alerts",
            "TP",
            "FP",
            "FN",
            "precision",
            "recall",
            "f1",
            "fpr",
        ]
    ].to_string(index=False)
)


# ================================================================
# 8. BEST SINGLE THRESHOLD
# ================================================================

print("\n" + "=" * 80)
print("BEST SINGLE-THRESHOLD RESULTS")
print("=" * 80)


best_f1 = (
    threshold_df
    .sort_values(
        [
            "f1",
            "precision",
            "recall"
        ],
        ascending=False
    )
    .iloc[0]
)


best_precision = (
    threshold_df
    .sort_values(
        [
            "precision",
            "f1",
            "recall"
        ],
        ascending=False
    )
    .iloc[0]
)


print("\nBEST F1:")

print(
    best_f1.to_string()
)


print("\nBEST PRECISION:")

print(
    best_precision.to_string()
)


# ================================================================
# 9. LOW-FPR ANALYSIS
# ================================================================

print("\n" + "=" * 80)
print("BEST RESULTS UNDER LOW FPR")
print("=" * 80)


for max_fpr in [
    0.01,
    0.005,
    0.001,
    0.0005
]:

    subset = threshold_df[
        threshold_df["fpr"] <= max_fpr
    ]


    print(
        f"\nFPR <= {max_fpr * 100:.03f}%"
    )


    if subset.empty:

        print(
            "No threshold found."
        )

    else:

        best = (
            subset
            .sort_values(
                [
                    "f1",
                    "precision",
                    "recall"
                ],
                ascending=False
            )
            .iloc[0]
        )


        print(
            best[
                [
                    "threshold",
                    "alerts",
                    "TP",
                    "FP",
                    "FN",
                    "precision",
                    "recall",
                    "f1",
                    "fpr",
                ]
            ].to_string()
        )


# ================================================================
# 10. SCORE SEPARATION SUMMARY
# ================================================================

print("\n" + "=" * 80)
print("SCORE SEPARATION SUMMARY")
print("=" * 80)


normal_p99 = np.percentile(
    normal,
    99
)

normal_p999 = np.percentile(
    normal,
    99.9
)

malicious_min = malicious.min()
malicious_median = malicious.median()
malicious_max = malicious.max()


print(
    f"\nNormal P99       : {normal_p99:.6f}"
)

print(
    f"Normal P99.9     : {normal_p999:.6f}"
)

print(
    f"Malicious Min    : {malicious_min:.6f}"
)

print(
    f"Malicious Median : {malicious_median:.6f}"
)

print(
    f"Malicious Max    : {malicious_max:.6f}"
)


if malicious_min > normal_p99:

    print(
        "\nGOOD SEPARATION:"
        "\nThe lowest malicious score is above "
        "the normal P99."
    )

else:

    print(
        "\nOVERLAP DETECTED:"
        "\nNormal scores overlap with malicious scores."
    )


# ================================================================
# 11. SAVE
# ================================================================

output_file = (
    OUTPUT_DIR /
    "login_score_distribution_analysis.parquet"
)


threshold_df.to_parquet(
    output_file,
    index=False
)


rank_output = (
    OUTPUT_DIR /
    "login_score_rank_analysis.parquet"
)


rank_df.to_parquet(
    rank_output,
    index=False
)


print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print(
    f"Threshold results saved : {output_file}"
)

print(
    f"Rank results saved      : {rank_output}"
)