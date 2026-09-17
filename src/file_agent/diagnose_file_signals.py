from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"

AGENT_FILE = OUTPUT_DIR / "file_agent_results.parquet"
GT_FILE = OUTPUT_DIR / "user_day_features_labeled.parquet"


print("=" * 80)
print("FILE AGENT - SIGNAL DIAGNOSTIC")
print("=" * 80)


# ================================================================
# 1. LOAD DATA
# ================================================================

agent = pd.read_parquet(
    AGENT_FILE,
    engine="fastparquet"
)

gt = pd.read_parquet(
    GT_FILE,
    engine="fastparquet"
)


# ================================================================
# 2. NORMALIZE
# ================================================================

agent["user"] = (
    agent["user"]
    .astype(str)
    .str.lower()
    .str.strip()
)

gt["user"] = (
    gt["user"]
    .astype(str)
    .str.lower()
    .str.strip()
)

agent["day"] = pd.to_datetime(agent["day"]).dt.normalize()
gt["day"] = pd.to_datetime(gt["day"]).dt.normalize()


# ================================================================
# 3. MERGE GROUND TRUTH
# ================================================================

data = agent.merge(
    gt[
        [
            "user",
            "day",
            "ground_truth",
            "scenario"
        ]
    ],
    on=["user", "day"],
    how="left"
)

data["ground_truth"] = (
    data["ground_truth"]
    .fillna(0)
    .astype(int)
)


print(f"\nTotal rows      : {len(data):,}")
print(
    f"Malicious days  : "
    f"{int(data['ground_truth'].sum()):,}"
)
print(
    f"Normal days     : "
    f"{int((data['ground_truth'] == 0).sum()):,}"
)


# ================================================================
# 4. SIGNAL LIST
# ================================================================

signal_columns = [
    "file_high_volume_signal",
    "file_high_diversity_signal",
    "file_multiple_pc_signal",
    "file_executable_signal",
    "file_archive_signal",
    "file_extension_diversity_signal",
    "file_behavioral_anomaly_signal",
    "file_strong_zscore_signal",
    "file_temporal_change_signal",
    "file_strong_temporal_signal",
    "file_volume_anomaly_signal",
    "file_diversity_anomaly_signal",
    "file_executable_anomaly_signal",
    "file_archive_volume_signal",
    "file_behavior_change_signal",
]


# ================================================================
# 5. SIGNAL DIAGNOSTIC
# ================================================================

results = []

malicious = data[data["ground_truth"] == 1]
normal = data[data["ground_truth"] == 0]

total_malicious = len(malicious)
total_normal = len(normal)


for signal in signal_columns:

    malicious_count = int(
        malicious[signal].sum()
    )

    normal_count = int(
        normal[signal].sum()
    )

    malicious_rate = (
        malicious_count / total_malicious
        if total_malicious > 0
        else 0
    )

    normal_rate = (
        normal_count / total_normal
        if total_normal > 0
        else 0
    )

    total_signal = malicious_count + normal_count

    signal_precision = (
        malicious_count / total_signal
        if total_signal > 0
        else 0
    )

    # Lift tells us how much more often the signal occurs
    # in malicious days compared with normal days.
    lift = (
        malicious_rate / normal_rate
        if normal_rate > 0
        else np.inf
    )

    results.append(
        {
            "signal": signal,
            "malicious_count": malicious_count,
            "malicious_rate": malicious_rate,
            "normal_count": normal_count,
            "normal_rate": normal_rate,
            "signal_precision": signal_precision,
            "lift": lift,
        }
    )


diagnostic = pd.DataFrame(results)


diagnostic = diagnostic.sort_values(
    "lift",
    ascending=False
)


# ================================================================
# 6. PRINT DIAGNOSTIC TABLE
# ================================================================

print("\n" + "=" * 80)
print("SIGNAL DISCRIMINATION ANALYSIS")
print("=" * 80)

print(
    diagnostic.to_string(
        index=False,
        formatters={
            "malicious_rate": lambda x: f"{x:.4%}",
            "normal_rate": lambda x: f"{x:.4%}",
            "signal_precision": lambda x: f"{x:.4%}",
            "lift": lambda x: (
                "INF"
                if np.isinf(x)
                else f"{x:.2f}"
            ),
        }
    )
)


# ================================================================
# 7. RANK SIGNALS BY MALICIOUS COVERAGE
# ================================================================

print("\n" + "=" * 80)
print("TOP SIGNALS BY MALICIOUS COVERAGE")
print("=" * 80)

coverage = diagnostic.sort_values(
    "malicious_rate",
    ascending=False
)

print(
    coverage[
        [
            "signal",
            "malicious_count",
            "malicious_rate",
            "normal_rate",
            "lift"
        ]
    ].head(10).to_string(
        index=False,
        formatters={
            "malicious_rate": lambda x: f"{x:.4%}",
            "normal_rate": lambda x: f"{x:.4%}",
            "lift": lambda x: (
                "INF"
                if np.isinf(x)
                else f"{x:.2f}"
            ),
        }
    )
)


# ================================================================
# 8. TOP SIGNALS BY LOWEST FALSE-POSITIVE RATE
# ================================================================

print("\n" + "=" * 80)
print("TOP SIGNALS BY LOWEST NORMAL ACTIVATION")
print("=" * 80)

low_fp = diagnostic.sort_values(
    "normal_rate",
    ascending=True
)

print(
    low_fp[
        [
            "signal",
            "malicious_count",
            "malicious_rate",
            "normal_count",
            "normal_rate",
            "lift"
        ]
    ].head(10).to_string(
        index=False,
        formatters={
            "malicious_rate": lambda x: f"{x:.4%}",
            "normal_rate": lambda x: f"{x:.4%}",
            "lift": lambda x: (
                "INF"
                if np.isinf(x)
                else f"{x:.2f}"
            ),
        }
    )
)


# ================================================================
# 9. MALICIOUS-DAY SIGNAL MATRIX
# ================================================================

print("\n" + "=" * 80)
print("MALICIOUS USER-DAY SIGNAL MATRIX")
print("=" * 80)

malicious_matrix = malicious[
    [
        "user",
        "day",
        "scenario",
        "file_agent_score",
        "file_evidence_count",
    ] + signal_columns
].sort_values(
    ["scenario", "user", "day"]
)

print(
    malicious_matrix.to_string(
        index=False
    )
)


# ================================================================
# 10. SCENARIO COVERAGE
# ================================================================

print("\n" + "=" * 80)
print("SIGNAL COVERAGE BY SCENARIO")
print("=" * 80)


for scenario in sorted(
    malicious["scenario"].unique()
):

    scenario_data = malicious[
        malicious["scenario"] == scenario
    ]

    print(
        f"\nSCENARIO {int(scenario)} "
        f"({len(scenario_data)} malicious days)"
    )

    for signal in signal_columns:

        count = int(
            scenario_data[signal].sum()
        )

        rate = count / len(scenario_data)

        print(
            f"{signal:40s} "
            f"{count:2d}/{len(scenario_data):2d} "
            f"({rate:.1%})"
        )


# ================================================================
# 11. SAVE DIAGNOSTIC
# ================================================================

output_file = (
    OUTPUT_DIR /
    "file_signal_diagnostic.csv"
)

diagnostic.to_csv(
    output_file,
    index=False
)


print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)

print(
    f"Saved : {output_file}"
)