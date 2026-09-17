from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONTEXT_FILE = PROJECT_ROOT / "data" / "outputs" / "rule_context_features.parquet"
RULE_FILE = PROJECT_ROOT / "data" / "outputs" / "rule_agent_results.parquet"
GROUND_TRUTH_FILE = PROJECT_ROOT / "data" / "outputs" / "final_behavioral_features.parquet"

print("=" * 70)
print("RULE AGENT - CONTEXT STRATEGY EVALUATION")
print("=" * 70)

# Load existing outputs
context_df = pd.read_parquet(CONTEXT_FILE)
rule_df = pd.read_parquet(RULE_FILE)
gt_df = pd.read_parquet(GROUND_TRUTH_FILE)

# Ground truth only
gt_df = gt_df[["user", "day", "ground_truth", "scenario"]]

# Remove any ground-truth columns from rule results if present
rule_columns_to_keep = [
    col for col in rule_df.columns
    if col not in ["ground_truth", "scenario"]
]
rule_df = rule_df[rule_columns_to_keep]

# Merge
df = context_df.merge(
    rule_df,
    on=["user", "day"],
    how="inner"
)

df = df.merge(
    gt_df,
    on=["user", "day"],
    how="inner"
)

print(f"\nJoined rows: {len(df):,}")
print(f"Malicious user-days: {(df['ground_truth'] == 1).sum():,}")
print(f"Normal user-days: {(df['ground_truth'] == 0).sum():,}")

# ------------------------------------------------------------
# Existing Rule Agent trigger
# ------------------------------------------------------------

df["existing_rule"] = df["rule_triggered"] == 1

# ------------------------------------------------------------
# Context conditions
# ------------------------------------------------------------

df["device_plus_high_ratio"] = (
    (df["connect_events"] > 0) &
    (df["connect_disconnect_ratio"] >= 0.5)
)

df["device_plus_file"] = (
    (df["connect_events"] > 0) &
    (
        (df["unique_file_extensions"] > 0) |
        (df["executable_files"] > 0) |
        (df["archive_files"] > 0)
    )
)

df["device_plus_logon"] = (
    (df["connect_events"] > 0) &
    (
        (df["after_hours_logons"] > 0) |
        (df["weekend_logons"] > 0)
    )
)

# ------------------------------------------------------------
# Strategies
# ------------------------------------------------------------

strategies = {
    "Existing Rule":
        df["existing_rule"],

    "Context: Device + High Ratio":
        df["device_plus_high_ratio"],

    "Context: Device + File":
        df["device_plus_file"],

    "Existing Rule + Device + High Ratio":
        df["existing_rule"] & df["device_plus_high_ratio"],

    "Existing Rule + Device + File":
        df["existing_rule"] & df["device_plus_file"],

    "Existing Rule OR Device + High Ratio":
        df["existing_rule"] | df["device_plus_high_ratio"],

    "Existing Rule OR Device + File":
        df["existing_rule"] | df["device_plus_file"],

    "Device + High Ratio + File":
        df["device_plus_high_ratio"] & df["device_plus_file"],
}

# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STRATEGY EVALUATION")
print("=" * 70)

actual = df["ground_truth"] == 1

for name, prediction in strategies.items():

    tp = int((prediction & actual).sum())
    fp = int((prediction & ~actual).sum())
    fn = int((~prediction & actual).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0
    )

    print(f"\n{name}")
    print(f"  TP        : {tp}")
    print(f"  FP        : {fp}")
    print(f"  FN        : {fn}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1        : {f1:.4f}")

print("\n" + "=" * 70)
print("EVALUATION COMPLETED")
print("=" * 70)