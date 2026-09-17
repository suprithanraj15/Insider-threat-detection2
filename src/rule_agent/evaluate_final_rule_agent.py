from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "final_rule_agent_results.parquet"
)

df = pd.read_parquet(INPUT_FILE)

actual = df["ground_truth"] == 1
predicted = df["rule_triggered"] == 1

tp = int((predicted & actual).sum())
fp = int((predicted & ~actual).sum())
fn = int((~predicted & actual).sum())
tn = int((~predicted & ~actual).sum())

precision = tp / (tp + fp) if (tp + fp) else 0
recall = tp / (tp + fn) if (tp + fn) else 0
f1 = (
    2 * precision * recall / (precision + recall)
    if (precision + recall)
    else 0
)

accuracy = (tp + tn) / len(df)

print("=" * 70)
print("FINAL RULE AGENT EVALUATION")
print("=" * 70)

print(f"\nTotal user-days : {len(df):,}")
print(f"Malicious       : {int(actual.sum()):,}")
print(f"Normal          : {int((~actual).sum()):,}")

print("\nConfusion Matrix:")
print(f"  TP : {tp}")
print(f"  FP : {fp}")
print(f"  FN : {fn}")
print(f"  TN : {tn}")

print("\nMetrics:")
print(f"  Accuracy  : {accuracy:.4f}")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1        : {f1:.4f}")

if "scenario" in df.columns:
    print("\nScenario Detection:")

    for scenario in sorted(
        df.loc[df["ground_truth"] == 1, "scenario"].dropna().unique()
    ):
        scenario_df = df[
            (df["ground_truth"] == 1) &
            (df["scenario"] == scenario)
        ]

        detected = int(
            scenario_df["rule_triggered"].sum()
        )

        total = len(scenario_df)

        print(
            f"  Scenario {int(scenario)}: "
            f"{detected}/{total}"
        )

print("\n" + "=" * 70)
print("EVALUATION COMPLETED")
print("=" * 70)