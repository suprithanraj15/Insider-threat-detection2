from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "final_rule_agent_results.parquet"
)


print("=" * 60)
print("RULE AGENT EVALUATION")
print("=" * 60)

# ------------------------------------------------------------
# 1. Load results
# ------------------------------------------------------------

print("\n1. Loading Rule Agent results...")

df = pd.read_parquet(INPUT_FILE)

print(f"Rows: {len(df):,}")
print(f"Users: {df['user'].nunique():,}")


# ------------------------------------------------------------
# 2. Confusion matrix
# ------------------------------------------------------------

actual = df["ground_truth"]
predicted = df["rule_triggered"]

tp = ((actual == 1) & (predicted == 1)).sum()
tn = ((actual == 0) & (predicted == 0)).sum()
fp = ((actual == 0) & (predicted == 1)).sum()
fn = ((actual == 1) & (predicted == 0)).sum()


print("\n2. Confusion Matrix")
print("-" * 40)

print(f"True Positives  (TP): {tp:,}")
print(f"True Negatives  (TN): {tn:,}")
print(f"False Positives (FP): {fp:,}")
print(f"False Negatives (FN): {fn:,}")


# ------------------------------------------------------------
# 3. Metrics
# ------------------------------------------------------------

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0

f1 = (
    2 * precision * recall / (precision + recall)
    if (precision + recall) > 0
    else 0
)

accuracy = (
    (tp + tn) / len(df)
    if len(df) > 0
    else 0
)


print("\n3. Rule Agent Metrics")
print("-" * 40)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# ------------------------------------------------------------
# 4. Malicious user-day detection
# ------------------------------------------------------------

malicious = df[df["ground_truth"] == 1]

print("\n4. Ground-Truth Malicious Days")
print("-" * 40)

print(f"Total malicious user-days: {len(malicious):,}")
print(f"Detected by Rule Agent: {malicious['rule_triggered'].sum():,}")
print(
    f"Missed by Rule Agent: "
    f"{(malicious['rule_triggered'] == 0).sum():,}"
)


# ------------------------------------------------------------
# 5. Scenario-wise detection
# ------------------------------------------------------------

print("\n5. Scenario-wise Detection")
print("-" * 40)

scenario_results = (
    malicious
    .groupby("scenario")
    .agg(
        malicious_days=("ground_truth", "count"),
        detected_days=("rule_triggered", "sum")
    )
)

scenario_results["detection_rate"] = (
    scenario_results["detected_days"]
    / scenario_results["malicious_days"]
)

print(scenario_results.to_string())


# ------------------------------------------------------------
# 6. Save evaluation
# ------------------------------------------------------------

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "rule_agent_evaluation.csv"
)

scenario_results.to_csv(OUTPUT_FILE)

print("\nSaved:")
print(OUTPUT_FILE)

print("\n" + "=" * 60)
print("EVALUATION COMPLETED")
print("=" * 60)