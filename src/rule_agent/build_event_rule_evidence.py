from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ANSWERS_DIR = PROJECT_ROOT / "CERT_Dataset" / "answers"
GROUND_TRUTH_FILE = PROJECT_ROOT / "data" / "outputs" / "final_behavioral_features.parquet"

print("=" * 70)
print("RULE AGENT - EVENT RULE EVIDENCE")
print("=" * 70)

# ------------------------------------------------------------
# Load ground truth
# ------------------------------------------------------------

gt = pd.read_parquet(GROUND_TRUTH_FILE)

malicious = gt[gt["ground_truth"] == 1][
    ["user", "day", "scenario"]
].copy()

print(f"\nMalicious user-days: {len(malicious)}")

# ------------------------------------------------------------
# Read CERT answer files
# ------------------------------------------------------------

records = []

for answer_file in sorted(ANSWERS_DIR.glob("r4.1-*.csv")):

    scenario_text = answer_file.stem.replace("r4.1-", "")

    with open(answer_file, "r", encoding="utf-8", errors="ignore") as f:

        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split(",")

            event_type = parts[0].strip().lower()

            if event_type not in {
                "logon",
                "device",
                "file",
                "email",
                "http"
            }:
                continue

            records.append({
                "scenario": int(scenario_text),
                "event_type": event_type,
                "raw_record": line
            })

events = pd.DataFrame(records)

# ------------------------------------------------------------
# Event type counts
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MALICIOUS EVENT TYPES IN CERT ANSWERS")
print("=" * 70)

if events.empty:
    print("No valid answer records found.")
else:
    print(events["event_type"].value_counts().to_string())

    print("\nScenario-wise:")
    print(
        pd.crosstab(
            events["scenario"],
            events["event_type"]
        ).to_string()
    )

# ------------------------------------------------------------
# Save evidence
# ------------------------------------------------------------

output_file = PROJECT_ROOT / "data" / "outputs" / "rule_event_evidence.parquet"

events.to_parquet(output_file, index=False)

print("\n" + "=" * 70)
print("OUTPUT")
print("=" * 70)
print(f"Saved: {output_file}")
print(f"Evidence rows: {len(events)}")

print("\n" + "=" * 70)
print("EVENT EVIDENCE COMPLETED")
print("=" * 70)