from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "outputs"

RISK_FILE = OUT / "risk_engine_results.parquet"
INTEGRATION_FILE = OUT / "agent_integration_dataset.parquet"
OUTPUT_FILE = OUT / "alert_explanations.parquet"

KEYS = ["user", "day"]

AGENT_COLUMNS = {
    "rule": ["rule_final_rule_score", "rule_xgb_probability", "rule_final_threat_level"],
    "login": ["login_login_agent_score", "login_login_agent_status", "login_login_explanation"],
    "file": ["file_file_agent_score", "file_file_agent_status", "file_file_explanation"],
    "device": ["device_device_agent_score", "device_device_agent_status", "device_device_explanation"],
    "network": ["network_network_v2_score", "network_network_v2_status"],
}


def main():
    risk = pd.read_parquet(RISK_FILE)
    evidence = pd.read_parquet(INTEGRATION_FILE)

    for name, df in [("Risk Engine", risk), ("Integration", evidence)]:
        missing = [c for c in KEYS if c not in df.columns]
        if missing:
            raise ValueError(f"{name} is missing keys: {missing}")

    risk["day"] = pd.to_datetime(risk["day"], errors="coerce")
    evidence["day"] = pd.to_datetime(evidence["day"], errors="coerce")

    if risk.duplicated(KEYS).any() or evidence.duplicated(KEYS).any():
        raise ValueError("Duplicate user/day keys detected; refusing unsafe merge.")

    available = {
        agent: [col for col in cols if col in evidence.columns]
        for agent, cols in AGENT_COLUMNS.items()
    }

    merged = risk.merge(
        evidence[KEYS + sorted({c for cols in available.values() for c in cols})],
        on=KEYS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    merged["evidence_match"] = merged["_merge"].eq("both")
    merged.drop(columns="_merge", inplace=True)

    for agent, cols in available.items():
        for col in cols:
            if any(token in col for token in ("_score", "_probability")):
                merged[col] = pd.to_numeric(merged[col], errors="coerce")

    def build_explanation(row):
        level = row.get("risk_level", "UNKNOWN")
        evidence_items = []

        for agent, cols in available.items():
            score_cols = [c for c in cols if "_score" in c or "_probability" in c]
            status_cols = [c for c in cols if "_status" in c]
            explanation_cols = [c for c in cols if "_explanation" in c]
            threat_cols = [c for c in cols if "_threat_level" in c]

            for col in score_cols:
                value = row.get(col)
                if pd.notna(value):
                    evidence_items.append(f"{agent} score={float(value):.3f}")

            for col in status_cols + threat_cols:
                value = row.get(col)
                if pd.notna(value) and str(value).strip():
                    evidence_items.append(f"{agent} {col.rsplit('_', 1)[-1]}={value}")

            for col in explanation_cols:
                value = row.get(col)
                if pd.notna(value) and str(value).strip():
                    evidence_items.append(f"{agent} explanation: {str(value).strip()}")

        if level == "HIGH":
            opening = "High-risk user-day based on the Supervisor score; prioritize investigation."
        elif level == "MEDIUM":
            opening = "Medium-risk user-day based on the Supervisor score; review activity."
        elif level == "LOW":
            opening = "Low-risk user-day based on the Supervisor score; continue routine monitoring."
        else:
            opening = "Risk is unavailable; verify data and assess manually."

        if evidence_items:
            return opening + " Agent evidence: " + "; ".join(evidence_items) + "."
        return opening + " No usable agent evidence was available."

    def action_for_level(level):
        if level == "HIGH":
            return "Investigate supporting logs and validate the activity before taking action."
        if level == "MEDIUM":
            return "Review the agent indicators and compare with normal user activity."
        if level == "LOW":
            return "Continue routine monitoring."
        return "Verify the data and assess manually."

    merged["explanation"] = merged.apply(build_explanation, axis=1)
    merged["recommended_action"] = merged["risk_level"].apply(action_for_level)
    merged["explanation_source"] = merged["evidence_match"].map({
        True: "Supervisor risk plus matched agent integration evidence",
        False: "Supervisor risk only; no matching integration record",
    })

    output_columns = [
        c for c in [
            "user", "day", "supervisor_risk_score", "risk_level",
            "alert_priority", "alert_status", "explanation",
            "explanation_source", "recommended_action", "evidence_match",
        ] if c in merged.columns
    ]
    for cols in available.values():
        output_columns.extend(c for c in cols if c in merged.columns and c not in output_columns)

    result = merged[output_columns].copy()
    result.to_parquet(OUTPUT_FILE, index=False)

    print("=" * 60)
    print("ALERT EXPLANATIONS UPDATED")
    print("=" * 60)
    print(f"Rows: {len(result):,}")
    print(f"Matched evidence: {int(result['evidence_match'].sum()):,}")
    print(f"Agent evidence columns used: {available}")
    print(f"Saved: {OUTPUT_FILE}")
    print("\nSample explanations:")
    for _, row in result[result["risk_level"].isin(["HIGH", "MEDIUM"])].head(3).iterrows():
        print(f"\nUser: {row['user']} | Day: {row['day']} | Risk: {row['risk_level']}")
        print(row["explanation"])
        print("Action:", row["recommended_action"])


if __name__ == "__main__":
    main()
