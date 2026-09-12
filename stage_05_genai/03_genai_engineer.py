"""
ROLE: GENAI ENGINEER (GenAI stage)
JOB: "Synthesize novel disaster scenarios by sampling from baseline
distributions and targeting identified blind spots."

WHY THIS STEP EXISTS:
Using the statistical baseline from the Data Engineer and the targeted
prompts from the EDA/Prompt Engineer, this role generates the synthetic
test scenarios (including compound edge cases) that will battle-test
earlier pipeline stages.
"""

import os
import json
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

baseline_pkl = os.path.join(DATA_DIR, "baseline_stats.pkl")
if not os.path.exists(baseline_pkl) and os.path.exists("data/baseline_stats.pkl"):
    baseline_pkl = "data/baseline_stats.pkl"

blind_path = os.path.join(DATA_DIR, "blind_spots.json")
if not os.path.exists(blind_path) and os.path.exists("data/blind_spots.json"):
    blind_path = "data/blind_spots.json"

baseline = pd.read_pickle(baseline_pkl)
print("Loaded baseline distributions:")
print(baseline)

blind_spots = []
if os.path.exists(blind_path):
    with open(blind_path, "r", encoding="utf-8") as f:
        blind_spots = json.load(f)

np.random.seed(42)
synthetic_records = []

# 1. Sample baseline distributions across LOW, MODERATE, SEVERE
for sev in ["LOW", "MODERATE", "SEVERE"]:
    r_mean = baseline.loc[sev, ("rainfall_mm", "mean")]
    r_std = baseline.loc[sev, ("rainfall_mm", "std")]
    g_mean = baseline.loc[sev, ("gauge_level_m", "mean")]
    g_std = baseline.loc[sev, ("gauge_level_m", "std")]
    c_mean = baseline.loc[sev, ("call_volume", "mean")]
    c_std = baseline.loc[sev, ("call_volume", "std")]

    for i in range(350):
        rain = min(245.0, max(0.0, float(np.random.normal(r_mean, r_std * 0.8))))
        gauge = min(9.5, max(0.5, float(np.random.normal(g_mean, g_std * 0.8))))
        calls = min(950, max(5, int(np.random.normal(c_mean, c_std * 0.8))))
        synthetic_records.append({
            "scenario_name": f"Synthetic_{sev}_Sample_{i+1:04d}",
            "rainfall_mm": round(rain, 1),
            "gauge_level_m": round(gauge, 2),
            "call_volume": calls,
            "risk_label": sev,
            "regime": "Baseline Distribution Sample"
        })

# 2. Inject targeted compound blind-spot scenarios
for b in blind_spots:
    synthetic_records.append({
        "scenario_name": b["scenario_name"],
        "rainfall_mm": b["rainfall_mm"],
        "gauge_level_m": b["gauge_level_m"],
        "call_volume": b["call_volume"],
        "risk_label": b["risk_label"],
        "regime": "Targeted Compound Blind Spot"
    })

synthetic_df = pd.DataFrame(synthetic_records)
out_csv = os.path.join(DATA_DIR, "synthetic_scenarios.csv")
synthetic_df.to_csv(out_csv, index=False)

if os.path.exists("data"):
    synthetic_df.to_csv("data/synthetic_scenarios.csv", index=False)

print(f"\nGenerated {len(synthetic_df)} synthetic scenarios saved to {out_csv}:")
print(synthetic_df[["scenario_name", "rainfall_mm", "gauge_level_m", "call_volume", "risk_label"]])
