"""
ROLE: DATA ENGINEER (GenAI stage)
JOB: "Assemble reference baseline datasets from real historical disaster
distributions."

WHY THIS STEP EXISTS:
Before generating FAKE (synthetic) disasters, we must understand what
REAL ones look like statistically - otherwise the fake ones won't be
realistic.
"""

import os
import pandas as pd

# Support running from workspace root or inside stage_05_genai directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

csv_path = os.path.join(DATA_DIR, "historical_disaster_stats.csv")
if not os.path.exists(csv_path) and os.path.exists("data/historical_disaster_stats.csv"):
    csv_path = "data/historical_disaster_stats.csv"

df = pd.read_csv(csv_path)

print("Historical data loaded:")
print(df.head())
print(f"\nTotal historical records: {len(df)}")

# STEP: Compute the statistical baseline (mean + spread) per risk level
# This baseline is what the GenAI Engineer will sample new scenarios from.
baseline = df.groupby("risk_label")[["rainfall_mm", "gauge_level_m", "call_volume"]].agg(["mean", "std"])
print("\nStatistical baseline per risk level (mean, std):")
print(baseline)

baseline_pkl = os.path.join(DATA_DIR, "baseline_stats.pkl")
historical_pkl = os.path.join(DATA_DIR, "historical_df.pkl")

baseline.to_pickle(baseline_pkl)
df.to_pickle(historical_pkl)

# Also write to local "data/" if running from repository root
if os.path.exists("data"):
    baseline.to_pickle("data/baseline_stats.pkl")
    df.to_pickle("data/historical_df.pkl")

print(f"\nSaved baseline to {baseline_pkl}")
print(f"Saved historical dataframe to {historical_pkl}")
