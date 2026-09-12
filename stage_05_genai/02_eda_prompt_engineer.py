"""
ROLE: EDA / PROMPT ENGINEER
JOB: "Identify blind spots in historical data and craft prompts for
extreme edge scenarios."

WHY THIS STEP EXISTS:
Historical data alone won't include RARE, extreme combinations (e.g. very
high rainfall AND very high call volume happening together). This role
identifies which combinations are missing/rare, so the GenAI Engineer
knows what to specifically target when generating synthetic data.
"""

import os
import json
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

hist_pkl = os.path.join(DATA_DIR, "historical_df.pkl")
if not os.path.exists(hist_pkl) and os.path.exists("data/historical_df.pkl"):
    hist_pkl = "data/historical_df.pkl"

df = pd.read_pickle(hist_pkl)

# STEP 1: Find the observed range of each signal
print("Observed ranges in historical data:")
for col in ["rainfall_mm", "gauge_level_m", "call_volume"]:
    print(f"  {col}: min={df[col].min()}, max={df[col].max()}")

# STEP 2: Define a "blind spot" - a compound extreme not yet seen
# e.g. rainfall > max observed AND call_volume > max observed together
max_rain = df["rainfall_mm"].max()
max_calls = df["call_volume"].max()
max_gauge = df["gauge_level_m"].max()

print(f"\nBLIND SPOT identified: no historical case has rainfall > {max_rain}mm "
      f"AND call_volume > {max_calls} happening TOGETHER.")
print("This is exactly the kind of compound extreme scenario the GenAI "
      "Engineer should generate next, to stress-test the pipeline.")

# A "prompt" for a real LLM-based generator would look like this:
prompt = (f"Generate a disaster scenario with rainfall above {max_rain}mm, "
          f"call volume above {max_calls}, combined with a power outage.")
print(f"\nExample prompt for a real GenAI system:\n  '{prompt}'")

# STEP 3: Catalog structured prompts for the GenAI Engineer
blind_spots = [
    {
        "id": "BLIND-01",
        "scenario_name": "Hyper-Deluge & Total Grid Blackout",
        "condition": f"rainfall > {max_rain}mm AND call_volume > {max_calls}",
        "rainfall_mm": round(float(max_rain * 1.10), 1),
        "gauge_level_m": round(float(max_gauge * 1.05), 2),
        "call_volume": int(max_calls * 1.10),
        "risk_label": "SEVERE",
        "notes": "Simultaneous peak atmospheric rainfall and grid communications collapse."
    },
    {
        "id": "BLIND-02",
        "scenario_name": "Adversarial Submerged Sensor (0.0m Gauge)",
        "condition": "river gauge submerged and reading 0.0m during 180mm storm",
        "rainfall_mm": 180.0,
        "gauge_level_m": 0.0,
        "call_volume": int(max_calls * 0.95),
        "risk_label": "SEVERE",
        "notes": "Tests if pipeline falls prey to sensor failure or catches rain/call elevation."
    },
    {
        "id": "BLIND-03",
        "scenario_name": "4.8m Astronomical Spring Tide Outfall Lock",
        "condition": "astronomical high tide locking sea gates with steady 95mm downpour",
        "rainfall_mm": 95.0,
        "gauge_level_m": 5.85,
        "call_volume": 320,
        "risk_label": "SEVERE",
        "notes": "Drainage channels blocked by sea backpressure."
    }
]

blind_path = os.path.join(DATA_DIR, "blind_spots.json")
with open(blind_path, "w", encoding="utf-8") as f:
    json.dump(blind_spots, f, indent=2)

print(f"\nSaved targeted blind-spot specifications to {blind_path}")
