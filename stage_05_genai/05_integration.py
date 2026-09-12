"""
ROLE: INTEGRATION ENGINEER (GenAI stage)
JOB: "Integrate synthetic scenario output back into the testing dashboard
for continuous evaluation."

WHY THIS STEP EXISTS:
The whole point of generating synthetic disasters is to feed them BACK
into the real pipeline (Stage 1's ML model) as a stress test. This closes
the loop between GenAI and ML.
"""

import os
import sys
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

synth_csv = os.path.join(DATA_DIR, "synthetic_scenarios.csv")
if not os.path.exists(synth_csv) and os.path.exists("data/synthetic_scenarios.csv"):
    synth_csv = "data/synthetic_scenarios.csv"

synthetic = pd.read_csv(synth_csv)

# Try to reuse the REAL trained model from Stage 1 (shows how stages connect)
stage1_model_candidates = [
    os.path.join(BASE_DIR, "stage_01_ml", "models", "risk_model.joblib"),
    "../Stage01_ML/data/risk_model.pkl",
    "stage_01_ml/models/risk_model.joblib"
]

stage1_model_path = next((p for p in stage1_model_candidates if os.path.exists(p)), None)

if stage1_model_path is not None:
    import joblib
    sys.path.insert(0, os.path.join(BASE_DIR, "stage_01_ml"))
    try:
        import safety_guard  # noqa: F401
    except ImportError:
        pass

    model = joblib.load(stage1_model_path)
    print(f"Loaded the real Stage 1 ML risk model ({stage1_model_path}) to stress-test with synthetic data.\n")

    for i, row in synthetic.iterrows():
        # Construct input row conforming to Stage 1 model requirements
        zone_id = "Zone_A" if row["risk_label"] == "LOW" else ("Zone_B" if row["risk_label"] == "MODERATE" else "Zone_C")
        
        # Build features DataFrame with all required columns
        input_row = pd.DataFrame([{
            'zone_id': zone_id,
            'river_level': float(row["gauge_level_m"]),
            'rainfall': float(row["rainfall_mm"]),
            'emergency_call_volume': int(row["call_volume"]),
            'road_closures': 2,
            'bridge_closures': 0 if row["risk_label"] == "LOW" else 1,
            'historical_flood_probability': 0.15 if zone_id == "Zone_A" else (0.65 if zone_id == "Zone_B" else 0.85),
            'river_level_rolling_72h_avg': float(max(1.0, row["gauge_level_m"] - 0.2)),
            'rainfall_rolling_72h_sum': float(max(10.0, row["rainfall_mm"] * 2.2)),
            'emergency_calls_24h_sum': int(row["call_volume"] * 7),
            'total_infrastructure_closures': 3,
            'river_level_trend': 0.25
        }])

        prediction = model.predict(input_row)[0]
        match = "MATCHES expected label" if prediction == row["risk_label"] else "MISMATCH - investigate!"
        print(f"Synthetic scenario {i+1} [{row['scenario_name']}]: expected={row['risk_label']}, "
              f"model predicted={prediction} -> {match}")
else:
    print("(Run Stage01_ML first to enable full cross-stage stress testing.)")
    print("Showing synthetic scenarios only:\n")
    print(synthetic)

print("\nThis is how Generative AI stress-tests the earlier ML stage before real deployment.")
