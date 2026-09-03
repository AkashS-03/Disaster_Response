import os
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Disaster Response AI", layout="centered")

st.title("🌊 Disaster Response Risk Dashboard")
st.markdown("Predict the disaster risk level based on core telemetry.")

# Set up paths
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(base_dir, "models", "risk_model.joblib")

# Load model
@st.cache_resource
def load_model():
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

# Minimal Sidebar Inputs
st.sidebar.header("Core Inputs")

zone_id = st.sidebar.selectbox("City Zone", ["Zone_A", "Zone_B", "Zone_C", "Zone_D"])
river_level = st.sidebar.slider("Current River Level (m)", 0.0, 15.0, 2.5)
rainfall = st.sidebar.slider("Current Rainfall (mm)", 0.0, 150.0, 5.0)
emergency_call_volume = st.sidebar.slider("Emergency Calls (Hour)", 0, 500, 10)

# Automatically infer hidden/advanced metrics for the ML model to keep UI minimal
# We dynamically scale these based on primary inputs so that max values accurately trigger a 'SEVERE' prediction
road_closures = int(river_level)
bridge_closures = int(river_level / 2)
historical_flood_probability = 0.8 if river_level > 4.0 else 0.1
river_level_rolling_72h_avg = river_level * 0.9  
rainfall_rolling_72h_sum = rainfall * 20   
emergency_calls_24h_sum = emergency_call_volume * 20
total_infrastructure_closures = road_closures + bridge_closures
river_level_trend = 0.5 if river_level > 3.0 else 0.0

st.subheader("Model Prediction")
if model is None:
    st.error("Model not found. Please run the ML Engineer pipeline first.")
else:
    if st.button("Predict Risk Level", type="primary"):
        # Construct DataFrame exactly matching model training
        input_data = pd.DataFrame([{
            'river_level': river_level,
            'rainfall': rainfall,
            'emergency_call_volume': emergency_call_volume,
            'road_closures': road_closures,
            'bridge_closures': bridge_closures,
            'historical_flood_probability': historical_flood_probability,
            'river_level_rolling_72h_avg': river_level_rolling_72h_avg,
            'rainfall_rolling_72h_sum': rainfall_rolling_72h_sum,
            'emergency_calls_24h_sum': emergency_calls_24h_sum,
            'total_infrastructure_closures': total_infrastructure_closures,
            'river_level_trend': river_level_trend,
            'zone_id_Zone_B': 1 if zone_id == 'Zone_B' else 0,
            'zone_id_Zone_C': 1 if zone_id == 'Zone_C' else 0,
            'zone_id_Zone_D': 1 if zone_id == 'Zone_D' else 0
        }])
        
        try:
            prediction = model.predict(input_data)[0]
            proba = model.predict_proba(input_data)[0]
            
            # Dynamic styling based on risk
            color = "green" if prediction == "LOW" else "orange" if prediction == "MODERATE" else "red"
            st.markdown(f"### Predicted Risk: <span style='color:{color}'>{prediction}</span>", unsafe_allow_html=True)
            
            # Show probabilities
            st.write("**Prediction Confidence:**")
            for cls, prob in zip(model.classes_, proba):
                st.progress(float(prob), text=f"{cls}: {prob:.1%}")
                
        except Exception as e:
            st.error(f"Prediction Error: {e}")

st.markdown("---")
st.markdown("*Disaster Response Coordination - Stage 01*")
