import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import os
from datetime import datetime

API_URL = "http://127.0.0.1:8001"

st.set_page_config(
    page_title="Disaster Response AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- STYLING ---
st.markdown(
    """
    <style>
    .risk-banner {
        padding: 1.25rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.25rem;
        border-left: 8px solid;
    }
    .risk-banner h2 { margin: 0; font-size: 1.6rem; }
    .risk-banner p { margin: 0.25rem 0 0 0; opacity: 0.85; }
    .risk-LOW { background-color: rgba(46, 160, 67, 0.15); border-color: #2ea043; }
    .risk-MODERATE { background-color: rgba(230, 154, 0, 0.15); border-color: #e69a00; }
    .risk-SEVERE { background-color: rgba(218, 54, 51, 0.18); border-color: #da3633; }
    .risk-UNKNOWN { background-color: rgba(139, 148, 158, 0.15); border-color: #8b949e; color: #8b949e; }
    .metric-card {
        background-color: #1e1e1e;
        border: 1px solid #333;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card b { color: #8b949e; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em; }
    .caveat-box {
        font-size: 0.85rem; color: #a1a1aa; padding: 0.75rem;
        background-color: #1a1a1a; border-left: 4px solid #555;
        border-radius: 4px; margin: 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Disaster Response Command Center")
st.markdown("Unified Intelligence Platform: Tabular Machine Learning (Stage 1) & Multi-modal Deep Learning (Stage 2)")

main_tab1, main_tab2, main_tab3, main_tab4 = st.tabs([
    "📊 Stage 1: Tabular ML Risk", 
    "📈 Stage 2: River Forecast (LSTM)", 
    "💬 Stage 2: Dispatch Text (GRU)", 
    "📸 Stage 2: Drone Feed (CNN)"
])

# =============================================================================
# STAGE 1: TABULAR ML RISK PREDICTION
# =============================================================================
with main_tab1:
    ZONE_CONTEXT = {
        "Zone_A": {"label": "South Mumbai", "flood_history": "Low", "hist_prob": 0.15},
        "Zone_B": {"label": "Bandra/Khar", "flood_history": "High", "hist_prob": 0.65},
        "Zone_C": {"label": "Kurla/Sion", "flood_history": "Critical", "hist_prob": 0.85},
        "Zone_D": {"label": "Borivali/Dahisar", "flood_history": "Moderate", "hist_prob": 0.40},
    }
    RISK_COLOR = {"LOW": "#2ea043", "MODERATE": "#e69a00", "SEVERE": "#da3633", "UNKNOWN": "#8b949e"}
    
    if "history" not in st.session_state:
        st.session_state.history = []

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "stage_01_ml", "models", "risk_model.joblib")
    
    @st.cache_resource
    def load_model():
        if os.path.exists(model_path):
            return joblib.load(model_path)
        return None

    model = load_model()
    
    st.sidebar.header("Stage 1 Core Telemetry")
    zone_id = st.sidebar.selectbox("City Zone", list(ZONE_CONTEXT.keys()), format_func=lambda z: f"{z} — {ZONE_CONTEXT[z]['label']}")
    zone_info = ZONE_CONTEXT[zone_id]
    
    river_level = st.sidebar.slider("Current River Level (m)", 0.0, 15.0, 2.5, 0.1)
    rainfall = st.sidebar.slider("Current Rainfall (mm/hr)", 0.0, 150.0, 5.0, 0.5)
    emergency_call_volume = st.sidebar.slider("Emergency Calls (last hour)", 0, 500, 10)

    with st.sidebar.expander("Advanced: rolling & infrastructure inputs"):
        river_level_rolling_72h_avg = st.number_input("River level — 72h rolling avg (m)", 0.0, 15.0, round(river_level * 0.9, 2))
        rainfall_rolling_72h_sum = st.number_input("Rainfall — 72h rolling sum (mm)", 0.0, 4000.0, round(rainfall * 24.0, 1))
        emergency_calls_24h_sum = st.number_input("Emergency calls — 24h sum", 0, 10000, int(emergency_call_volume * 12))
        road_closures = st.number_input("Road closures (active)", 0, 50, int(min(river_level // 3.5, 5)))
        bridge_closures = st.number_input("Bridge closures (active)", 0, 20, int(min(river_level // 4.5, 3)))
        river_level_trend = st.number_input("River level trend (m/hr change)", -5.0, 5.0, 0.3 if river_level > 3.0 else 0.0)
        historical_flood_probability = st.number_input("Historical flood probability", 0.0, 1.0, zone_info["hist_prob"])

    total_infrastructure_closures = road_closures + bridge_closures

    col_predict, col_spacer = st.columns([1, 3])
    with col_predict:
        run_prediction = st.button("🚨 Predict Risk Level", type="primary", use_container_width=True)

    if model is None:
        st.error(f"Stage 1 ML Model not found at `{model_path}`.")
    else:
        if run_prediction:
            input_data = pd.DataFrame([{
                "zone_id": zone_id, "river_level": river_level, "rainfall": rainfall,
                "emergency_call_volume": emergency_call_volume, "road_closures": road_closures,
                "bridge_closures": bridge_closures, "historical_flood_probability": historical_flood_probability,
                "river_level_rolling_72h_avg": river_level_rolling_72h_avg, "rainfall_rolling_72h_sum": rainfall_rolling_72h_sum,
                "emergency_calls_24h_sum": emergency_calls_24h_sum, "total_infrastructure_closures": total_infrastructure_closures,
                "river_level_trend": river_level_trend
            }])

            try:
                prediction = model.predict(input_data)[0]
                proba = model.predict_proba(input_data)[0]
                classes = list(model.classes_)
                prob_map = {c: float(p) for c, p in zip(classes, proba)}
                confidence = prob_map.get(prediction, 0.0)

                st.session_state.history.insert(0, {
                    "time": datetime.now().strftime("%H:%M:%S"), "zone": zone_id, "river_level": river_level,
                    "rainfall": rainfall, "prediction": prediction, "confidence": confidence,
                })
                st.session_state.history = st.session_state.history[:15]

                color = RISK_COLOR.get(prediction, RISK_COLOR["UNKNOWN"])
                st.markdown(f'''
                    <div class="risk-banner risk-{prediction}">
                        <h2 style="color:{color};">⚠ {prediction} RISK — {zone_info['label']}</h2>
                        <p>Model confidence: {confidence:.1%} | Predicted {datetime.now().strftime('%H:%M:%S')}</p>
                    </div>''', unsafe_allow_html=True)

                m1, m2, m3 = st.columns(3)
                m1.markdown(f'<div class="metric-card"><b>River Level</b><br>{river_level:.2f} m</div>', unsafe_allow_html=True)
                m2.markdown(f'<div class="metric-card"><b>Rainfall</b><br>{rainfall:.1f} mm/hr</div>', unsafe_allow_html=True)
                m3.markdown(f'<div class="metric-card"><b>Emergency Calls</b><br>{emergency_call_volume}/hr</div>', unsafe_allow_html=True)

                st.write("")
                st.subheader("Prediction Confidence Breakdown")
                for cls in ["LOW", "MODERATE", "SEVERE"]:
                    if cls in prob_map:
                        st.progress(prob_map[cls], text=f"{cls}: {prob_map[cls]:.1%}")
            except Exception as e:
                st.error(f"Prediction error: {e}")
        else:
            st.markdown('<div class="risk-banner risk-UNKNOWN"><h2>No prediction yet</h2><p>Set telemetry values in the sidebar and click Predict Risk Level.</p></div>', unsafe_allow_html=True)

# =============================================================================
# STAGE 2: DL LSTM FORECAST
# =============================================================================
with main_tab2:
    st.header("Time-Series River Forecaster (PyTorch LSTM)")
    st.write("Predicts the river level 12 hours into the future based on 48-hour trailing telemetry.")
    if st.button("Generate 48h Mock Sequence & Forecast"):
        mock_seq = []
        base_river = 4.0
        for i in range(48):
            mock_seq.append([base_river + (i * 0.05), 20.0 + np.random.normal(0, 5), 100 + i*2, 3])
        try:
            res = requests.post(f"{API_URL}/dl/forecast", json={"sequence": mock_seq})
            if res.status_code == 200:
                data = res.json()
                st.success(f"**Predicted River Level (in 12 hours):** {data['predicted_river_level_12h']:.2f} meters")
                st.line_chart([x[0] for x in mock_seq] + [data['predicted_river_level_12h']])
            else:
                st.error("API Error")
        except Exception as e:
            st.error(f"Failed to connect to API: {e}. Is `api_dl.py` running?")

# =============================================================================
# STAGE 2: DL NLP TRANSCRIPT
# =============================================================================
with main_tab3:
    st.header("Dispatch Transcript Analyzer (PyTorch GRU)")
    st.write("Extracts disaster severity automatically from 911 dispatch transcripts.")
    transcript = st.text_area("Live 911 Transcript", placeholder="e.g., People are trapped on the roof on MG Road!")
    if st.button("Analyze Severity"):
        if transcript:
            try:
                res = requests.post(f"{API_URL}/dl/text", json={"transcript": transcript})
                if res.status_code == 200:
                    severity = res.json()['severity']
                    colors = {"LOW": "green", "MODERATE": "orange", "SEVERE": "red"}
                    st.markdown(f"<h3 style='color: {colors[severity]};'>Detected Severity: {severity}</h3>", unsafe_allow_html=True)
                else:
                    st.error("API Error")
            except Exception as e:
                st.error(f"Failed to connect to API: {e}. Is `api_dl.py` running?")

# =============================================================================
# STAGE 2: DL VISION CNN
# =============================================================================
with main_tab4:
    st.header("UAV/Drone Feed Classifier (MobileNetV2)")
    st.write("Analyzes live images to detect urban flooding.")
    uploaded_file = st.file_uploader("Upload Drone/Camera Feed", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Feed", width=400)
        if st.button("Process Image"):
            try:
                files = {"file": ("image.jpg", uploaded_file.getvalue(), "image/jpeg")}
                res = requests.post(f"{API_URL}/dl/vision", files=files)
                if res.status_code == 200:
                    data = res.json()
                    classification = data['classification']
                    conf = data['confidence'] * 100
                    color = "red" if classification == "FLOODED" else "green"
                    st.markdown(f"<h3 style='color: {color};'>Classification: {classification} ({conf:.1f}% confidence)</h3>", unsafe_allow_html=True)
                else:
                    st.error("API Error")
            except Exception as e:
                st.error(f"Failed to connect to API: {e}. Is `api_dl.py` running?")
