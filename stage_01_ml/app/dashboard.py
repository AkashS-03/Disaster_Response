import os
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Disaster Response AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# STYLING
# =============================================================================
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
    .risk-UNKNOWN { background-color: rgba(120, 120, 120, 0.12); border-color: #888; }
    .metric-card {
        background-color: rgba(255,255,255,0.04);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .caveat-box {
        background-color: rgba(230, 154, 0, 0.08);
        border: 1px solid rgba(230, 154, 0, 0.3);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# PATHS & MODEL LOADING
# =============================================================================
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(base_dir, "models", "risk_model.joblib")


@st.cache_resource
def load_model():
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


model = load_model()

# Zone reference context — used only to give dispatchers situational awareness
# in the UI. These are display hints, not model inputs.
ZONE_CONTEXT = {
    "Zone_A": {"label": "South Mumbai", "flood_history": "Low", "hist_prob": 0.3},
    "Zone_B": {"label": "Bandra", "flood_history": "Low-Moderate", "hist_prob": 0.3},
    "Zone_C": {"label": "Kurla (Mithi River)", "flood_history": "High — chronic flood zone", "hist_prob": 0.9},
    "Zone_D": {"label": "Borivali", "flood_history": "Low-Moderate", "hist_prob": 0.3},
}

RISK_COLOR = {"LOW": "#2ea043", "MODERATE": "#e69a00", "SEVERE": "#da3633", "UNKNOWN": "#888"}

if "history" not in st.session_state:
    st.session_state.history = []  # audit trail of predictions this session

# =============================================================================
# HEADER
# =============================================================================
st.title("🌊 Disaster Response Risk Dashboard")
st.caption("Mumbai Flood Risk — Stage 01 Machine Learning Pipeline")

# =============================================================================
# SIDEBAR — INPUTS
# =============================================================================
st.sidebar.header("Core Telemetry")

zone_id = st.sidebar.selectbox(
    "City Zone",
    list(ZONE_CONTEXT.keys()),
    format_func=lambda z: f"{z} — {ZONE_CONTEXT[z]['label']}",
)
zone_info = ZONE_CONTEXT[zone_id]
st.sidebar.caption(f"Flood history: **{zone_info['flood_history']}**")

river_level = st.sidebar.slider("Current River Level (m)", 0.0, 15.0, 2.5, 0.1)
rainfall = st.sidebar.slider("Current Rainfall (mm/hr)", 0.0, 150.0, 5.0, 0.5)
emergency_call_volume = st.sidebar.slider("Emergency Calls (last hour)", 0, 500, 10)

with st.sidebar.expander("Advanced: rolling & infrastructure inputs"):
    st.caption(
        "These feed the model directly. Defaults below are ESTIMATES derived "
        "from current readings, not real historical rolling data — override "
        "them if you have actual 72h/24h figures for this zone."
    )
    river_level_rolling_72h_avg = st.number_input(
        "River level — 72h rolling avg (m)", 0.0, 15.0, round(river_level * 0.9, 2)
    )
    rainfall_rolling_72h_sum = st.number_input(
        "Rainfall — 72h rolling sum (mm)", 0.0, 4000.0, round(rainfall * 24.0, 1)
    )
    emergency_calls_24h_sum = st.number_input(
        "Emergency calls — 24h sum", 0, 10000, int(emergency_call_volume * 12)
    )
    road_closures = st.number_input("Road closures (active)", 0, 50, int(min(river_level // 3.5, 5)))
    bridge_closures = st.number_input("Bridge closures (active)", 0, 20, int(min(river_level // 4.5, 3)))
    river_level_trend = st.number_input(
        "River level trend (m/hr change)", -5.0, 5.0, 0.3 if river_level > 3.0 else 0.0
    )
    historical_flood_probability = st.number_input(
        "Historical flood probability (zone baseline)", 0.0, 1.0, zone_info["hist_prob"]
    )

st.sidebar.markdown(
    '<div class="caveat-box">⚠️ Rolling/infrastructure fields above are '
    "estimated from current-hour readings for demo purposes. Production use "
    "should feed real 24h/72h aggregates from the data pipeline, not derived "
    "approximations.</div>",
    unsafe_allow_html=True,
)

total_infrastructure_closures = road_closures + bridge_closures

# =============================================================================
# MAIN — PREDICTION
# =============================================================================
col_predict, col_spacer = st.columns([1, 3])
with col_predict:
    run_prediction = st.button("🔍 Predict Risk Level", type="primary", use_container_width=True)

if model is None:
    st.error("Model not found at `models/risk_model.joblib`. Run the ML Engineer pipeline first.")
else:
    if run_prediction:
        # UPDATED: Passing the string "zone_id" directly, matching the new ColumnTransformer ML schema
        input_data = pd.DataFrame(
            [
                {
                    "zone_id": zone_id,
                    "river_level": river_level,
                    "rainfall": rainfall,
                    "emergency_call_volume": emergency_call_volume,
                    "road_closures": road_closures,
                    "bridge_closures": bridge_closures,
                    "historical_flood_probability": historical_flood_probability,
                    "river_level_rolling_72h_avg": river_level_rolling_72h_avg,
                    "rainfall_rolling_72h_sum": rainfall_rolling_72h_sum,
                    "emergency_calls_24h_sum": emergency_calls_24h_sum,
                    "total_infrastructure_closures": total_infrastructure_closures,
                    "river_level_trend": river_level_trend
                }
            ]
        )

        try:
            prediction = model.predict(input_data)[0]
            proba = model.predict_proba(input_data)[0]
            classes = list(model.classes_)
            prob_map = {c: float(p) for c, p in zip(classes, proba)}
            confidence = prob_map.get(prediction, 0.0)

            st.session_state.history.insert(
                0,
                {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "zone": zone_id,
                    "river_level": river_level,
                    "rainfall": rainfall,
                    "prediction": prediction,
                    "confidence": confidence,
                },
            )
            st.session_state.history = st.session_state.history[:15]

            color = RISK_COLOR.get(prediction, RISK_COLOR["UNKNOWN"])
            st.markdown(
                f"""
                <div class="risk-banner risk-{prediction}">
                    <h2 style="color:{color};">⬤ {prediction} RISK — {zone_info['label']}</h2>
                    <p>Model confidence: {confidence:.1%} · Predicted {datetime.now().strftime('%H:%M:%S')}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            m1, m2, m3 = st.columns(3)
            m1.markdown(
                f'<div class="metric-card"><b>River Level</b><br>{river_level:.2f} m</div>',
                unsafe_allow_html=True,
            )
            m2.markdown(
                f'<div class="metric-card"><b>Rainfall</b><br>{rainfall:.1f} mm/hr</div>',
                unsafe_allow_html=True,
            )
            m3.markdown(
                f'<div class="metric-card"><b>Emergency Calls</b><br>{emergency_call_volume}/hr</div>',
                unsafe_allow_html=True,
            )

            st.write("")
            st.subheader("Prediction Confidence Breakdown")
            for cls in ["LOW", "MODERATE", "SEVERE"]:
                if cls in prob_map:
                    st.progress(prob_map[cls], text=f"{cls}: {prob_map[cls]:.1%}")

            # Simple rationale — surfaces the dominant signal driving risk.
            drivers = []
            if river_level >= 4.5:
                drivers.append(f"river level ({river_level:.1f}m) at/above severe threshold")
            elif river_level >= 3.0:
                drivers.append(f"river level ({river_level:.1f}m) elevated")
            if rainfall_rolling_72h_sum >= 150:
                drivers.append(f"72h rainfall accumulation ({rainfall_rolling_72h_sum:.0f}mm) high")
            if emergency_call_volume >= 100:
                drivers.append(f"emergency call volume ({emergency_call_volume}/hr) elevated")
            if drivers:
                st.info("**Key factors:** " + "; ".join(drivers) + ".")
            else:
                st.info("**Key factors:** all core indicators within normal range.")

            if prediction == "SEVERE":
                st.warning(
                    "⚠️ Recommend dispatcher review and activation of zone response "
                    "protocol. Model guarantees high recall on SEVERE cases but "
                    "confirm with live field reports before acting."
                )

        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.markdown(
            '<div class="risk-banner risk-UNKNOWN"><h2>No prediction yet</h2>'
            "<p>Set telemetry values in the sidebar and click Predict Risk Level.</p></div>",
            unsafe_allow_html=True,
        )

# =============================================================================
# PREDICTION HISTORY (AUDIT TRAIL)
# =============================================================================
st.subheader("Session Prediction Log")
if st.session_state.history:
    hist_df = pd.DataFrame(st.session_state.history)
    hist_df["confidence"] = hist_df["confidence"].map(lambda x: f"{x:.1%}")
    st.dataframe(hist_df, use_container_width=True, hide_index=True)
else:
    st.caption("No predictions made yet this session.")

st.markdown("---")
st.caption("Disaster Response Coordination — Stage 01 · Not a substitute for official emergency guidance.")
