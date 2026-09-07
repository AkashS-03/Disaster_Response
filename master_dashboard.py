import os
import io
import joblib
import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models

# =============================================================================
# PAGE SETUP
# =============================================================================
st.set_page_config(
    page_title="Disaster Response AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Clean, modern, minimalist dark styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Clean Result Cards */
    .result-box {
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        border-left: 6px solid;
    }
    
    .box-critical {
        background: rgba(218, 54, 51, 0.15);
        border-color: #da3633;
        color: #ffffff;
    }
    
    .box-warning {
        background: rgba(210, 153, 34, 0.15);
        border-color: #d29922;
        color: #ffffff;
    }
    
    .box-safe {
        background: rgba(46, 160, 67, 0.15);
        border-color: #2ea043;
        color: #ffffff;
    }
    
    .card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    
    .badge {
        display: inline-block;
        font-size: 1rem;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 6px;
    }
    .badge-red { background: #da3633; color: white; }
    .badge-yellow { background: #d29922; color: white; }
    .badge-green { background: #2ea043; color: white; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# MODEL ARCHITECTURE (LSTM)
# =============================================================================
class FloodLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2):
        super(FloodLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# =============================================================================
# CACHED MODEL LOADER
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STAGE_01_DIR = os.path.join(BASE_DIR, "stage_01_ml")
STAGE_02_DIR = os.path.join(BASE_DIR, "stage_02_dl")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_models():
    models_dict = {}
    
    # 1. Classical ML
    ml_p = os.path.join(STAGE_01_DIR, "models", "risk_model.joblib")
    models_dict['ml'] = joblib.load(ml_p) if os.path.exists(ml_p) else None
    
    # 2. LSTM Forecaster
    lstm_p = os.path.join(STAGE_02_DIR, "models", "lstm_forecaster.pth")
    scaler_p = os.path.join(STAGE_02_DIR, "data", "time_series", "ts_scaler.joblib")
    if os.path.exists(lstm_p) and os.path.exists(scaler_p):
        lstm = FloodLSTM(input_size=4, hidden_size=64, num_layers=2).to(DEVICE)
        lstm.load_state_dict(torch.load(lstm_p, map_location=DEVICE, weights_only=True))
        lstm.eval()
        models_dict['lstm'] = lstm
        models_dict['scaler'] = joblib.load(scaler_p)
    else:
        models_dict['lstm'] = None
        models_dict['scaler'] = None

    # 3. Vision CNN
    vision_p = os.path.join(STAGE_02_DIR, "models", "vision_classifier.pth")
    if os.path.exists(vision_p):
        v_model = models.mobilenet_v2()
        v_model.classifier[1] = nn.Linear(v_model.last_channel, 2)
        v_model.load_state_dict(torch.load(vision_p, map_location=DEVICE, weights_only=True))
        v_model.to(DEVICE)
        v_model.eval()
        models_dict['vision'] = v_model
        models_dict['v_transform'] = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        models_dict['vision'] = None
        models_dict['v_transform'] = None
        
    return models_dict

MODELS = load_models()

# Contextual zone dictionary
ZONES = {
    "Zone_C": {"label": "Kurla / Sion (Critical Basin)", "prob": 0.85},
    "Zone_B": {"label": "Bandra / Khar (Moderate)", "prob": 0.65},
    "Zone_A": {"label": "South Mumbai (Low Risk)", "prob": 0.15},
    "Zone_D": {"label": "Borivali / Dahisar (Suburban)", "prob": 0.40}
}

# Sample image paths
SAMPLE_FLOOD = os.path.join(STAGE_02_DIR, "data", "vision", "flooded", "drone_flood_00001.png")
SAMPLE_CLEAR = os.path.join(STAGE_02_DIR, "data", "vision", "clear", "drone_clear_00001.png")

# State for selected image
if "selected_img" not in st.session_state:
    if os.path.exists(SAMPLE_FLOOD):
        st.session_state["selected_img"] = Image.open(SAMPLE_FLOOD).convert("RGB")
        st.session_state["img_name"] = "Sample Flooded Drone Feed"
    else:
        st.session_state["selected_img"] = None
        st.session_state["img_name"] = "No image loaded"

# =============================================================================
# HEADER
# =============================================================================
st.title("🌊 Disaster Response AI")
st.caption("Simple Multi-Modal Command Center: Drone Flood Detection, River Forecasting & Emergency Classification")
st.write("")

# =============================================================================
# SIMPLE TWO-COLUMN LAYOUT
# =============================================================================
col_input, col_result = st.columns([1, 1], gap="large")

# -----------------------------------------------------------------------------
# LEFT COLUMN: INPUTS
# -----------------------------------------------------------------------------
with col_input:
    st.subheader("1. Drone Image Input")
    
    # Quick buttons to pick sample images
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("🌊 Sample Flooded Image", use_container_width=True):
            if os.path.exists(SAMPLE_FLOOD):
                st.session_state["selected_img"] = Image.open(SAMPLE_FLOOD).convert("RGB")
                st.session_state["img_name"] = "Sample Flooded Drone Shot"
    with btn_col2:
        if st.button("☀️ Sample Clear Image", use_container_width=True):
            if os.path.exists(SAMPLE_CLEAR):
                st.session_state["selected_img"] = Image.open(SAMPLE_CLEAR).convert("RGB")
                st.session_state["img_name"] = "Sample Clear Drone Shot"

    # File uploader
    uploaded = st.file_uploader("Or Upload Custom Drone Image", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        st.session_state["selected_img"] = Image.open(uploaded).convert("RGB")
        st.session_state["img_name"] = uploaded.name

    # Image Preview (Fixed: using use_column_width=True for Streamlit 1.36)
    if st.session_state["selected_img"] is not None:
        st.image(st.session_state["selected_img"], caption=st.session_state["img_name"], use_column_width=True)

    st.write("")
    st.subheader("2. River & Weather Conditions")
    
    zone_key = st.selectbox("Select Zone", list(ZONES.keys()), format_func=lambda z: f"{z} ({ZONES[z]['label']})")
    
    river_level = st.slider("Current River Level (meters)", 0.0, 15.0, 4.2, 0.1)
    rainfall = st.slider("Rainfall Rate (mm/hr)", 0.0, 150.0, 40.0, 1.0)
    emergency_calls = st.slider("Emergency Calls / hr", 0, 500, 60)

# -----------------------------------------------------------------------------
# RIGHT COLUMN: PREDICTIONS & FINAL RESULT CLASS
# -----------------------------------------------------------------------------
with col_result:
    st.subheader("Assessment Results")
    
    # --- 1. RUN CNN DRONE INFERENCE ---
    vision_label = "UNKNOWN"
    vision_conf = 0.0
    if MODELS['vision'] is not None and st.session_state["selected_img"] is not None:
        try:
            tensor = MODELS['v_transform'](st.session_state["selected_img"]).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out = MODELS['vision'](tensor)
                pred = torch.argmax(out, dim=1).item()
                prob = torch.softmax(out, dim=1)[0][pred].item()
            vision_label = "FLOODED" if pred == 1 else "CLEAR"
            vision_conf = prob * 100.0
        except Exception as e:
            st.error(f"CNN Error: {e}")

    # --- 2. RUN LSTM RIVER FORECAST ---
    predicted_river_12h = river_level
    if MODELS['lstm'] is not None and MODELS['scaler'] is not None:
        try:
            # Simple sequence leading to current river level
            seq_river = np.linspace(max(0.5, river_level - 1.2), river_level, 48)
            seq_rain = np.linspace(max(0, rainfall - 20), rainfall, 48)
            seq_calls = np.linspace(emergency_calls * 0.7, emergency_calls, 48)
            seq_closures = np.full(48, min(river_level // 1.5, 10))
            
            raw_seq = np.column_stack([seq_river, seq_rain, seq_calls, seq_closures])
            scaled_seq = MODELS['scaler'].transform(raw_seq)
            t_in = torch.tensor(scaled_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                pred_scaled = MODELS['lstm'](t_in).cpu().numpy()
                
            dummy = np.zeros((1, 4))
            dummy[0, 0] = pred_scaled[0, 0]
            predicted_river_12h = float(MODELS['scaler'].inverse_transform(dummy)[0, 0])
        except Exception as e:
            predicted_river_12h = river_level + 0.5

    # --- 3. COMPUTE FINAL RESULT CLASS ---
    # Combine Vision + LSTM + Telemetry into one clear disaster status
    is_flood_img = (vision_label == "FLOODED")
    is_river_high = (predicted_river_12h >= 5.0)
    is_rain_heavy = (rainfall >= 50.0)

    if (is_flood_img and is_river_high) or predicted_river_12h >= 7.0:
        final_class = "CRITICAL FLOOD EMERGENCY"
        box_style = "box-critical"
        badge_style = "badge-red"
        advice = "Immediate evacuation of low-lying areas required. Dispatch rescue teams and close bridges."
    elif is_flood_img or is_river_high or is_rain_heavy:
        final_class = "MODERATE FLOOD WARNING"
        box_style = "box-warning"
        badge_style = "badge-yellow"
        advice = "Water accumulation observed. Monitor river banks, stage emergency pumps, and divert traffic."
    else:
        final_class = "NORMAL / SAFE"
        box_style = "box-safe"
        badge_style = "badge-green"
        advice = "Normal operational status. No active flood emergency detected."

    # --- DISPLAY: FINAL RESULT CLASS (PROMINENT) ---
    st.markdown(f"""
    <div class="result-box {box_style}">
        <span style="font-size: 0.85rem; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">
            Final Result Class
        </span>
        <h2 style="margin: 4px 0 8px 0; font-size: 1.8rem; font-weight: 800;">
            {final_class}
        </h2>
        <p style="margin: 0; font-size: 1rem; opacity: 0.95;">
            <b>Action Directive:</b> {advice}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- DISPLAY: DRONE CNN DETECTION ---
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<b>📸 Drone Detection Result (CNN)</b>", unsafe_allow_html=True)
    
    if vision_label == "FLOODED":
        st.markdown(f'<div style="margin-top: 8px;"><span class="badge badge-red">🌊 FLOODED</span> &nbsp; Confidence: <b>{vision_conf:.1f}%</b></div>', unsafe_allow_html=True)
    elif vision_label == "CLEAR":
        st.markdown(f'<div style="margin-top: 8px;"><span class="badge badge-green">☀️ CLEAR / DRY</span> &nbsp; Confidence: <b>{vision_conf:.1f}%</b></div>', unsafe_allow_html=True)
    else:
        st.write("No image loaded.")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- DISPLAY: LSTM RIVER PREDICTION ---
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<b>📈 LSTM 12-Hour River Prediction</b>", unsafe_allow_html=True)
    
    p_c1, p_c2 = st.columns(2)
    with p_c1:
        st.metric("Current River Level", f"{river_level:.2f} m")
    with p_c2:
        delta = predicted_river_12h - river_level
        st.metric("Predicted Level in 12h", f"{predicted_river_12h:.2f} m", delta=f"{delta:+.2f} m", delta_color="inverse")
        
    # Clean, simple mini chart
    timeline = ["Now", "+3h", "+6h", "+9h", "+12h"]
    proj_vals = np.linspace(river_level, predicted_river_12h, 5)
    chart_df = pd.DataFrame({"Timeline": timeline, "Projected River (m)": proj_vals, "Danger Line (5.0m)": [5.0]*5}).set_index("Timeline")
    st.line_chart(chart_df, color=["#58a6ff", "#da3633"], height=160)
    st.markdown('</div>', unsafe_allow_html=True)
