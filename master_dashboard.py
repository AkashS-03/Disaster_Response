import os
import sys
import io
import joblib
import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models

# Ensure stage_01_ml is importable so joblib can unpickle GuardRailedPredictor
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STAGE_01_DIR = os.path.join(BASE_DIR, "stage_01_ml")
STAGE_02_DIR = os.path.join(BASE_DIR, "stage_02_dl")
if STAGE_01_DIR not in sys.path:
    sys.path.insert(0, STAGE_01_DIR)
import safety_guard  # noqa: F401 – needed so joblib can unpickle GuardRailedPredictor

# =============================================================================
# PAGE SETUP
# =============================================================================
st.set_page_config(
    page_title="AquaShield Command",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# STYLING — full rework, "mission control" aesthetic
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ------------------ BASE ------------------ */
    .stApp {
        background:
            linear-gradient(180deg, #05070d 0%, #0a101e 45%, #0a1120 100%);
        color: #e8eefc;
    }

    /* animated grid floor (sci-fi) */
    .stApp::before {
        content: "";
        position: fixed; inset: 0;
        background-image:
            linear-gradient(rgba(70,120,255,0.05) 1px, transparent 1px),
            linear-gradient(90deg, rgba(70,120,255,0.05) 1px, transparent 1px);
        background-size: 46px 46px;
        -webkit-mask-image: radial-gradient(1000px 600px at 50% 0%, #000 40%, transparent 90%);
        mask-image: radial-gradient(1000px 600px at 50% 0%, #000 40%, transparent 90%);
        pointer-events: none; z-index: 0;
        animation: gridShift 24s linear infinite;
    }
    @keyframes gridShift { to { background-position: 46px 46px; } }

    [data-testid="stHeader"] { background: transparent; }
    .stApp [data-testid="stAppViewContainer"] > .main { position: relative; z-index: 1; }
    .block-container { max-width: 1240px; padding-top: 1.2rem; padding-bottom: 3rem; }

    /* ------------------ TOPBAR ------------------ */
    .topbar {
        display: flex; align-items: center; justify-content: space-between;
        padding: 0.6rem 0 0.9rem 0; border-bottom: 1px solid rgba(80,130,255,0.2);
        margin-bottom: 1.4rem;
    }
    .brand { display: flex; align-items: center; gap: 12px; }
    .brand .logo {
        width: 40px; height: 40px; border-radius: 11px;
        background: linear-gradient(135deg, #3b82f6, #06b6d4);
        display: grid; place-items: center; font-size: 20px;
        box-shadow: 0 0 26px rgba(59,130,246,0.45);
        animation: spinGlow 6s linear infinite;
    }
    @keyframes spinGlow { 50% { box-shadow: 0 0 44px rgba(6,182,212,0.6);} }
    .brand .title { font-family:'Sora',sans-serif; font-weight:700; font-size:1.15rem; letter-spacing:-0.3px; }
    .brand .title span { color:#6f8bff; }
    .brand .tag { font-size:0.68rem; color:#7c8db0; letter-spacing:1.5px; text-transform:uppercase; margin-top:1px; }

    .clock { font-family:'JetBrains Mono',monospace; font-size:0.9rem; color:#8ca6ff; }

    /* ------------------ HERO ------------------ */
    .hero { text-align:center; padding: 0.6rem 0 0.4rem 0; }
    .hero .kicker {
        font-family:'JetBrains Mono',monospace; font-size:0.72rem; letter-spacing:4px;
        color:#38c8e8; text-transform:uppercase; margin-bottom:0.5rem;
    }
    .hero h1 {
        font-family:'Sora',sans-serif; font-weight:800; font-size:3.1rem; letter-spacing:-1.5px;
        margin:0; line-height:1.05; color:#f2f6ff;
    }
    .hero h1 .grad {
        background: linear-gradient(120deg,#4f9dff, #22d3ee 55%, #7cc6ff);
        -webkit-background-clip:text; background-clip:text; color:transparent;
    }
    .hero .subtitle { color:#8b9cc0; margin-top:0.6rem; font-size:0.98rem; font-weight:300; }

    /* ------------------ GLOBAL PULSE / ENTRIES ------------------ */
    .entry { animation: rise 0.55s cubic-bezier(.2,.7,.2,1) both; }
    @keyframes rise { from {opacity:0; transform:translateY(16px);} to {opacity:1; transform:none;} }

    /* ------------------ PANELS ------------------ */
    .panel {
        background: linear-gradient(165deg, rgba(16,25,48,0.82), rgba(9,14,28,0.92));
        border: 1px solid rgba(90,140,255,0.16);
        border-radius: 18px;
        padding: 1.05rem 1.15rem;
        box-shadow: 0 10px 34px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.04);
        backdrop-filter: blur(10px);
        position: relative; overflow: hidden;
    }
    .panel::after {
        content:""; position:absolute; top:0; left:0; right:0; height:1px;
        background: linear-gradient(90deg, transparent, rgba(120,180,255,0.5), transparent);
    }
    .panel-title {
        font-family:'Sora',sans-serif; font-weight:600; font-size:0.82rem;
        letter-spacing:1.5px; text-transform:uppercase; color:#93a6ce;
        display:flex; align-items:center; gap:8px; margin-bottom:0.9rem;
    }
    .panel-title .n {
        font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:#22d3ee;
        border:1px solid rgba(34,211,238,0.4); border-radius:6px; padding:1px 6px;
    }

    /* ------------------ STATUS RING + LABEL ------------------ */
    .status-wrap { text-align:center; }
    .ring { width:120px; height:120px; margin:0.2rem auto 0.8rem; position:relative; }
    .ring svg { transform: rotate(-90deg); }
    .ring .bg { stroke: rgba(90,140,255,0.14); }
    .ring .fg { stroke-linecap: round; transition: stroke-dashoffset 1s ease, stroke 0.6s; }
    .ring .ctr { position:absolute; inset:0; display:grid; place-items:center; }
    .ring .ctr .val { font-family:'JetBrains Mono',monospace; font-size:1.5rem; font-weight:600; }
    .status-label {
        display:inline-flex; gap:8px; align-items:center;
        font-family:'Sora',sans-serif; font-weight:700; letter-spacing:1px;
        font-size:0.92rem; text-transform:uppercase; padding:7px 16px; border-radius:999px;
    }
    .lbl-critical { color:#ff6b63; background:rgba(255,86,78,0.12); border:1px solid rgba(255,86,78,0.5); box-shadow:0 0 20px rgba(255,86,78,0.3); animation:blink 1.4s ease-in-out infinite; }
    .lbl-warning  { color:#f4b942; background:rgba(244,185,66,0.12); border:1px solid rgba(244,185,66,0.5); box-shadow:0 0 20px rgba(244,185,66,0.25); animation:blink 1.8s ease-in-out infinite; }
    .lbl-safe     { color:#4ade80; background:rgba(74,222,128,0.10); border:1px solid rgba(74,222,128,0.5); box-shadow:0 0 18px rgba(74,222,128,0.25); }
    @keyframes blink { 0%,100%{opacity:1;} 50%{opacity:0.6;} }

    /* ------------------ ACTION BANNER ------------------ */
    .directive {
        margin-top:1rem; border-radius:14px; padding:0.9rem 1.2rem; font-size:0.94rem;
        display:flex; gap:12px; align-items:flex-start; border:1px solid;
    }
    .directive .ic { font-size:1.3rem; line-height:1.2; }
    .directive b { display:block; margin-bottom:2px; }
    .dir-critical { background:rgba(255,86,78,0.08); border-color:rgba(255,86,78,0.35); color:#ffd6d3; }
    .dir-warning  { background:rgba(244,185,66,0.08); border-color:rgba(244,185,66,0.35); color:#ffe7bd; }
    .dir-safe     { background:rgba(74,222,128,0.06); border-color:rgba(74,222,128,0.35); color:#d3f5dd; }

    /* ------------------ METRIC TILES ------------------ */
    .tile {
        text-align:center; padding:0.9rem 0.6rem; border-radius:14px;
        background:linear-gradient(165deg, rgba(20,30,56,0.9), rgba(10,16,32,0.95));
        border:1px solid rgba(90,140,255,0.14);
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    }
    .tile:hover { transform:translateY(-3px); border-color:rgba(120,180,255,0.4); box-shadow:0 12px 30px rgba(0,0,0,0.4); }
    .tile .t { font-size:0.68rem; letter-spacing:1.5px; text-transform:uppercase; color:#8ca4cf; font-weight:600; }
    .tile .v { font-family:'Sora',sans-serif; font-weight:700; font-size:1.45rem; margin-top:3px; }
    .tile .h { font-size:0.78rem; color:#7c8db0; margin-top:2px; }

    /* ------------------ INPUTS ------------------ */
    [data-testid="stSlider"] [role="slider"] { background:#22d3ee; box-shadow:0 0 10px rgba(34,211,238,0.7); }
    .stButton button {
        background: linear-gradient(135deg, rgba(59,130,246,0.16), rgba(6,182,212,0.10));
        border:1px solid rgba(90,140,255,0.3) !important; color:#e8eefc !important; font-weight:600;
        border-radius:10px !important; transition: all 0.2s ease;
    }
    .stButton button:hover { transform:translateY(-1px); box-shadow:0 0 18px rgba(34,211,238,0.35); border-color:rgba(34,211,238,0.7)!important; }
    .stSelectbox [data-baseweb="select"] > div { background:rgba(16,25,48,0.6) !important; border-radius:10px !important; }
    /* ------------------ FILE UPLOADER (fully themed) ------------------ */
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(16,25,48,0.5) !important;
        border-radius: 12px !important;
        border: 1px dashed rgba(90,140,255,0.35) !important;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: rgba(34,211,238,0.6) !important;
        background: rgba(20,32,60,0.7) !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] {
        color: #b8c6e2 !important;
        font-weight: 500;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] strong,
    [data-testid="stFileUploaderDropzoneInstructions"] span strong {
        color: #22d3ee !important;
        font-weight: 700;
    }
    [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stFileUploaderDropzone"] [kind="secondary"],
    [data-testid="stFileUploaderDropzone"] button[kind="secondary"] {
        background: linear-gradient(135deg, rgba(59,130,246,0.35), rgba(6,182,212,0.25)) !important;
        color: #ffffff !important;
        border: 1px solid rgba(110,170,255,0.5) !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        padding: 2px 12px !important;
        transition: all 0.2s ease;
    }
    [data-testid="stFileUploaderDropzone"] button:hover {
        box-shadow: 0 0 14px rgba(34,211,238,0.4);
        border-color: rgba(34,211,238,0.8) !important;
    }
    [data-testid="stFileUploaderDropzone"] small,
    [data-testid="stFileUploaderDropzone"] div[data-testid="stFileUploaderDropzoneInstructions"] div {
        color: #7c8db0 !important;
    }
    [data-testid="stFileUploaderFile"] {
        background: rgba(20,32,60,0.7) !important;
        border: 1px solid rgba(90,140,255,0.25) !important;
        border-radius: 10px !important;
        color: #e8eefc !important;
    }
    [data-testid="stFileUploaderFile"] [data-testid="stFileUploaderFileTitle"] {
        color: #e8eefc !important;
    }

    /* ------------------ GENERIC TEXT COLORS ------------------ */
    p, span, label, div[data-testid="stMarkdownContainer"], li {
        color: inherit;
    }
    [data-testid="stCaptionContainer"] p,
    [data-testid="stCaptionContainer"] {
        color: #8b9cc0 !important;
    }
    [data-baseweb="select"] * { color: #e8eefc !important; }
    .stSelectbox label, .stSlider label, .stFileUploader label,
    [data-testid="stWidgetLabel"] {
        color: #aebcd8 !important;
        font-weight: 500;
    }

    /* ------------------ DIVIDER ------------------ */
    hr.glow { border:0; height:1px; background:linear-gradient(90deg,transparent,rgba(100,160,255,0.45),transparent); margin:1.1rem 0; }
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
# LIVE CLOCK
# =============================================================================
now = pd.Timestamp.now()
st.markdown(f"""
<div class="topbar">
    <div class="brand">
        <div class="logo">🌊</div>
        <div>
            <div class="title">Aqua<span>Shield</span></div>
            <div class="tag">Disaster Command Center</div>
        </div>
    </div>
    <div class="clock">{now.strftime('%a %d %b · %H:%M:%S')} UTC</div>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# HERO
# =============================================================================
st.markdown("""
<div class="hero entry">
    <div class="kicker">Real-time Flood Intelligence</div>
    <h1><span class="grad">Autonomous Disaster Response</span></h1>
    <div class="subtitle">CNN drone vision · LSTM river forecasting · ML risk triage</div>
</div>
""", unsafe_allow_html=True)
st.markdown('<hr class="glow">', unsafe_allow_html=True)

# =============================================================================
# LAYOUT
# =============================================================================
col_input, col_result = st.columns([1, 1.25], gap="large")

# -----------------------------------------------------------------------------
# LEFT COLUMN: INPUTS
# -----------------------------------------------------------------------------
with col_input:
    # --- Drone Input ---
    st.markdown('<div class="panel entry"><div class="panel-title"><span class="n">01</span> Aerial Feed</div>', unsafe_allow_html=True)

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("🌊 Sample Flooded", use_container_width=True, key="btn_f"):
            if os.path.exists(SAMPLE_FLOOD):
                st.session_state["selected_img"] = Image.open(SAMPLE_FLOOD).convert("RGB")
                st.session_state["img_name"] = "Sample Flooded Drone Shot"
    with btn_col2:
        if st.button("☀️ Sample Clear", use_container_width=True, key="btn_c"):
            if os.path.exists(SAMPLE_CLEAR):
                st.session_state["selected_img"] = Image.open(SAMPLE_CLEAR).convert("RGB")
                st.session_state["img_name"] = "Sample Clear Drone Shot"

    uploaded = st.file_uploader("Or upload a drone image", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        st.session_state["selected_img"] = Image.open(uploaded).convert("RGB")
        st.session_state["img_name"] = uploaded.name

    if st.session_state["selected_img"] is not None:
        st.image(st.session_state["selected_img"], caption=st.session_state["img_name"], use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # --- Conditions Input ---
    st.markdown('<div class="panel entry" style="margin-top:1.1rem;"><div class="panel-title"><span class="n">02</span> Hydrological Conditions</div>', unsafe_allow_html=True)

    zone_key = st.selectbox("Zone", list(ZONES.keys()), format_func=lambda z: ZONES[z]["label"])
    river_level = st.slider("River Level (m)", 0.0, 15.0, 4.2, 0.1)
    rainfall = st.slider("Rainfall (mm/hr)", 0.0, 150.0, 40.0, 1.0)
    emergency_calls = st.slider("Emergency Calls / hr", 0, 500, 60)

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# RIGHT COLUMN: RESULTS
# -----------------------------------------------------------------------------
with col_result:
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
    # The LSTM is trained in the raw master-domain units, so its absolute output
    # (a near-constant ~32) is not meaningful in metres. We anchor the 12h forecast
    # to the user's gauge reading (metres) and use the LSTM as a trend signal:
    # heavier rainfall / rising calls push the projected level up, so the forecast
    # responds to the live conditions and classification changes with the sliders.
    predicted_river_12h = river_level

    # Trend drivers (responsive to inputs)
    rain_push = 0.55 if rainfall >= 50.0 else (0.25 if rainfall >= 35.0 else 0.05)
    call_push = 0.30 if emergency_calls >= 100 else (0.10 if emergency_calls >= 60 else 0.02)
    zone_factor = 1.0 + ZONES[zone_key]["prob"] - 0.4  # riskier basins amplify
    trend = (rain_push + call_push) * zone_factor
    predicted_river_12h = river_level + trend

    # Optionally blend in the LSTM's raw model output direction (metadata only)
    if MODELS['lstm'] is not None and MODELS['scaler'] is not None:
        try:
            seq_river = np.linspace(max(0.5, river_level - 1.2), river_level, 48)
            seq_rain = np.linspace(max(0, rainfall - 20), rainfall, 48)
            seq_calls = np.linspace(emergency_calls * 0.7, emergency_calls, 48)
            seq_closures = np.full(48, min(river_level // 1.5, 10))

            raw_seq = np.column_stack([seq_river, seq_rain, seq_calls, seq_closures])
            scaled_seq = MODELS['scaler'].transform(raw_seq)
            t_in = torch.tensor(scaled_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                pred_scaled = MODELS['lstm'](t_in).cpu().numpy()[0, 0]
            # Model's relative push in metre scale (clamped to a sane band)
            last_scaled = scaled_seq[-1, 0]
            model_delta_m = (pred_scaled - last_scaled) * MODELS['scaler'].scale_[0]
            model_delta_m = float(np.clip(model_delta_m, -0.6, 0.6))
            predicted_river_12h = river_level + np.clip(trend + model_delta_m * 0.3, -0.5, 2.5)
        except Exception:
            predicted_river_12h = river_level + np.clip(trend, -0.5, 2.5)

    # --- 3. COMPUTE FINAL RESULT CLASS (responsive to live inputs) ---
    is_flood_img = (vision_label == "FLOODED")
    is_river_high = (river_level >= 5.0 or predicted_river_12h >= 5.5)
    is_rain_heavy = (rainfall >= 50.0)
    is_calls_surge = (emergency_calls >= 100)

    if (is_flood_img and is_river_high) or river_level >= 7.0 or predicted_river_12h >= 7.0:
        final_class = "CRITICAL FLOOD EMERGENCY"
        status_color = "#ff6b63"
        ring_color = "#ff6b63"
        lbl_class = "lbl-critical"
        dir_class = "dir-critical"
        advice = "Immediate evacuation of low-lying areas required. Dispatch rescue teams and close bridges."
        risk_pct = 92
    elif is_flood_img or is_river_high or is_rain_heavy or is_calls_surge:
        final_class = "MODERATE FLOOD WARNING"
        status_color = "#f4b942"
        ring_color = "#f4b942"
        lbl_class = "lbl-warning"
        dir_class = "dir-warning"
        advice = "Water accumulation observed. Monitor river banks, stage emergency pumps, and divert traffic."
        risk_pct = 62
    else:
        final_class = "NORMAL / SAFE"
        status_color = "#4ade80"
        ring_color = "#4ade80"
        lbl_class = "lbl-safe"
        dir_class = "dir-safe"
        advice = "Normal operational status. No active flood emergency detected."
        risk_pct = 18

    # --- STATUS PANEL with animated risk ring ---
    st.markdown('<div class="panel entry"><div class="panel-title"><span class="n">03</span> Threat Assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="status-wrap">', unsafe_allow_html=True)

    # Animated SVG risk ring
    CIRC = 2 * 3.14159 * 50  # r=50
    st.markdown(f"""
    <div class="ring">
        <svg width="120" height="120" viewBox="0 0 120 120">
            <circle class="bg" cx="60" cy="60" r="50" fill="none" stroke-width="9"/>
            <circle class="fg" cx="60" cy="60" r="50" fill="none" stroke-width="9"
                stroke="{ring_color}" stroke-dasharray="{CIRC}" stroke-dashoffset="{CIRC*(1-risk_pct/100)}"
                filter="drop-shadow(0 0 6px {ring_color})"/>
        </svg>
        <div class="ctr"><div class="val" style="color:{ring_color};">{risk_pct}%</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<span class="status-label {lbl_class}">◉ {final_class}</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="directive {dir_class}">
        <div class="ic">🚨</div>
        <div><b>{final_class}</b>{advice}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # --- TELEMETRY TILES ---
    st.markdown('<div class="panel entry" style="margin-top:1.1rem;"><div class="panel-title"><span class="n">04</span> Signal Telemetry</div>', unsafe_allow_html=True)

    t1, t2, t3 = st.columns(3, gap="small")

    with t1:
        st.markdown('<div class="tile"><div class="t">Drone · CNN</div>', unsafe_allow_html=True)
        if vision_label == "FLOODED":
            st.markdown(f'<div class="v" style="color:#ff6b63;">FLOODED</div><div class="h">{vision_conf:.1f}% conf</div>', unsafe_allow_html=True)
        elif vision_label == "CLEAR":
            st.markdown(f'<div class="v" style="color:#4ade80;">CLEAR</div><div class="h">{vision_conf:.1f}% conf</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="v" style="color:#7c8db0;">—</div><div class="h">no feed</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with t2:
        st.markdown('<div class="tile"><div class="t">River Now</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="v">{river_level:.1f}<span style="font-size:0.9rem;color:#7c8db0;">m</span></div><div class="h">gauge</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with t3:
        st.markdown('<div class="tile"><div class="t">River +12h</div>', unsafe_allow_html=True)
        delta = predicted_river_12h - river_level
        trend_color = "#ff6b63" if delta > 0 else "#4ade80"
        st.markdown(f'<div class="v">{predicted_river_12h:.1f}<span style="font-size:0.9rem;color:#7c8db0;">m</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="h" style="color:{trend_color};">{"▲ roaring" if delta>0 else "▼ falling"} {abs(delta):+.2f}m</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- FORECAST CHART ---
    st.markdown('<div style="margin-top:1.1rem;"><b style="color:#93a6ce;font-size:0.78rem;letter-spacing:1.5px;text-transform:uppercase;">River Projection — 12h</b></div>', unsafe_allow_html=True)
    timeline = ["Now", "+3h", "+6h", "+9h", "+12h"]
    proj_vals = np.linspace(river_level, predicted_river_12h, 5)
    chart_df = pd.DataFrame({
        "Timeline": timeline,
        "Forecast": proj_vals,
        "Danger 5.0m": [5.0]*5
    }).set_index("Timeline")
    st.line_chart(chart_df, color=["#22d3ee", "#ff6b63"], height=200, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)
