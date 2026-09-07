import streamlit as st
import requests
import json
import numpy as np

API_URL = "http://127.0.0.1:8001"

st.set_page_config(page_title="Deep Learning Command Center", layout="wide")

st.title("Stage 2: Deep Learning Command Center")
st.markdown("Multi-modal Artificial Intelligence Integration for Disaster Response")

tab1, tab2 = st.tabs(["📊 Time-Series Forecast (LSTM)", "📸 Drone Camera (Vision)"])

with tab1:
    st.header("River Level Forecaster")
    st.write("Predicts the river level 12 hours into the future based on 48-hour trailing telemetry.")
    
    if st.button("Generate 48h Mock Sequence & Forecast"):
        mock_seq = []
        base_river = 4.0
        for i in range(48):
            mock_seq.append([
                base_river + (i * 0.05),
                20.0 + np.random.normal(0, 5),
                100 + i*2,
                3
            ])
            
        payload = {"sequence": mock_seq}
        try:
            res = requests.post(f"{API_URL}/dl/forecast", json=payload)
            if res.status_code == 200:
                data = res.json()
                st.success(f"**Predicted River Level (in 12 hours):** {data['predicted_river_level_12h']:.2f} meters")
                st.line_chart([x[0] for x in mock_seq] + [data['predicted_river_level_12h']])
            else:
                st.error("API Error")
        except Exception as e:
            st.error(f"Failed to connect to API: {e}")

with tab2:
    st.header("UAV/Drone Feed Classifier (CNN)")
    st.write("Analyzes live images to detect urban flooding using fine-tuned MobileNetV2.")
    
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
                st.error(f"Failed to connect to API: {e}")
