import os
import torch
import joblib
import numpy as np
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List

import sys
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_dir)

from dl_engineer.time_series_forecaster import FloodLSTM, inverse_transform_river
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import io

app = FastAPI(title="Disaster Response DL API (Vision & Hydrological Forecaster)")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load LSTM Forecaster ---
lstm_model = FloodLSTM(input_size=4, hidden_size=64, num_layers=2).to(device)
lstm_model.load_state_dict(torch.load(os.path.join(base_dir, "models", "lstm_forecaster.pth"), map_location=device, weights_only=True))
lstm_model.eval()
ts_scaler = joblib.load(os.path.join(base_dir, "data", "time_series", "ts_scaler.joblib"))

# --- Load Drone Vision CNN (MobileNetV2) ---
vision_model = models.mobilenet_v2()
vision_model.classifier[1] = nn.Linear(vision_model.last_channel, 2)
vision_model.load_state_dict(torch.load(os.path.join(base_dir, "models", "vision_classifier.pth"), map_location=device, weights_only=True))
vision_model.to(device)
vision_model.eval()

vision_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ForecastRequest(BaseModel):
    sequence: List[List[float]] 

@app.get("/health")
def health():
    return {"status": "DL Backend Active (Vision & LSTM only)"}

@app.post("/dl/forecast")
def forecast_river_level(req: ForecastRequest):
    seq = np.array(req.sequence)  # shape (48, 4)
    if seq.shape != (48, 4):
        return {"error": f"Expected sequence shape (48, 4), got {seq.shape}"}
    
    seq_scaled = ts_scaler.transform(seq)
    seq_tensor = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out = lstm_model(seq_tensor)
    
    pred_scaled = out.cpu().numpy()
    dummy = np.zeros((1, 4))
    dummy[0, 0] = pred_scaled[0, 0]
    pred_real = ts_scaler.inverse_transform(dummy)[0, 0]
    
    return {"predicted_river_level_12h": float(pred_real)}

@app.post("/dl/vision")
async def analyze_vision(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = vision_transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out = vision_model(tensor)
        pred = torch.argmax(out, dim=1).item()
        prob = torch.nn.functional.softmax(out, dim=1)[0][pred].item()
        
    mapping = {0: "CLEAR", 1: "FLOODED"}
    return {"classification": mapping[pred], "confidence": float(prob)}
