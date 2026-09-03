import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Disaster Response AI API", description="Predicts disaster risk level for city zones.")

# Load the trained ML model
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(base_dir, "models", "risk_model.joblib")

model = None
expected_features = None

@app.on_event("startup")
def load_model():
    global model
    global expected_features
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
        # Infer expected features from the pipeline scaler or classifier
        expected_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
    except Exception as e:
        print(f"Failed to load model: {e}")

class RiskPredictionRequest(BaseModel):
    river_level: float
    rainfall: float
    emergency_call_volume: float
    road_closures: float
    bridge_closures: float
    historical_flood_probability: float
    river_level_rolling_72h_avg: float
    rainfall_rolling_72h_sum: float
    emergency_calls_24h_sum: float
    total_infrastructure_closures: float
    river_level_trend: float
    zone_id_Zone_B: bool = False
    zone_id_Zone_C: bool = False
    zone_id_Zone_D: bool = False

@app.post("/predict")
def predict_risk(request: RiskPredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
        
    # Convert request to dataframe in the exact order required
    input_data = pd.DataFrame([request.dict()])
    
    # Predict
    try:
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        classes = model.classes_
        
        prob_dict = {classes[i]: float(probabilities[i]) for i in range(len(classes))}
        
        return {
            "status": "success",
            "predicted_risk": prediction,
            "probabilities": prob_dict
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    print("Starting API on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
