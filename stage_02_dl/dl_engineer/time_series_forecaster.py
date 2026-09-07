import os
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Configuration ---
EPOCHS = 15
BATCH_SIZE = 64
LR = 0.001
HIDDEN_SIZE = 64
NUM_LAYERS = 2

class FloodLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(FloodLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] # Take the last timestep's output
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def inverse_transform_river(preds, scaler, feature_idx=0):
    # Create dummy array matching scaler shape
    dummy = np.zeros((len(preds), scaler.n_features_in_))
    dummy[:, feature_idx] = preds.flatten()
    return scaler.inverse_transform(dummy)[:, feature_idx]

def run_baselines_and_lstm():
    print("--- Loading Data ---")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data", "time_series")
    
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    scaler = joblib.load(os.path.join(data_dir, "ts_scaler.joblib"))
    
    y_test_true = inverse_transform_river(y_test, scaler)
    
    # ---------------------------------------------------------
    # 1. NAIVE PERSISTENCE BASELINE
    # Predict that river level in 12 hours is exactly what it is right now
    print("\n--- 1. Evaluating Naive Persistence Baseline ---")
    current_river_level_scaled = X_test[:, -1, 0] # Index 0 is river_level, -1 is the latest timestep
    naive_preds = inverse_transform_river(current_river_level_scaled, scaler)
    
    naive_mae = mean_absolute_error(y_test_true, naive_preds)
    naive_mse = mean_squared_error(y_test_true, naive_preds)
    print(f"Naive MAE: {naive_mae:.4f}m | MSE: {naive_mse:.4f}")
    
    # ---------------------------------------------------------
    # 2. XGBOOST LAG BASELINE
    print("\n--- 2. Training XGBoost Lag Baseline ---")
    # Flatten the sequence window into a 1D feature vector for XGBoost
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train_flat, y_train)
    
    xgb_preds_scaled = xgb_model.predict(X_test_flat)
    xgb_preds = inverse_transform_river(xgb_preds_scaled, scaler)
    
    xgb_mae = mean_absolute_error(y_test_true, xgb_preds)
    xgb_mse = mean_squared_error(y_test_true, xgb_preds)
    print(f"XGBoost MAE: {xgb_mae:.4f}m | MSE: {xgb_mse:.4f}")
    
    joblib.dump(xgb_model, os.path.join(base_dir, "models", "xgb_baseline.joblib"))
    
    # ---------------------------------------------------------
    # 3. PYTORCH LSTM
    print("\n--- 3. Training PyTorch LSTM ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32).unsqueeze(1))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    num_features = X_train.shape[2]
    model = FloodLSTM(input_size=num_features, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = model(batch_X)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{EPOCHS} | Loss: {train_loss/len(train_loader):.4f}")
            
    # Evaluate LSTM
    model.eval()
    lstm_preds_scaled = []
    with torch.no_grad():
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(device)
            out = model(batch_X)
            lstm_preds_scaled.extend(out.cpu().numpy())
            
    lstm_preds_scaled = np.array(lstm_preds_scaled)
    lstm_preds = inverse_transform_river(lstm_preds_scaled, scaler)
    
    lstm_mae = mean_absolute_error(y_test_true, lstm_preds)
    lstm_mse = mean_squared_error(y_test_true, lstm_preds)
    print(f"LSTM MAE: {lstm_mae:.4f}m | MSE: {lstm_mse:.4f}")
    
    torch.save(model.state_dict(), os.path.join(base_dir, "models", "lstm_forecaster.pth"))
    
    print("\n--- SUMMARY REPORT ---")
    print(f"1. Naive Persistence MAE: {naive_mae:.4f}m")
    print(f"2. XGBoost Baseline MAE:  {xgb_mae:.4f}m")
    print(f"3. LSTM Forecaster MAE:   {lstm_mae:.4f}m")
    
    # Save report
    with open(os.path.join(base_dir, "reports", "forecasting_report.md"), "w") as f:
        f.write(f"# Time-Series Forecasting Evaluation\n\n")
        f.write(f"Target: Predict `river_level` 12 hours into the future using 48-hour trailing windows.\n")
        f.write(f"Chronological Split enforced. Test Set: Nov 2024 - Dec 2024.\n\n")
        f.write(f"## Metrics (Mean Absolute Error in meters)\n")
        f.write(f"- **Naive Persistence:** {naive_mae:.4f}m\n")
        f.write(f"- **XGBoost Lag Baseline:** {xgb_mae:.4f}m\n")
        f.write(f"- **PyTorch LSTM:** {lstm_mae:.4f}m\n")
        
if __name__ == "__main__":
    run_baselines_and_lstm()
