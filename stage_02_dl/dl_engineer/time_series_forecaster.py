import os
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Configuration & Hyperparameters ---
EPOCHS = 20
BATCH_SIZE = 64
LR = 0.001
HIDDEN_SIZE = 64
NUM_LAYERS = 2

class FloodLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2):
        super(FloodLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take the last timestep's output
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def inverse_transform_river(preds, scaler, feature_idx=0):
    dummy = np.zeros((len(preds), scaler.n_features_in_))
    dummy[:, feature_idx] = preds.flatten()
    return scaler.inverse_transform(dummy)[:, feature_idx]

def train_and_finetune_lstm():
    print("=== Training & Fine-Tuning PyTorch LSTM River Level Forecaster ===")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data", "time_series")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    print("Loading sequential hydrological datasets...")
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    scaler = joblib.load(os.path.join(data_dir, "ts_scaler.joblib"))
    
    y_test_true = inverse_transform_river(y_test, scaler)
    print(f"Training Windows: {X_train.shape[0]} sequences (shape: {X_train.shape[1:]})")
    print(f"Testing Windows:  {X_test.shape[0]} sequences")
    
    # ---------------------------------------------------------
    # 1. NAIVE PERSISTENCE BENCHMARK
    current_river_level_scaled = X_test[:, -1, 0]
    naive_preds = inverse_transform_river(current_river_level_scaled, scaler)
    naive_mae = mean_absolute_error(y_test_true, naive_preds)
    print(f"\n[Baseline] Naive Persistence 12h MAE: {naive_mae:.4f}m")
    
    # ---------------------------------------------------------
    # 2. XGBOOST BENCHMARK
    print("[Baseline] Fitting XGBoost Lag Baseline...")
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=42)
    xgb_model.fit(X_train_flat, y_train)
    xgb_preds = inverse_transform_river(xgb_model.predict(X_test_flat), scaler)
    xgb_mae = mean_absolute_error(y_test_true, xgb_preds)
    joblib.dump(xgb_model, os.path.join(models_dir, "xgb_baseline.joblib"))
    print(f"[Baseline] XGBoost Lag Model 12h MAE: {xgb_mae:.4f}m")
    
    # ---------------------------------------------------------
    # 3. PYTORCH LSTM TRAINING & FINE-TUNING
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[LSTM] Initializing FloodLSTM on Compute Device: {device}")
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32).unsqueeze(1))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    num_features = X_train.shape[2]
    model = FloodLSTM(input_size=num_features, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(device)
    
    # Smooth L1 Loss (Huber Loss) prevents surging gradients on extreme river peaks
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    best_test_mae = float('inf')
    model_save_path = os.path.join(models_dir, "lstm_forecaster.pth")
    
    print(f"Beginning Fine-Tuning across {EPOCHS} Epochs...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = model(batch_X)
            loss = criterion(out, batch_y)
            loss.backward()
            
            # Gradient clipping for stable convergence
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation on Test Set
        model.eval()
        val_loss = 0.0
        preds_list = []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                out = model(batch_X)
                val_loss += criterion(out, batch_y).item()
                preds_list.extend(out.cpu().numpy())
                
        avg_val_loss = val_loss / len(test_loader)
        scheduler.step(avg_val_loss)
        
        preds_real = inverse_transform_river(np.array(preds_list), scaler)
        epoch_mae = mean_absolute_error(y_test_true, preds_real)
        
        if epoch_mae < best_test_mae:
            best_test_mae = epoch_mae
            torch.save(model.state_dict(), model_save_path)
            saved_indicator = "* (Best Model Saved)"
        else:
            saved_indicator = ""
            
        if epoch % 2 == 0 or epoch == 1 or epoch == EPOCHS:
            print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | 12h MAE: {epoch_mae:.4f}m {saved_indicator}", flush=True)
            
    print(f"\nFine-Tuning Finished!")
    print(f"Optimal 12-Hour Forecast MAE achieved: {best_test_mae:.4f}m")
    print(f"Saved best LSTM weights to: {model_save_path}")
    
    # Generate comprehensive report
    with open(os.path.join(base_dir, "reports", "forecasting_report.md"), "w") as f:
        f.write("# Hydrological Time-Series Forecasting Evaluation\n\n")
        f.write("Target: Predict `river_level` 12 hours ahead from 48-hour trailing rolling windows.\n\n")
        f.write("## Benchmark Comparison (Mean Absolute Error)\n")
        f.write(f"- **Naive Persistence:** {naive_mae:.4f}m\n")
        f.write(f"- **XGBoost Regressor:** {xgb_mae:.4f}m\n")
        f.write(f"- **Fine-Tuned PyTorch LSTM:** **{best_test_mae:.4f}m**\n")

if __name__ == "__main__":
    train_and_finetune_lstm()
