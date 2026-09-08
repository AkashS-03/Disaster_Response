import os
import pandas as pd
import numpy as np

def analyze_dl_datasets():
    print("--- EDA Engineer (Deep Learning) ---")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # 1. Time-Series EDA
    ts_path = os.path.join(base_dir, "data", "time_series", "X_train.npy")
    ts_test_path = os.path.join(base_dir, "data", "time_series", "X_test.npy")
    if os.path.exists(ts_path):
        x_train = np.load(ts_path)
        x_test = np.load(ts_test_path)
        print(f"\n[Time-Series] Train Windows (Sequences, Timesteps, Features): {x_train.shape}")
        print(f"[Time-Series] Test Windows (Sequences, Timesteps, Features): {x_test.shape}")
        
    # 2. Vision EDA
    vision_flood = os.path.join(base_dir, "data", "vision", "flooded")
    vision_clear = os.path.join(base_dir, "data", "vision", "clear")
    if os.path.exists(vision_flood) and os.path.exists(vision_clear):
        print(f"\n[Vision] Dataset Balance:")
        print(f" - Flooded Images: {len(os.listdir(vision_flood))}")
        print(f" - Clear Images: {len(os.listdir(vision_clear))}")
        
if __name__ == "__main__":
    analyze_dl_datasets()
