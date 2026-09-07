import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

def prepare_time_series():
    print("--- Starting Time-Series Data Prep ---")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ml_data_path = os.path.join(base_dir, "..", "stage_01_ml", "data", "processed", "master_dataset.csv")
    out_dir = os.path.join(base_dir, "data", "time_series")
    os.makedirs(out_dir, exist_ok=True)
    
    df = pd.read_csv(ml_data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[df['timestamp'] < '2024-11-01']
    df = df.sort_values(['zone_id', 'timestamp']).reset_index(drop=True)
    
    # Configuration
    SEQ_LEN = 48  # Use past 48 hours
    HORIZON = 12  # Predict 12 hours ahead
    
    features = ['river_level', 'rainfall', 'emergency_call_volume', 'total_infrastructure_closures']
    target_col = 'river_level'
    target_idx = features.index(target_col)
    
    # Chronological Split: Enforce a gap to guarantee ABSOLUTELY NO OVERLAP in sliding windows
    TRAIN_END = pd.to_datetime('2024-08-15')
    TEST_START = pd.to_datetime('2024-09-01')
    
    X_train, y_train, timestamps_train_X = [], [], []
    X_test, y_test, timestamps_test_X = [], [], []
    
    zones = df['zone_id'].unique()
    
    # 1. Fit scaler strictly on Train data to prevent future data leakage
    train_df = df[df['timestamp'] <= TRAIN_END]
    scaler = StandardScaler()
    scaler.fit(train_df[features])
    joblib.dump(scaler, os.path.join(out_dir, "ts_scaler.joblib"))
    
    # Scale entire dataframe (safe now that scaler only knows Train distribution)
    df[features] = scaler.transform(df[features])
    
    for zone in zones:
        zone_df = df[df['zone_id'] == zone].reset_index(drop=True)
        vals = zone_df[features].values
        times = zone_df['timestamp'].values
        
        for i in range(len(zone_df) - SEQ_LEN - HORIZON):
            x_window = vals[i : i + SEQ_LEN]
            x_times = times[i : i + SEQ_LEN]
            
            y_val = vals[i + SEQ_LEN + HORIZON - 1, target_idx]
            y_time = times[i + SEQ_LEN + HORIZON - 1]
            
            if y_time <= np.datetime64(TRAIN_END):
                X_train.append(x_window)
                y_train.append(y_val)
                timestamps_train_X.append(x_times)
            elif x_times[0] >= np.datetime64(TEST_START):
                X_test.append(x_window)
                y_test.append(y_val)
                timestamps_test_X.append(x_times)

    # STRICT ASSERTION: No test window overlaps training timestamps
    max_train_time = np.max([np.max(times) for times in timestamps_train_X])
    min_test_time = np.min([np.min(times) for times in timestamps_test_X])
    
    assert max_train_time < min_test_time, f"LEAKAGE DETECTED! Max train {max_train_time} overlaps Min test {min_test_time}"
    print(f"Strict Assertion Passed: Gap established between train ({max_train_time}) and test ({min_test_time})")
    
    np.save(os.path.join(out_dir, "X_train.npy"), np.array(X_train))
    np.save(os.path.join(out_dir, "y_train.npy"), np.array(y_train))
    np.save(os.path.join(out_dir, "X_test.npy"), np.array(X_test))
    np.save(os.path.join(out_dir, "y_test.npy"), np.array(y_test))
    
    print(f"Time-Series Prep Complete. Saved to {out_dir}")
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

if __name__ == "__main__":
    prepare_time_series()
