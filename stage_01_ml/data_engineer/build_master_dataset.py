import os
import pandas as pd
import numpy as np

def build_master_dataset():
    print("--- Building Master Dataset from Real API Data ---")
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_dir = os.path.join(base_dir, "data", "raw")
    processed_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    # 1. Load the raw data fetched by data_ingestion.py
    print("Loading raw API datasets...")
    weather = pd.read_csv(os.path.join(raw_dir, "weather_data.csv"))
    river = pd.read_csv(os.path.join(raw_dir, "river_data.csv"))
    calls = pd.read_csv(os.path.join(raw_dir, "emergency_calls.csv"))
    infra = pd.read_csv(os.path.join(raw_dir, "infrastructure_status.csv"))
    
    # Ensure timestamps are parsed
    for df in [weather, river, calls, infra]:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 2. Merge Data on timestamp and zone_id
    print("Merging datasets...")
    master_df = river.merge(weather, on=['timestamp', 'zone_id'], how='left')
    master_df = master_df.merge(calls, on=['timestamp', 'zone_id'], how='left')
    master_df = master_df.merge(infra, on=['timestamp', 'zone_id'], how='left')
    
    # Sort chronologically per zone
    master_df = master_df.sort_values(by=['zone_id', 'timestamp']).reset_index(drop=True)
    
    # 3. Calculate 72-Hour Rolling Windows
    print("Calculating rolling features...")
    # Because data is hourly, 72 hours = 72 rows
    master_df['river_level_rolling_72h_avg'] = master_df.groupby('zone_id')['river_level'].rolling(window=72, min_periods=1).mean().reset_index(level=0, drop=True)
    master_df['rainfall_rolling_72h_sum'] = master_df.groupby('zone_id')['rainfall'].rolling(window=72, min_periods=1).sum().reset_index(level=0, drop=True)
    master_df['emergency_calls_24h_sum'] = master_df.groupby('zone_id')['emergency_call_volume'].rolling(window=24, min_periods=1).sum().reset_index(level=0, drop=True)
    
    # 4. Total Closures & Trends
    master_df['total_infrastructure_closures'] = master_df['road_closures'] + master_df['bridge_closures']
    master_df['river_level_trend'] = master_df.groupby('zone_id')['river_level'].diff(1).fillna(0)
    
    # 5. Assign Risk Label (The target variable)
    # We maintain our deterministic logic based on the raw telemetry
    print("Assigning risk labels...")
    def assign_risk(row):
        riv = row['river_level']
        rain_72 = row['rainfall_rolling_72h_sum']
        calls = row['emergency_call_volume']
        if riv >= 4.5 or (rain_72 >= 150.0 and riv >= 3.5):
            return "SEVERE"
        elif riv >= 3.0 or rain_72 >= 80.0 or calls >= 100:
            return "MODERATE"
        else:
            return "LOW"
            
    master_df['risk_label'] = master_df.apply(assign_risk, axis=1)
    
    # 6. Save the final Master Dataset
    out_path = os.path.join(processed_dir, "master_dataset.csv")
    master_df.to_csv(out_path, index=False)
    print(f"Master Dataset built successfully! Rows: {master_df.shape[0]}, Saved to {out_path}")

if __name__ == "__main__":
    build_master_dataset()
