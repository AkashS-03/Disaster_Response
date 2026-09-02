import os
import pandas as pd
import numpy as np
from data_engineer.data_ingestion import (
    generate_synthetic_data, load_river_rainfall, load_emergency_calls, 
    load_infrastructure_records, load_historical_flood_logs
)
from data_engineer.data_cleaning import (
    clean_river_rainfall, clean_emergency_calls, 
    clean_infrastructure, clean_historical_logs
)
from data_engineer.data_validation import validate_master_dataset

def build_master_dataset():
    """
    Main pipeline for the Data Engineer.
    Ingests raw data, cleans it, merges it, computes 72-hour rolling features,
    assigns synthetic risk labels, validates, and saves to data/processed/.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_dir = os.path.join(base_dir, "data", "raw")
    processed_dir = os.path.join(base_dir, "data", "processed")
    
    # 1. Ingest
    if not os.path.exists(os.path.join(raw_dir, "river_rainfall_data.csv")):
        generate_synthetic_data(raw_dir)
        
    df_riv_rain = load_river_rainfall(raw_dir)
    df_calls = load_emergency_calls(raw_dir)
    df_infra = load_infrastructure_records(raw_dir)
    df_hist = load_historical_flood_logs(raw_dir)
    
    # 2. Clean
    print("Cleaning datasets...")
    df_riv_rain_c = clean_river_rainfall(df_riv_rain)
    df_calls_c = clean_emergency_calls(df_calls)
    df_infra_c = clean_infrastructure(df_infra)
    df_hist_c = clean_historical_logs(df_hist)
    
    # 3. Merge temporal datasets
    print("Merging temporal datasets...")
    # Merge on zone_id and timestamp
    df_master = pd.merge(df_riv_rain_c, df_calls_c, on=["zone_id", "timestamp"], how="left")
    df_master = pd.merge(df_master, df_infra_c, on=["zone_id", "timestamp"], how="left")
    
    # Merge historical data based on rounded river level bucket
    # We ignore NaN river_levels for this merge.
    df_master['river_level_bucket'] = df_master['river_level'].round(0)
    df_master = pd.merge(df_master, df_hist_c, on=["zone_id", "river_level_bucket"], how="left")
    df_master = df_master.drop(columns=['river_level_bucket'])
    
    # 4. Feature Engineering (72-hour rolling windows as requested)
    print("Computing 72-hour rolling features...")
    df_master = df_master.sort_values(by=["zone_id", "timestamp"]).reset_index(drop=True)
    
    # Set timestamp as index for rolling calculations
    df_master_idx = df_master.set_index("timestamp")
    
    rolling_features = []
    for zone, group in df_master_idx.groupby("zone_id"):
        # 72H rolling average river level
        riv_72h = group["river_level"].rolling("72h", min_periods=1).mean()
        # 72H rolling sum of rainfall
        rain_72h = group["rainfall"].rolling("72h", min_periods=1).sum()
        # 24H rolling change in emergency calls
        calls_24h_sum = group["emergency_call_volume"].rolling("24h", min_periods=1).sum()
        
        group["river_level_rolling_72h_avg"] = riv_72h.values
        group["rainfall_rolling_72h_sum"] = rain_72h.values
        group["emergency_calls_24h_sum"] = calls_24h_sum.values
        
        # Additional derived metrics (these might become NaN if operands are NaN)
        group["total_infrastructure_closures"] = group["road_closures"] + group["bridge_closures"]
        group["river_level_trend"] = group["river_level"] - group["river_level"].shift(1)
        
        rolling_features.append(group.reset_index())
        
    df_master = pd.concat(rolling_features, ignore_index=True)
    
    # 5. Assign synthetic Risk Label (LOW, MODERATE, SEVERE)
    # The specification requires defining explicit numerical rules.
    print("Assigning risk labels...")
    def assign_risk(row):
        riv = row["river_level"]
        rain_72 = row["rainfall_rolling_72h_sum"]
        calls = row["emergency_calls_24h_sum"]
        
        # If the key fields are NaN, we still calculate based on whatever is available, 
        # or it naturally falls through to LOW. We will let pandas handle NaN comparisons (returns False).
        
        # Definition of SEVERE:
        if riv >= 4.5 or (rain_72 >= 150.0 and riv >= 3.5):
            return "SEVERE"
        # Definition of MODERATE:
        elif riv >= 3.0 or rain_72 >= 80.0 or calls >= 100:
            return "MODERATE"
        # Otherwise LOW
        else:
            return "LOW"
            
    df_master["risk_label"] = df_master.apply(assign_risk, axis=1)
    
    # 6. Validate
    print("Validating master dataset...")
    if validate_master_dataset(df_master):
        os.makedirs(processed_dir, exist_ok=True)
        out_path = os.path.join(processed_dir, "master_dataset.csv")
        df_master.to_csv(out_path, index=False)
        print(f"Master dataset successfully built and saved to: {out_path}")
    else:
        print("Master dataset build failed validation.")

if __name__ == "__main__":
    build_master_dataset()
