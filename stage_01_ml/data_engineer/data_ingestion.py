import os
import pandas as pd
import numpy as np

def generate_synthetic_data(raw_dir: str):
    """
    Generates synthetic realistic data for the disaster response ML system
    and saves them to the data/raw directory.
    """
    os.makedirs(raw_dir, exist_ok=True)
    np.random.seed(42)
    
    zones = ["Zone_A", "Zone_B", "Zone_C", "Zone_D"]
    # Generate 52 days of hourly data (52 * 24 * 4 = 4992 rows)
    num_hours = 52 * 24 
    start_date = pd.Timestamp("2026-06-01 00:00:00")
    
    # 1. River & Rainfall Gauges
    river_rainfall_records = []
    # 3. Emergency Calls
    emergency_calls_records = []
    # 4. Infrastructure Records
    infrastructure_records = []
    
    for zone in zones:
        timestamps = pd.date_range(start=start_date, periods=num_hours, freq="h")
        
        # Base realistic physical relationships:
        # Rainfall causes river levels to rise over time.
        # High river levels and high rainfall cause more emergency calls and road closures.
        
        base_river = np.random.uniform(1.0, 2.5)
        current_river = base_river
        
        for t in range(num_hours):
            dt = timestamps[t]
            
            # Simulate storm events
            if 100 <= t <= 140:
                rainfall = np.random.uniform(10.0, 50.0)
            else:
                rainfall = np.random.exponential(scale=2.0)
                
            # River level rises with heavy rain
            drainage = (current_river - base_river) * 0.1
            current_river = max(1.0, current_river + (rainfall * 0.05) - drainage + np.random.normal(0, 0.1))
            
            # Inject missing values (10% chance) and extreme outliers (5% chance) for EDA Engineer to solve
            if np.random.rand() < 0.10:
                r_val = np.nan
            elif np.random.rand() < 0.05:
                r_val = -999.0  # Impossible outlier
            else:
                r_val = round(rainfall, 2)
                
            if np.random.rand() < 0.10:
                current_river_val = np.nan
            elif np.random.rand() < 0.05:
                current_river_val = 5000.0 # Impossible river height
            else:
                current_river_val = round(current_river, 2)
                
            # Create messy zone string
            messy_zone = zone
            if np.random.rand() < 0.1:
                messy_zone = zone.lower()
            elif np.random.rand() < 0.1:
                messy_zone = zone.upper()
                
            record_1 = {
                "zone_id": messy_zone,
                "timestamp": dt.isoformat(),
                "river_level": current_river_val,
                "rainfall": r_val
            }
            river_rainfall_records.append(record_1)
            # Inject duplicate
            if np.random.rand() < 0.03:
                river_rainfall_records.append(record_1)
            
            # Emergency calls (correlated with bad conditions)
            base_calls = np.random.poisson(lam=2)
            if current_river > 4.5 or rainfall > 30:
                base_calls += np.random.poisson(lam=20)
                
            # Inject NaNs and negative outliers into calls
            if np.random.rand() < 0.08:
                final_calls = np.nan
            elif np.random.rand() < 0.04:
                final_calls = -50
            else:
                final_calls = int(base_calls)
                
            record_2 = {
                "zone": messy_zone.swapcase(), # Intentionally different case for cleaning step
                "time_stamp": dt.strftime("%Y/%m/%d %H:%M:%S"), # Intentionally different format
                "emergency_call_volume": final_calls
            }
            emergency_calls_records.append(record_2)
            if np.random.rand() < 0.03:
                emergency_calls_records.append(record_2)
            
            # Infrastructure closures
            prob_closure = 1 / (1 + np.exp(-(current_river - 4.0) * 2.0))
            road_closures = int(np.random.binomial(10, prob_closure))
            bridge_closures = int(np.random.binomial(3, prob_closure * 0.5))
            
            # Inject NaNs into infrastructure
            if np.random.rand() < 0.05:
                road_closures = np.nan
            if np.random.rand() < 0.05:
                bridge_closures = np.nan
                
            record_3 = {
                "zone_id": messy_zone,
                "timestamp": dt.isoformat(),
                "road_closures": road_closures,
                "bridge_closures": bridge_closures
            }
            infrastructure_records.append(record_3)
            if np.random.rand() < 0.03:
                infrastructure_records.append(record_3)
            
    # Save 1, 3, 4
    pd.DataFrame(river_rainfall_records).to_csv(os.path.join(raw_dir, "river_rainfall_data.csv"), index=False)
    pd.DataFrame(emergency_calls_records).to_csv(os.path.join(raw_dir, "emergency_calls.csv"), index=False)
    pd.DataFrame(infrastructure_records).to_csv(os.path.join(raw_dir, "infrastructure_records.csv"), index=False)
    
    # 2. Historical Flood Logs
    historical_records = []
    for zone in zones:
        for _ in range(50):
            hist_river = np.random.uniform(1.5, 7.0)
            hist_rain = np.random.uniform(0.0, 100.0)
            # Severe conditions cause floods
            outcome = "FLOODED" if hist_river > 4.5 or hist_rain > 60 else "NORMAL"
            historical_records.append({
                "zone_id": zone,
                "recorded_river_level": round(hist_river, 2),
                "recorded_rainfall_24h": round(hist_rain, 2),
                "historical_outcome": outcome
            })
    pd.DataFrame(historical_records).to_csv(os.path.join(raw_dir, "historical_flood_logs.csv"), index=False)
    print(f"Synthetic raw data successfully generated in '{raw_dir}'")


def load_river_rainfall(raw_dir: str) -> pd.DataFrame:
    path = os.path.join(raw_dir, "river_rainfall_data.csv")
    df = pd.read_csv(path)
    print(f"Loaded river/rainfall data: {len(df)} records, {len(df.columns)} columns")
    return df

def load_emergency_calls(raw_dir: str) -> pd.DataFrame:
    path = os.path.join(raw_dir, "emergency_calls.csv")
    df = pd.read_csv(path)
    print(f"Loaded emergency calls data: {len(df)} records, {len(df.columns)} columns")
    return df

def load_infrastructure_records(raw_dir: str) -> pd.DataFrame:
    path = os.path.join(raw_dir, "infrastructure_records.csv")
    df = pd.read_csv(path)
    print(f"Loaded infrastructure data: {len(df)} records, {len(df.columns)} columns")
    return df

def load_historical_flood_logs(raw_dir: str) -> pd.DataFrame:
    path = os.path.join(raw_dir, "historical_flood_logs.csv")
    df = pd.read_csv(path)
    print(f"Loaded historical flood logs: {len(df)} records, {len(df.columns)} columns")
    return df

if __name__ == "__main__":
    # Test execution
    raw_directory = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    generate_synthetic_data(raw_directory)
    load_river_rainfall(raw_directory)
    load_emergency_calls(raw_directory)
    load_infrastructure_records(raw_directory)
    load_historical_flood_logs(raw_directory)
