import os
import pandas as pd
import requests
import numpy as np

+
def run_ingestion():
    print("--- Starting Data Engineer Pipeline (Real API Data) ---")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_dir = os.path.join(base_dir, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    # We are using 4 distinct zones around Mumbai, India
    zones = ['Zone_A', 'Zone_B', 'Zone_C', 'Zone_D']
    lats = [18.93, 19.05, 19.07, 19.23] # South Mumbai, Bandra, Kurla (Mithi River), Borivali
    lons = [72.82, 72.83, 72.88, 72.85]
    
    # Let's pull exactly 1 full year of historical hourly data (8,784 rows per zone = ~35,000 total rows)
    start_date = '2024-01-01'
    end_date = '2024-12-31'
    
    weather_frames = []
    river_frames = []
    calls_frames = []
    infra_frames = []
    
    for i, zone in enumerate(zones):
        print(f"Fetching real weather data from Open-Meteo for {zone} (Mumbai)...")
        url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lats[i]}&longitude={lons[i]}&start_date={start_date}&end_date={end_date}&hourly=rain"
        
        try:
            res = requests.get(url)
            res.raise_for_status()
            data = res.json()['hourly']
            
            # 1. Weather Data (Real)
            df_weather = pd.DataFrame({
                'timestamp': pd.to_datetime(data['time']),
                'zone_id': zone,
                'rainfall': data['rain']
            })
            
            # 2. River Level (Simulated physics based on real rain)
            # Rivers rise when it rains heavily and slowly drain over time
            rain_array = df_weather['rainfall'].fillna(0).values
            river_levels = np.zeros(len(rain_array))
            current_level = 1.5 # Base river depth in meters
            
            for j in range(len(rain_array)):
                current_level += (rain_array[j] * 0.15) # Rain adds to river volume
                current_level -= 0.05 # Natural drainage per hour
                current_level = max(1.0, current_level) # River can't go below 1.0m
                river_levels[j] = current_level
                
            df_river = pd.DataFrame({
                'timestamp': df_weather['timestamp'],
                'zone_id': zone,
                # Add slight sensor noise to our physical runoff model
                'river_level': river_levels + np.random.normal(0, 0.05, len(river_levels)) 
            })
            
            # 3. Emergency Calls (Driven by real rain/river levels)
            calls = np.where(rain_array > 10, np.random.randint(50, 200, len(rain_array)), np.random.randint(5, 20, len(rain_array)))
            df_calls = pd.DataFrame({
                'timestamp': df_weather['timestamp'],
                'zone_id': zone,
                'emergency_call_volume': calls
            })
            
            # 4. Infrastructure (Driven by river levels)
            # Kurla (Zone C) is notoriously flood-prone near the Mithi river, so we increase its base probability
            road = np.where(river_levels > 3.5, np.random.randint(1, 5, len(river_levels)), 0)
            bridge = np.where(river_levels > 4.5, np.random.randint(1, 3, len(river_levels)), 0)
            df_infra = pd.DataFrame({
                'timestamp': df_weather['timestamp'],
                'zone_id': zone,
                'road_closures': road,
                'bridge_closures': bridge,
                'historical_flood_probability': 0.9 if zone == 'Zone_C' else 0.3
            })
            
            weather_frames.append(df_weather)
            river_frames.append(df_river)
            calls_frames.append(df_calls)
            infra_frames.append(df_infra)
            
        except Exception as e:
            print(f"Error fetching data for {zone}: {e}")
            
    # Combine and save locally
    pd.concat(weather_frames).to_csv(os.path.join(raw_dir, "weather_data.csv"), index=False)
    pd.concat(river_frames).to_csv(os.path.join(raw_dir, "river_data.csv"), index=False)
    pd.concat(calls_frames).to_csv(os.path.join(raw_dir, "emergency_calls.csv"), index=False)
    pd.concat(infra_frames).to_csv(os.path.join(raw_dir, "infrastructure_status.csv"), index=False)
    
    print("Data ingestion complete! Saved real API data to data/raw/")

if __name__ == "__main__":
    run_ingestion()
