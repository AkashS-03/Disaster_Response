import pandas as pd
import numpy as np

def clean_river_rainfall(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the river and rainfall gauges dataset.
    """
    df_clean = df.copy()
    
    # Standardize zone_id casing
    df_clean['zone_id'] = df_clean['zone_id'].str.title()
    
    # Standardize timestamps to datetime format
    df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'], utc=True)
    
    # Do NOT handle missing values or impossible values (negative rainfall/extreme rivers).
    # We leave these deliberately for the EDA engineer to discover and handle.
    
    # Deduplicate
    df_clean = df_clean.drop_duplicates(subset=['zone_id', 'timestamp'])
    
    return df_clean

def clean_emergency_calls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the emergency calls dataset.
    Handles inconsistent column names, formats, and handles missing/duplicate records.
    """
    df_clean = df.copy()
    
    # Standardize column names
    df_clean = df_clean.rename(columns={'zone': 'zone_id', 'time_stamp': 'timestamp'})
    
    # Standardize zone_id casing
    df_clean['zone_id'] = df_clean['zone_id'].str.title()
    
    # Standardize timestamps
    df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'], utc=True)
    
    # Deduplicate
    df_clean = df_clean.drop_duplicates(subset=['zone_id', 'timestamp'])
    
    return df_clean

def clean_infrastructure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the infrastructure records dataset.
    """
    df_clean = df.copy()
    
    df_clean['zone_id'] = df_clean['zone_id'].str.title()
    df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'], utc=True)
    
    df_clean = df_clean.drop_duplicates(subset=['zone_id', 'timestamp'])
    
    return df_clean

def clean_historical_logs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans historical flood logs.
    """
    df_clean = df.copy()
    df_clean['zone_id'] = df_clean['zone_id'].str.title()
    
    # Map text outcomes to a numerical indicator
    outcome_map = {"FLOODED": 1, "NORMAL": 0}
    df_clean['historical_flood_indicator'] = df_clean['historical_outcome'].map(outcome_map)
    df_clean = df_clean.drop(columns=['historical_outcome'])
    
    # Aggregate to provide a historical flood probability per zone per river_level bucket
    df_clean['river_level_bucket'] = df_clean['recorded_river_level'].round(0)
    
    hist_agg = df_clean.groupby(['zone_id', 'river_level_bucket'])['historical_flood_indicator'].mean().reset_index()
    hist_agg = hist_agg.rename(columns={'historical_flood_indicator': 'historical_flood_probability'})
    
    return hist_agg
