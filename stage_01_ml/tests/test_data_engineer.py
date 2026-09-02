import os
import sys
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_engineer.data_ingestion import generate_synthetic_data, load_river_rainfall
from data_engineer.data_cleaning import clean_river_rainfall, clean_emergency_calls
from data_engineer.data_validation import validate_master_dataset
from data_engineer.build_master_dataset import build_master_dataset

def test_generate_and_load_data(tmp_path):
    """Test data generation and loading."""
    raw_dir = tmp_path / "raw"
    generate_synthetic_data(str(raw_dir))
    
    assert os.path.exists(raw_dir / "river_rainfall_data.csv")
    
    df = load_river_rainfall(str(raw_dir))
    assert not df.empty
    assert "zone_id" in df.columns
    assert "river_level" in df.columns

def test_data_cleaning():
    """Test cleaning operations like negative value capping and casing."""
    raw_data = pd.DataFrame({
        "zone_id": ["zone_a", "ZONE_B"],
        "timestamp": ["2026-08-01 10:00", "2026-08-01 11:00"],
        "river_level": [2.5, 3.0],
        "rainfall": [-10.0, 50.0] # -10 should be capped to 0
    })
    
    clean_df = clean_river_rainfall(raw_data)
    
    # Check casing
    assert clean_df["zone_id"].iloc[0] == "Zone_A"
    assert clean_df["zone_id"].iloc[1] == "Zone_B"
    
    # Check negative rainfall cap is NO LONGER applied (left for EDA)
    assert clean_df["rainfall"].iloc[0] == -10.0

def test_data_validation_fails_on_missing_cols():
    """Test that validation correctly flags missing required columns."""
    df_invalid = pd.DataFrame({
        "zone_id": ["Zone_A"],
        "timestamp": ["2026-08-01"]
        # Missing all other required columns
    })
    
    assert validate_master_dataset(df_invalid) == False

def test_full_pipeline():
    """Test the complete dataset builder pipeline."""
    # Run the pipeline
    build_master_dataset()
    
    # Verify output exists
    processed_file = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "data", "processed", "master_dataset.csv"
    ))
    
    assert os.path.exists(processed_file), "Master dataset was not created."
    
    # Verify contents
    df = pd.read_csv(processed_file)
    assert not df.empty
    assert "risk_label" in df.columns
    assert "river_level_rolling_72h_avg" in df.columns
    assert "rainfall_rolling_72h_sum" in df.columns
