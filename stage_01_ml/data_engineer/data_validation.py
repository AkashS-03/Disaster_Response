import pandas as pd

def validate_master_dataset(df: pd.DataFrame) -> bool:
    """
    Validates the final master dataset to ensure data quality before
    handing off to the EDA and ML roles.
    Returns True if valid, raises an Exception or returns False if invalid.
    """
    errors = []
    
    # 1. Check required columns
    required_cols = [
        "zone_id", "timestamp", "river_level", "rainfall", 
        "emergency_call_volume", "road_closures", "bridge_closures",
        "river_level_rolling_72h_avg", "rainfall_rolling_72h_sum",
        "risk_label"
    ]
    for col in required_cols:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")
            
    if errors:
        for e in errors: print(f"FAIL: {e}")
        return False
        
    # 2. Check for missing values in critical columns
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\n[WARNING] Missing values deliberately retained for EDA Engineer:\n{missing[missing > 0]}\n")
        
    # 3. Check for duplicates
    duplicates = df.duplicated(subset=['zone_id', 'timestamp']).sum()
    if duplicates > 0:
        errors.append(f"Found {duplicates} duplicate records for same zone & timestamp.")
        
    # 4. Check numerical ranges (just warn for EDA)
    if (df['rainfall'] < 0).any() or (df['river_level'] < 0).any() or (df['emergency_call_volume'] < 0).any():
        print(f"\n[WARNING] Negative/impossible outliers deliberately retained for EDA Engineer.\n")
        
    # 5. Check risk labels
    valid_labels = {"LOW", "MODERATE", "SEVERE"}
    invalid_labels = set(df['risk_label'].unique()) - valid_labels
    if invalid_labels:
        errors.append(f"Invalid risk labels detected: {invalid_labels}")
        
    if errors:
        print("--- VALIDATION FAILED ---")
        for e in errors:
            print(f"- {e}")
        return False
        
    print("--- VALIDATION PASSED ---")
    print(f"Dataset contains {len(df)} rows and {len(df.columns)} columns.")
    print(f"Risk label distribution:\n{df['risk_label'].value_counts()}")
    return True
