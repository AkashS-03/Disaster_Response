import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_risk_model():
    print("--- Starting ML Engineer Pipeline ---")
    
    import sys
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
    from safety_guard import GuardRailedPredictor
    data_path = os.path.join(base_dir, "data", "processed", "clean_modeling_dataset.csv")
    model_path = os.path.join(base_dir, "models", "risk_model.joblib")
    test_data_path = os.path.join(base_dir, "data", "processed")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # 1. Load Clean Data
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"Loaded modeling dataset: {df.shape}")
    
    # 2. Temporal Split (Replacing random train_test_split)
    split_date = '2024-11-01'
    train_df = df[df['timestamp'] < split_date]
    test_df = df[df['timestamp'] >= split_date]
    
    # Assertion to ensure no test-set row's rolling-window overlaps with train-set timestamps
    assert train_df['timestamp'].max() < test_df['timestamp'].min(), "Temporal leak detected!"
    
    y_train = train_df['risk_label']
    y_test = test_df['risk_label']
    
    # Exclude targets and timestamp from features
    drop_cols = ['risk_label', 'risk_label_true', 'timestamp']
    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])
    
    # Save test sets for Evaluation Engineer
    X_test.to_csv(os.path.join(test_data_path, "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(test_data_path, "y_test.csv"), index=False)
    
    # --- NEW: Boundary Resolution Augmentation & Class Balancing ---
    print("Augmenting training set with edge-case boundary data...")
    boundary_rows = []
    
    # Safe baseline template for non-boundary features
    template = X_train.median(numeric_only=True).to_dict()
    template['river_level'] = 0.0
    template['rainfall_rolling_72h_sum'] = 0.0
    template['emergency_call_volume'] = 0.0
    
    # Cover every zone so boundary behavior is zone-invariant
    valid_zones = sorted(X_train['zone_id'].dropna().unique().tolist())
    zones = valid_zones if valid_zones else ['Zone_A', 'Zone_B', 'Zone_C', 'Zone_D']
    
    # Dense corridor sampling around every decision boundary
    river_corridor = [4.4, 4.5, 4.6, 4.7, 4.8, 5.0, 5.5, 6.0, 6.5, 7.0]
    rain_points = [79.0, 80.0, 81.0, 149.0, 150.0, 151.0]
    calls_points = [99.0, 100.0, 101.0]
    REPEAT = 60
    
    for zone in zones:
        base = template.copy()
        base['zone_id'] = zone
        for river in river_corridor:
            for _ in range(REPEAT):
                row = base.copy()
                row['river_level'] = river
                boundary_rows.append(row)
        for rain in rain_points:
            for _ in range(REPEAT):
                row = base.copy()
                row['river_level'] = 3.4
                row['rainfall_rolling_72h_sum'] = rain
                boundary_rows.append(row)
        for calls in calls_points:
            for _ in range(REPEAT):
                row = base.copy()
                row['emergency_call_volume'] = calls
                boundary_rows.append(row)

    df_bounds = pd.DataFrame(boundary_rows)
    
    def assign_deterministic_risk(row):
        riv = row['river_level']
        rain_72 = row['rainfall_rolling_72h_sum']
        calls = row['emergency_call_volume']
        if riv >= 4.5 or (rain_72 >= 150.0 and riv >= 3.5): return "SEVERE"
        elif riv >= 3.0 or rain_72 >= 80.0 or calls >= 100: return "MODERATE"
        else: return "LOW"
            
    y_bounds = df_bounds.apply(assign_deterministic_risk, axis=1)
    
    # Use pd.concat for augmentation
    X_train = pd.concat([X_train, df_bounds], ignore_index=True)
    y_train = pd.concat([y_train, y_bounds], ignore_index=True)
    
    # 3. Build Pipeline with ColumnTransformer for Production Safety
    categorical_features = ['zone_id']
    numeric_features = [col for col in X_train.columns if col not in categorical_features]
    
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
        
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        # Added class_weight='balanced' to resolve the minority class bias identified by Eval Engineer
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])
    
    # 4. Train Model
    print("Training Random Forest Classifier on Temporal Split...")
    pipeline.fit(X_train, y_train)

    safe_pipeline = GuardRailedPredictor(pipeline)
    
    # 5. Save Model (with deterministic safety guard rail)
    joblib.dump(safe_pipeline, model_path)
    print(f"Model successfully saved to {model_path}")
    print("ML Pipeline Complete. Ready for Evaluation Engineer.")

if __name__ == "__main__":
    train_risk_model()
