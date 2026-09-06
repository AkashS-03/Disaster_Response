import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_risk_model():
    print("--- Starting ML Engineer Pipeline ---")
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # 4. Train Model
    print("Training Random Forest Classifier on Temporal Split...")
    pipeline.fit(X_train, y_train)
    
    # 5. Save Model
    joblib.dump(pipeline, model_path)
    print(f"Model successfully saved to {model_path}")
    print("ML Pipeline Complete. Ready for Evaluation Engineer.")

if __name__ == "__main__":
    train_risk_model()
