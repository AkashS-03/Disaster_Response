import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
    print(f"Loaded modeling dataset: {df.shape}")
    
    # 2. Define Features & Target
    # Ignore timestamps and categorical zone ids for simple modeling (or one-hot encode zones)
    # Let's one-hot encode zone_id
    df = pd.get_dummies(df, columns=['zone_id'], drop_first=True)
    
    # Target
    y = df['risk_label']
    
    # Features (Drop target, timestamp)
    X = df.drop(columns=['risk_label', 'timestamp'])
    
    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Save test sets for Evaluation Engineer
    X_test.to_csv(os.path.join(test_data_path, "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(test_data_path, "y_test.csv"), index=False)
    
    # 4. Build Pipeline
    # Include an imputer just in case NaNs slip through, standard scaler for numeric
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # 5. Train Model
    print("Training Random Forest Classifier...")
    pipeline.fit(X_train, y_train)
    
    # 6. Save Model
    joblib.dump(pipeline, model_path)
    print(f"Model successfully saved to {model_path}")
    print("ML Pipeline Complete. Ready for Evaluation Engineer.")

if __name__ == "__main__":
    train_risk_model()
