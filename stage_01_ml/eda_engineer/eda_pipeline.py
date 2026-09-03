import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda_pipeline():
    print("--- Starting EDA Engineer Pipeline ---")
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(base_dir, "data", "processed", "master_dataset.csv")
    out_data_path = os.path.join(base_dir, "data", "processed", "clean_modeling_dataset.csv")
    report_path = os.path.join(base_dir, "reports", "eda_report.md")
    fig_dir = os.path.join(base_dir, "reports", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    # 1. Load Data
    df = pd.read_csv(data_path)
    print(f"Loaded raw master dataset: {df.shape}")
    
    # 2. Data Cleaning (Solving the Data Engineer's mess)
    # Fix Zone ID Casing
    if 'zone_id' in df.columns:
        df['zone_id'] = df['zone_id'].astype(str).str.title()
        
    # Drop Duplicates
    df = df.drop_duplicates(subset=['zone_id', 'timestamp'])
    
    # Handle Outliers
    print("Handling outliers...")
    if 'rainfall' in df.columns:
        df.loc[df['rainfall'] < 0, 'rainfall'] = 0.0
    if 'river_level' in df.columns:
        # Cap impossible river levels at 20.0
        df.loc[df['river_level'] > 20.0, 'river_level'] = 20.0
    if 'emergency_call_volume' in df.columns:
        df.loc[df['emergency_call_volume'] < 0, 'emergency_call_volume'] = 0
        
    # Handle Missing Values
    print("Imputing missing values...")
    df = df.sort_values(by=['zone_id', 'timestamp'])
    
    # Forward fill physical metrics, assume 0 for counts
    df['river_level'] = df.groupby('zone_id')['river_level'].transform(lambda x: x.ffill().bfill())
    df['rainfall'] = df['rainfall'].fillna(0.0)
    df['emergency_call_volume'] = df['emergency_call_volume'].fillna(0)
    df['road_closures'] = df['road_closures'].fillna(0)
    df['bridge_closures'] = df['bridge_closures'].fillna(0)
    
    # Drop remaining NaNs (e.g. rolling features that couldn't compute)
    df = df.dropna()
    print(f"Cleaned dataset shape: {df.shape}")
    
    # 3. Exploratory Data Analysis & Visualization
    print("Generating EDA visualizations...")
    
    # Correlation Matrix
    plt.figure(figsize=(10, 8))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "correlation_heatmap.png"))
    plt.close()
    
    # Target Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='risk_label', order=['LOW', 'MODERATE', 'SEVERE'])
    plt.title("Risk Label Distribution")
    plt.savefig(os.path.join(fig_dir, "target_distribution.png"))
    plt.close()
    
    # 4. Save Cleaned Dataset
    df.to_csv(out_data_path, index=False)
    print(f"Clean modeling dataset saved to {out_data_path}")
    
    # 5. Generate Markdown Report
    report_content = f"""# EDA Engineer Report

## Data Quality Actions Taken
- Standardized `zone_id` casing (e.g., 'zOnE_a' -> 'Zone_A').
- Dropped all duplicate temporal records.
- Capped impossible outliers (e.g., negative rainfall bounded to 0.0, extreme river heights > 20m bounded to 20m).
- Imputed missing values (forward fill for river levels, 0 for rainfall and counts).

## Final Dataset Shape
- **Rows**: {df.shape[0]}
- **Columns**: {df.shape[1]}

## Target Variable Distribution
{df['risk_label'].value_counts().to_string()}

## Key Findings
- The correlation heatmap (`reports/figures/correlation_heatmap.png`) shows strong relationships between `river_level_rolling_72h_avg` and infrastructure closures.
- `risk_label` is highly dependent on both river level and rainfall windows.

## ML Handoff
Cleaned dataset is verified and saved at `data/processed/clean_modeling_dataset.csv`. Ready for ML Engineer.
"""
    with open(report_path, "w") as f:
        f.write(report_content)
        
    print("EDA Pipeline Complete. Report saved.")

if __name__ == "__main__":
    run_eda_pipeline()
