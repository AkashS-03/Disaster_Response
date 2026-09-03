# EDA Engineer Report

## Data Quality Actions Taken
- Standardized `zone_id` casing (e.g., 'zOnE_a' -> 'Zone_A').
- Dropped all duplicate temporal records.
- Capped impossible outliers (e.g., negative rainfall bounded to 0.0, extreme river heights > 20m bounded to 20m).
- Imputed missing values (forward fill for river levels, 0 for rainfall and counts).

## Final Dataset Shape
- **Rows**: 3359
- **Columns**: 14

## Target Variable Distribution
risk_label
LOW         2799
MODERATE     528
SEVERE        32

## Key Findings
- The correlation heatmap (`reports/figures/correlation_heatmap.png`) shows strong relationships between `river_level_rolling_72h_avg` and infrastructure closures.
- `risk_label` is highly dependent on both river level and rainfall windows.

## ML Handoff
Cleaned dataset is verified and saved at `data/processed/clean_modeling_dataset.csv`. Ready for ML Engineer.
