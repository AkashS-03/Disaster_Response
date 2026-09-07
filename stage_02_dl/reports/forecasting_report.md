# Time-Series Forecasting Evaluation

Target: Predict `river_level` 12 hours into the future using 48-hour trailing windows.
Chronological Split enforced. Test Set: Nov 2024 - Dec 2024.

## Metrics (Mean Absolute Error in meters)
- **Naive Persistence:** 0.6609m
- **XGBoost Lag Baseline:** 16.5453m
- **PyTorch LSTM:** 13.0651m
