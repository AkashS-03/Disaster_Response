# Hydrological Time-Series Forecasting Evaluation

Target: Predict `river_level` 12 hours ahead from 48-hour trailing rolling windows.

## Benchmark Comparison (Mean Absolute Error)
- **Naive Persistence:** 0.6609m
- **XGBoost Regressor:** 16.6231m
- **Fine-Tuned PyTorch LSTM:** **6.2570m**
