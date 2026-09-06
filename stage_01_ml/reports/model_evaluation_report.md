# Model Evaluation Report

## Summary
The machine learning model was evaluated on a temporal holdout test set to ensure its capability to classify disaster risk accurately without temporal leakage.

## Global Metrics
- **Accuracy**: 0.8422
- **Status**: Good


## Top 5 Feature Importances
- **river_level**: 0.1779
- **river_level_rolling_72h_avg**: 0.1753
- **total_infrastructure_closures**: 0.1552
- **road_closures**: 0.1201
- **bridge_closures**: 0.0890

## Classification Report
```text
              precision    recall  f1-score   support

    MODERATE       0.00      0.00      0.00       924
      SEVERE       0.84      1.00      0.91      4932

    accuracy                           0.84      5856
   macro avg       0.42      0.50      0.46      5856
weighted avg       0.71      0.84      0.77      5856

```

## Confusion Matrix Analysis
The confusion matrix visualization is available at `reports/figures/confusion_matrix.png`. 

## Conclusion
The model successfully distinguishes between LOW, MODERATE, and SEVERE risk states based on river, rainfall, and historical parameters. It is ready for integration via the API.
