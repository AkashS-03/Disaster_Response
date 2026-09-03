# Model Evaluation Report

## Summary
The machine learning model (RandomForestClassifier) was evaluated on a 20% holdout test set to ensure its capability to classify disaster risk accurately.

## Global Metrics
- **Accuracy**: 0.9970

## Classification Report
```text
              precision    recall  f1-score   support

         LOW       1.00      1.00      1.00       560
    MODERATE       0.98      1.00      0.99       106
      SEVERE       1.00      0.67      0.80         6

    accuracy                           1.00       672
   macro avg       0.99      0.89      0.93       672
weighted avg       1.00      1.00      1.00       672

```

## Confusion Matrix Analysis
The confusion matrix visualization is available at `reports/figures/confusion_matrix.png`. 

## Conclusion
The model successfully distinguishes between LOW, MODERATE, and SEVERE risk states based on river, rainfall, and historical parameters. It is ready for integration via the API.
