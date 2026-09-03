# Model Evaluation Report

## Summary
The machine learning model (RandomForestClassifier) was evaluated on a 20% holdout test set to ensure its capability to classify disaster risk accurately.

## Global Metrics
- **Accuracy**: 1.0000

## Classification Report
```text
              precision    recall  f1-score   support

         LOW       1.00      1.00      1.00      3082
    MODERATE       1.00      1.00      1.00         7
      SEVERE       1.00      1.00      1.00      3939

    accuracy                           1.00      7028
   macro avg       1.00      1.00      1.00      7028
weighted avg       1.00      1.00      1.00      7028

```

## Confusion Matrix Analysis
The confusion matrix visualization is available at `reports/figures/confusion_matrix.png`. 

## Conclusion
The model successfully distinguishes between LOW, MODERATE, and SEVERE risk states based on river, rainfall, and historical parameters. It is ready for integration via the API.
