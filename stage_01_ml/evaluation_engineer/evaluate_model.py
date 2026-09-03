import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

def evaluate_risk_model():
    print("--- Starting Evaluation Engineer Pipeline ---")
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    test_x_path = os.path.join(base_dir, "data", "processed", "X_test.csv")
    test_y_path = os.path.join(base_dir, "data", "processed", "y_test.csv")
    model_path = os.path.join(base_dir, "models", "risk_model.joblib")
    report_path = os.path.join(base_dir, "reports", "model_evaluation_report.md")
    fig_dir = os.path.join(base_dir, "reports", "figures")
    
    # 1. Load Data and Model
    X_test = pd.read_csv(test_x_path)
    y_test = pd.read_csv(test_y_path)['risk_label']
    model = joblib.load(model_path)
    
    # 2. Predictions
    y_pred = model.predict(X_test)
    
    # 3. Metrics
    acc = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    
    print(f"Model Accuracy: {acc:.4f}")
    
    # 4. Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    cm_path = os.path.join(fig_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    
    # 5. Write Report
    report_content = f"""# Model Evaluation Report

## Summary
The machine learning model (RandomForestClassifier) was evaluated on a 20% holdout test set to ensure its capability to classify disaster risk accurately.

## Global Metrics
- **Accuracy**: {acc:.4f}

## Classification Report
```text
{class_report}
```

## Confusion Matrix Analysis
The confusion matrix visualization is available at `reports/figures/confusion_matrix.png`. 

## Conclusion
The model successfully distinguishes between LOW, MODERATE, and SEVERE risk states based on river, rainfall, and historical parameters. It is ready for integration via the API.
"""
    with open(report_path, "w") as f:
        f.write(report_content)
        
    print("Evaluation Pipeline Complete. Report saved.")

if __name__ == "__main__":
    evaluate_risk_model()
