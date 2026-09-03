import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


def detect_overconfidence():
    """
    Detect cases where the model is confident but wrong.
    These are the most dangerous predictions in a disaster scenario.
    """
    print("--- Starting Overconfidence Detection ---")
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    test_x_path = os.path.join(base_dir, "data", "processed", "X_test.csv")
    test_y_path = os.path.join(base_dir, "data", "processed", "y_test.csv")
    model_path = os.path.join(base_dir, "models", "risk_model.joblib")
    fig_dir = os.path.join(base_dir, "reports", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    # 1. Load Data and Model
    print("Loading test data and model...")
    X_test = pd.read_csv(test_x_path)
    y_test = pd.read_csv(test_y_path)['risk_label']
    model = joblib.load(model_path)
    
    # 2. Get Predictions and Probabilities
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    classes = model.classes_
    
    # 3. Build Results DataFrame
    results_df = X_test.copy()
    results_df['actual'] = y_test.values
    results_df['predicted'] = y_pred
    results_df['correct'] = results_df['actual'] == results_df['predicted']
    results_df['confidence'] = np.max(y_prob, axis=1)
    
    # Add per-class probabilities
    for i, cls in enumerate(classes):
        results_df[f'prob_{cls}'] = y_prob[:, i]
    
    # 4. Identify Overconfident Wrong Predictions
    # Definition: confidence > 80% but prediction is wrong
    overconfident_threshold = 0.8
    overconfident_wrong = results_df[
        (results_df['confidence'] > overconfident_threshold) & 
        (~results_df['correct'])
    ].copy()
    
    total_predictions = len(results_df)
    total_wrong = (~results_df['correct']).sum()
    total_overconfident_wrong = len(overconfident_wrong)
    
    print(f"\n=== Overconfidence Analysis ===")
    print(f"Total predictions: {total_predictions}")
    print(f"Total incorrect: {total_wrong}")
    print(f"Overconfident wrong (>80% confidence, wrong): {total_overconfident_wrong}")
    
    if total_wrong > 0:
        print(f"Percentage of errors that are overconfident: {total_overconfident_wrong/total_wrong:.2%}")
    
    # 5. Analyze by Confidence Bucket
    confidence_buckets = pd.cut(results_df['confidence'], bins=[0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0])
    bucket_analysis = results_df.groupby(confidence_buckets, observed=True).agg(
        total=('correct', 'count'),
        correct=('correct', 'sum'),
        incorrect=('correct', lambda x: (~x).sum())
    ).reset_index()
    bucket_analysis['error_rate'] = bucket_analysis['incorrect'] / bucket_analysis['total']
    bucket_analysis['confidence_bucket'] = bucket_analysis['confidence'].astype(str)
    
    print(f"\n=== Error Rate by Confidence Bucket ===")
    for _, row in bucket_analysis.iterrows():
        print(f"  {row['confidence_bucket']}: {row['error_rate']:.2%} error rate ({row['incorrect']}/{row['total']})")
    
    # 6. Per-Class Overconfidence Analysis
    print(f"\n=== Per-Class Overconfidence ===")
    class_overconfidence = {}
    for cls in classes:
        cls_mask = results_df['actual'] == cls
        cls_results = results_df[cls_mask]
        cls_wrong = cls_results[~cls_results['correct']]
        cls_overconfident = cls_wrong[cls_wrong['confidence'] > overconfident_threshold]
        
        class_overconfidence[cls] = {
            'total': len(cls_results),
            'wrong': len(cls_wrong),
            'overconfident_wrong': len(cls_overconfident),
            'overconfidence_rate': len(cls_overconfident) / len(cls_wrong) if len(cls_wrong) > 0 else 0
        }
        
        print(f"  {cls}: {len(cls_overconfident)}/{len(cls_wrong)} errors are overconfident "
              f"({class_overconfidence[cls]['overconfidence_rate']:.2%})")
    
    # 7. Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Confidence vs Correctness
    correct_conf = results_df[results_df['correct']]['confidence']
    incorrect_conf = results_df[~results_df['correct']]['confidence']
    
    axes[0, 0].hist(correct_conf, bins=20, alpha=0.7, label='Correct', color='green', density=True)
    if len(incorrect_conf) > 0:
        axes[0, 0].hist(incorrect_conf, bins=20, alpha=0.7, label='Incorrect', color='red', density=True)
    axes[0, 0].set_xlabel('Confidence')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Confidence Distribution by Correctness')
    axes[0, 0].legend()
    axes[0, 0].axvline(x=overconfident_threshold, color='black', linestyle='--', label='80% threshold')
    
    # Plot 2: Error Rate by Confidence Bucket
    axes[0, 1].bar(range(len(bucket_analysis)), bucket_analysis['error_rate'], color='coral', edgecolor='black')
    axes[0, 1].set_xticks(range(len(bucket_analysis)))
    axes[0, 1].set_xticklabels(bucket_analysis['confidence_bucket'], rotation=45, ha='right')
    axes[0, 1].set_xlabel('Confidence Bucket')
    axes[0, 1].set_ylabel('Error Rate')
    axes[0, 1].set_title('Error Rate by Confidence Level')
    axes[0, 1].set_ylim([0, 1])
    
    # Plot 3: Per-Class Overconfidence
    class_names = list(class_overconfidence.keys())
    overconf_counts = [class_overconfidence[c]['overconfident_wrong'] for c in class_names]
    total_wrong_counts = [class_overconfidence[c]['wrong'] for c in class_names]
    
    x = np.arange(len(class_names))
    width = 0.35
    axes[1, 0].bar(x - width/2, total_wrong_counts, width, label='Total Wrong', color='salmon')
    axes[1, 0].bar(x + width/2, overconf_counts, width, label='Overconfident Wrong', color='darkred')
    axes[1, 0].set_xlabel('Class')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Overconfidence by Class')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(class_names)
    axes[1, 0].legend()
    
    # Plot 4: Most Dangerous Predictions (High confidence + Wrong)
    if len(overconfident_wrong) > 0:
        top_dangerous = overconfident_wrong.nlargest(10, 'confidence')
        y_pos = range(len(top_dangerous))
        axes[1, 1].barh(y_pos, top_dangerous['confidence'].values, color='darkred')
        axes[1, 1].set_yticks(y_pos)
        axes[1, 1].set_yticklabels([f"{row['actual']}->{row['predicted']}" 
                                      for _, row in top_dangerous.iterrows()], fontsize=8)
        axes[1, 1].set_xlabel('Confidence')
        axes[1, 1].set_title('Top 10 Most Dangerous Predictions')
        axes[1, 1].axvline(x=overconfident_threshold, color='black', linestyle='--')
    else:
        axes[1, 1].text(0.5, 0.5, 'No overconfident\nwrong predictions', 
                        ha='center', va='center', fontsize=14)
        axes[1, 1].set_title('Top 10 Most Dangerous Predictions')
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'overconfidence_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nVisualization saved to: {fig_dir}/overconfidence_analysis.png")
    
    # 8. Generate Report
    report_content = f"""# Overconfidence Detection Report

## Overview
This report identifies cases where the model is confident but wrong.
In disaster response, these are the **most dangerous predictions** - the system believes 
it's right but is actually making a wrong decision.

## Key Metrics
| Metric | Value |
|--------|-------|
| Total Predictions | {total_predictions} |
| Total Incorrect | {total_wrong} |
| Overconfident Wrong (>80% confidence) | {total_overconfident_wrong} |
| Error Rate | {total_wrong/total_predictions:.2%} |

## Error Rate by Confidence Level
| Confidence Bucket | Error Rate | Incorrect/Total |
|-------------------|------------|-----------------|
"""
    for _, row in bucket_analysis.iterrows():
        report_content += f"| {row['confidence_bucket']} | {row['error_rate']:.2%} | {row['incorrect']}/{row['total']} |\n"
    
    report_content += f"""
## Per-Class Overconfidence
| Class | Total | Wrong | Overconfident Wrong | Overconfidence Rate |
|-------|-------|-------|---------------------|---------------------|
"""
    for cls in classes:
        stats = class_overconfidence[cls]
        report_content += f"| {cls} | {stats['total']} | {stats['wrong']} | {stats['overconfident_wrong']} | {stats['overconfidence_rate']:.2%} |\n"
    
    report_content += f"""
## Visualizations
- **Overconfidence Analysis**: `reports/figures/overconfidence_analysis.png`

## Key Findings
- {"No overconfident wrong predictions detected. Model is well-calibrated." if total_overconfident_wrong == 0 else f"{total_overconfident_wrong} overconfident wrong predictions detected."}
- {"The model shows appropriate uncertainty when making errors." if total_overconfident_wrong == 0 else "Some errors have high confidence, which is dangerous for disaster response."}

## Risk Assessment for Disaster Response
- **Low Risk**: Model is uncertain when wrong (can be caught by confidence threshold)
- **Medium Risk**: Some errors have moderate confidence (need human review)
- **High Risk**: Errors with >90% confidence (critical failures)
- **Critical**: Errors with >95% confidence (system must be retrained)

## Recommendations
1. {"No immediate action needed - model confidence is well-calibrated." if total_overconfident_wrong == 0 else "Implement confidence threshold alerts for predictions >80% confidence."}
2. {"Continue monitoring during deployment." if total_overconfident_wrong == 0 else "Focus retraining on cases where model is overconfident but wrong."}
3. {"Consider adding uncertainty estimation to the API responses." if total_overconfident_wrong == 0 else "Add human-in-the-loop verification for high-confidence predictions."}
"""
    
    report_path = os.path.join(base_dir, "reports", "overconfidence_report.md")
    with open(report_path, "w") as f:
        f.write(report_content)
    print(f"Overconfidence report saved to: {report_path}")
    
    print("--- Overconfidence Detection Complete ---")
    return {
        'total_predictions': total_predictions,
        'total_wrong': total_wrong,
        'overconfident_wrong': total_overconfident_wrong,
        'class_overconfidence': class_overconfidence
    }


if __name__ == "__main__":
    detect_overconfidence()
