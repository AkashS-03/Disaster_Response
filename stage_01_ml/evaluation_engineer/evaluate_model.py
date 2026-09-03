import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

try:
    from .calibration_test import run_calibration_analysis
    from .edge_case_test import run_edge_case_tests
    from .overconfidence_detector import detect_overconfidence
    from .cross_validation import run_cross_validation
except ImportError:
    from calibration_test import run_calibration_analysis
    from edge_case_test import run_edge_case_tests
    from overconfidence_detector import detect_overconfidence
    from cross_validation import run_cross_validation


def evaluate_risk_model():
    """
    Main Evaluation Engineer Pipeline.
    
    This script orchestrates all evaluation components:
    1. Basic Metrics (accuracy, precision, recall, F1)
    2. Confusion Matrix Analysis
    3. Calibration Testing (are probabilities trustworthy?)
    4. Edge Case Testing (boundary conditions)
    5. Overconfidence Detection (confident but wrong)
    6. Cross-Validation (reliability of results)
    
    Returns a comprehensive evaluation of the model's fitness for disaster response.
    """
    print("=" * 70)
    print("  DISASTER RESPONSE AI - COMPREHENSIVE MODEL EVALUATION")
    print("=" * 70)
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    test_x_path = os.path.join(base_dir, "data", "processed", "X_test.csv")
    test_y_path = os.path.join(base_dir, "data", "processed", "y_test.csv")
    model_path = os.path.join(base_dir, "models", "risk_model.joblib")
    report_path = os.path.join(base_dir, "reports", "model_evaluation_report.md")
    fig_dir = os.path.join(base_dir, "reports", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    # =========================================================================
    # PHASE 1: Basic Metrics
    # =========================================================================
    print("\n" + "=" * 70)
    print("  PHASE 1: Basic Metrics Analysis")
    print("=" * 70)
    
    print("Loading test data and model...")
    X_test = pd.read_csv(test_x_path)
    y_test = pd.read_csv(test_y_path)['risk_label']
    model = joblib.load(model_path)
    
    print("Generating predictions...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    classes = model.classes_
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    
    print(f"\nModel Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix - Disaster Risk Classification")
    plt.xlabel("Predicted Risk Level")
    plt.ylabel("Actual Risk Level")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "confusion_matrix.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Confusion matrix saved.")
    
    # Per-class analysis
    print("\n=== Per-Class Performance ===")
    for cls in classes:
        cls_idx = list(classes).index(cls)
        tp = cm[cls_idx, cls_idx]
        fp = cm[:, cls_idx].sum() - tp
        fn = cm[cls_idx, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        print(f"\n{cls}:")
        print(f"  True Positives:  {tp}")
        print(f"  False Positives: {fp}")
        print(f"  False Negatives: {fn}")
        print(f"  True Negatives:  {tn}")
        
        if cls == 'SEVERE':
            if fn > 0:
                print(f"  [!] WARNING: {fn} SEVERE cases MISSED!")
            else:
                print(f"  [OK] All SEVERE cases correctly identified")
    
    # =========================================================================
    # PHASE 2: Calibration Analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("  PHASE 2: Calibration Analysis")
    print("=" * 70)
    print("Testing if model probabilities are trustworthy...")
    brier_scores, conf_stats = run_calibration_analysis()
    
    # =========================================================================
    # PHASE 3: Edge Case Testing
    # =========================================================================
    print("\n" + "=" * 70)
    print("  PHASE 3: Edge Case Testing")
    print("=" * 70)
    print("Testing model at decision boundaries...")
    edge_results = run_edge_case_tests()
    
    # =========================================================================
    # PHASE 4: Overconfidence Detection
    # =========================================================================
    print("\n" + "=" * 70)
    print("  PHASE 4: Overconfidence Detection")
    print("=" * 70)
    print("Finding cases where model is confident but wrong...")
    overconf_results = detect_overconfidence()
    
    # =========================================================================
    # PHASE 5: Cross-Validation
    # =========================================================================
    print("\n" + "=" * 70)
    print("  PHASE 5: Cross-Validation")
    print("=" * 70)
    print("Validating results with K-fold cross-validation...")
    cv_metrics = run_cross_validation(n_folds=5)
    
    # =========================================================================
    # FINAL REPORT
    # =========================================================================
    print("\n" + "=" * 70)
    print("  GENERATING COMPREHENSIVE EVALUATION REPORT")
    print("=" * 70)
    
    # Calculate summary statistics
    total_predictions = len(y_test)
    total_correct = (y_test.values == y_pred).sum()
    total_wrong = total_predictions - total_correct
    severe_mask = y_test.values == 'SEVERE'
    severe_total = severe_mask.sum()
    severe_correct = ((y_test.values == 'SEVERE') & (y_pred == 'SEVERE')).sum()
    severe_missed = severe_total - severe_correct
    
    # Edge case summary
    edge_correct = sum(1 for r in edge_results if r['correct'])
    edge_total = len(edge_results)
    
    # Generate comprehensive report
    report_content = f"""# Disaster Response AI - Comprehensive Model Evaluation Report

## Executive Summary
This report provides a complete evaluation of the Zone Risk Score Model for the 
Disaster Response Coordination system. The model classifies city zones into 
**LOW**, **MODERATE**, or **SEVERE** risk levels based on sensor telemetry.

### Key Findings
| Metric | Value | Status |
|--------|-------|--------|
| Overall Accuracy | {acc:.4f} | {"[OK] Excellent" if acc > 0.99 else "[!] Good" if acc > 0.95 else "[X] Poor"} |
| SEVERE Class Recall | {class_report['SEVERE']['recall']:.4f} | {"[OK] Good" if class_report['SEVERE']['recall'] > 0.95 else "[!] Needs Attention" if class_report['SEVERE']['recall'] > 0.80 else "[X] Critical"} |
| SEVERE Cases Missed | {severe_missed} | {"[OK] None" if severe_missed == 0 else f"[X] {severe_missed} cases missed"} |
| Edge Case Accuracy | {edge_correct/edge_total:.2%} | {"[OK] Good" if edge_correct/edge_total > 0.95 else "[!] Needs Improvement"} |
| Calibration (Brier) | {brier_scores['macro_avg']:.4f} | {"[OK] Well Calibrated" if brier_scores['macro_avg'] < 0.1 else "[!] Moderately Calibrated" if brier_scores['macro_avg'] < 0.2 else "[X] Poorly Calibrated"} |
| Overconfident Errors | {overconf_results['overconfident_wrong']} | {"[OK] None" if overconf_results['overconfident_wrong'] == 0 else f"[X] {overconf_results['overconfident_wrong']} found"} |

---

## 1. Basic Metrics Analysis

### Global Metrics
- **Accuracy**: {acc:.4f}
- **Total Predictions**: {total_predictions}
- **Correct Predictions**: {total_correct}
- **Incorrect Predictions**: {total_wrong}

### Classification Report
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| LOW | {class_report['LOW']['precision']:.4f} | {class_report['LOW']['recall']:.4f} | {class_report['LOW']['f1-score']:.4f} | {int(class_report['LOW']['support'])} |
| MODERATE | {class_report['MODERATE']['precision']:.4f} | {class_report['MODERATE']['recall']:.4f} | {class_report['MODERATE']['f1-score']:.4f} | {int(class_report['MODERATE']['support'])} |
| SEVERE | {class_report['SEVERE']['precision']:.4f} | {class_report['SEVERE']['recall']:.4f} | {class_report['SEVERE']['f1-score']:.4f} | {int(class_report['SEVERE']['support'])} |

### Confusion Matrix
The confusion matrix visualization is available at `reports/figures/confusion_matrix.png`.

---

## 2. Calibration Analysis

**Question**: When the model says "90% confident SEVERE", is it actually SEVERE 90% of the time?

### Brier Scores (Lower = Better)
| Class | Brier Score | Rating |
|-------|-------------|--------|
"""
    for cls, score in brier_scores.items():
        rating = "Excellent" if score < 0.05 else "Good" if score < 0.1 else "Moderate" if score < 0.2 else "Poor"
        report_content += f"| {cls} | {score:.6f} | {rating} |\n"
    
    report_content += f"""
### Confidence Distribution
- Mean confidence (correct predictions): {conf_stats['mean_confidence_correct']:.4f}
- Mean confidence (incorrect predictions): {conf_stats['mean_confidence_incorrect'] if conf_stats['mean_confidence_incorrect'] is not None else 'N/A (no errors)'}

---

## 3. Edge Case Testing

**Question**: Does the model behave correctly at decision boundaries?

### Decision Rules
- **SEVERE**: river_level >= 4.5 OR (rainfall_72h >= 150 AND river_level >= 3.5)
- **MODERATE**: river_level >= 3.0 OR rainfall_72h >= 80 OR emergency_calls >= 100
- **LOW**: Everything else

### Results
- **Total Edge Cases Tested**: {edge_total}
- **Correct Predictions**: {edge_correct}
- **Edge Case Accuracy**: {edge_correct/edge_total:.2%}

### Failed Edge Cases
"""
    failed_edges = [r for r in edge_results if not r['correct']]
    if failed_edges:
        for f in failed_edges:
            report_content += f"- **{f['description']}**: Expected {f['expected']}, Got {f['predicted']} (Confidence: {f['confidence']:.2%})\n"
    else:
        report_content += "No failed edge cases. All boundary conditions correctly classified.\n"
    
    report_content += f"""
---

## 4. Overconfidence Detection

**Question**: Is the model ever confident but wrong?

### Summary
- **Total Predictions**: {overconf_results['total_predictions']}
- **Total Incorrect**: {overconf_results['total_wrong']}
- **Overconfident Wrong (>80% confidence)**: {overconf_results['overconfident_wrong']}

### Per-Class Overconfidence
"""
    for cls, stats in overconf_results['class_overconfidence'].items():
        report_content += f"- **{cls}**: {stats['overconfident_wrong']}/{stats['wrong']} errors are overconfident ({stats['overconfidence_rate']:.2%})\n"
    
    report_content += f"""
---

## 5. Cross-Validation

**Question**: Are these results reliable, or just lucky?

### 5-Fold Stratified Cross-Validation Results
| Metric | Mean | 95% CI | Overfitting Gap |
|--------|------|--------|-----------------|
"""
    for metric_name, stats in cv_metrics.items():
        ci_lower = stats['test_mean'] - 1.96 * stats['test_std']
        ci_upper = stats['test_mean'] + 1.96 * stats['test_std']
        report_content += (f"| {metric_name} | {stats['test_mean']:.4f} | "
                          f"[{ci_lower:.4f}, {ci_upper:.4f}] | {stats['overfitting_gap']:.4f} |\n")
    
    report_content += f"""
---

## 6. Overall Assessment

### Fitness for Disaster Response
"""
    
    # Determine overall assessment
    issues = []
    if severe_missed > 0:
        issues.append(f"SEVERE class recall is {class_report['SEVERE']['recall']:.2%} ({severe_missed} cases missed)")
    if brier_scores['macro_avg'] > 0.1:
        issues.append(f"Calibration is not optimal (Brier: {brier_scores['macro_avg']:.4f})")
    if overconf_results['overconfident_wrong'] > 0:
        issues.append(f"{overconf_results['overconfident_wrong']} overconfident wrong predictions detected")
    if edge_correct/edge_total < 0.95:
        issues.append(f"Edge case accuracy is {edge_correct/edge_total:.2%}")
    
    if not issues:
        report_content += """**VERDICT: MODEL IS FIT FOR DISASTER RESPONSE**

The model demonstrates:
- High accuracy across all risk classes
- Well-calibrated probability outputs
- Correct behavior at decision boundaries
- No overconfident wrong predictions
- Consistent performance across cross-validation folds

The model is ready for integration into the disaster response system.
"""
    else:
        report_content += "**VERDICT: MODEL NEEDS IMPROVEMENT BEFORE DEPLOYMENT**\n\n"
        report_content += "Issues identified:\n"
        for issue in issues:
            report_content += f"- {issue}\n"
    
    report_content += f"""
---

## 7. Recommendations

### For ML Engineer
1. {"Address SEVERE class recall - consider class weighting or oversampling" if severe_missed > 0 else "SEVERE class performance is adequate"}
2. {"Investigate calibration improvement techniques" if brier_scores['macro_avg'] > 0.1 else "Calibration is acceptable"}
3. Consider ensemble methods to improve edge case performance

### For Integration Engineer
1. Implement confidence threshold alerts for predictions >80%
2. Add human-in-the-loop verification for SEVERE predictions
3. Log all predictions for continuous monitoring

### For Evaluation Engineer (Future Work)
1. Test with real-world data when available
2. Implement A/B testing framework
3. Add temporal analysis (does performance degrade over time?)

---

## 8. Visualizations

All visualizations are saved in `reports/figures/`:
- `confusion_matrix.png` - Confusion matrix heatmap
- `calibration_reliability_diagram.png` - Calibration curves
- `confidence_distribution.png` - Confidence distribution analysis
- `edge_case_results.png` - Edge case test results
- `overconfidence_analysis.png` - Overconfidence detection
- `cross_validation_results.png` - Cross-validation metrics

---

*Report generated by Evaluation Engineer - Stage 01 ML*
*Disaster Response Coordination System*
"""
    
    with open(report_path, "w") as f:
        f.write(report_content)
    print(f"\nComprehensive evaluation report saved to: {report_path}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("  EVALUATION COMPLETE - SUMMARY")
    print("=" * 70)
    print(f"  Overall Accuracy: {acc:.4f}")
    print(f"  SEVERE Recall: {class_report['SEVERE']['recall']:.4f}")
    print(f"  SEVERE Cases Missed: {severe_missed}")
    print(f"  Edge Case Accuracy: {edge_correct/edge_total:.2%}")
    print(f"  Calibration (Brier): {brier_scores['macro_avg']:.4f}")
    print(f"  Overconfident Errors: {overconf_results['overconfident_wrong']}")
    print(f"  Reports saved to: {base_dir}/reports/")
    print("=" * 70)
    
    return {
        'accuracy': acc,
        'classification_report': class_report,
        'brier_scores': brier_scores,
        'edge_case_accuracy': edge_correct/edge_total,
        'overconfidence': overconf_results,
        'cross_validation': cv_metrics
    }


if __name__ == "__main__":
    evaluate_risk_model()
