import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import joblib


def calculate_brier_score(y_true, y_prob, classes):
    """
    Calculate Brier Score for multi-class classification.
    Brier Score measures the accuracy of probabilistic predictions.
    Lower is better (0.0 = perfect, 1.0 = worst).
    """
    brier_scores = {}
    
    for i, cls in enumerate(classes):
        # Create binary labels: 1 if this class, 0 otherwise
        y_binary = (y_true == cls).astype(int)
        # Get probabilities for this class
        y_class_prob = y_prob[:, i]
        # Brier Score = mean((probability - actual)^2)
        brier_scores[cls] = np.mean((y_class_prob - y_binary) ** 2)
    
    # Macro average across all classes
    brier_scores['macro_avg'] = np.mean(list(brier_scores.values()))
    return brier_scores


def create_reliability_diagram(y_true, y_prob, classes, n_bins=10, save_dir=None):
    """
    Create reliability diagrams for each class.
    A perfectly calibrated model follows the diagonal line.
    """
    fig, axes = plt.subplots(1, len(classes), figsize=(5 * len(classes), 4))
    
    if len(classes) == 1:
        axes = [axes]
    
    for i, cls in enumerate(classes):
        ax = axes[i]
        
        # Binary labels for this class
        y_binary = (y_true == cls).astype(int)
        y_class_prob = y_prob[:, i]
        
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_binary, y_class_prob, n_bins=n_bins, strategy='uniform'
        )
        
        # Plot
        ax.plot(mean_predicted_value, fraction_of_positives, 's-', label='Model', color='blue')
        ax.plot([0, 1], [0, 1], '--', label='Perfectly Calibrated', color='gray')
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(f'Calibration Curve - {cls}')
        ax.legend(loc='lower right')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'calibration_reliability_diagram.png'), dpi=150, bbox_inches='tight')
    
    plt.close()


def analyze_confidence_distribution(y_true, y_pred, y_prob, classes, save_dir=None):
    """
    Analyze the distribution of confidence scores for correct vs incorrect predictions.
    """
    # Get confidence (max probability) for each prediction
    max_probs = np.max(y_prob, axis=1)
    correct_mask = (y_true == y_pred)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot 1: Confidence distribution for correct vs incorrect
    axes[0].hist(max_probs[correct_mask], bins=20, alpha=0.7, label='Correct', color='green', density=True)
    if (~correct_mask).sum() > 0:
        axes[0].hist(max_probs[~correct_mask], bins=20, alpha=0.7, label='Incorrect', color='red', density=True)
    axes[0].set_xlabel('Confidence (Max Probability)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Confidence Distribution: Correct vs Incorrect')
    axes[0].legend()
    
    # Plot 2: Per-class confidence distribution
    for i, cls in enumerate(classes):
        cls_mask = (y_true == cls)
        cls_confidence = max_probs[cls_mask]
        axes[1].hist(cls_confidence, bins=20, alpha=0.5, label=cls, density=True)
    axes[1].set_xlabel('Confidence (Max Probability)')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Per-Class Confidence Distribution')
    axes[1].legend()
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'confidence_distribution.png'), dpi=150, bbox_inches='tight')
    
    plt.close()
    
    # Return statistics
    stats = {
        'mean_confidence_correct': float(max_probs[correct_mask].mean()),
        'mean_confidence_incorrect': float(max_probs[~correct_mask].mean()) if (~correct_mask).sum() > 0 else None,
        'min_confidence_correct': float(max_probs[correct_mask].min()),
        'max_confidence_incorrect': float(max_probs[~correct_mask].max()) if (~correct_mask).sum() > 0 else None,
    }
    
    return stats


def run_calibration_analysis():
    """
    Main calibration analysis pipeline.
    Tests if the model's probability predictions are trustworthy.
    """
    print("--- Starting Calibration Analysis ---")
    
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
    
    # 3. Brier Score Analysis
    print("\n=== Brier Score Analysis ===")
    brier_scores = calculate_brier_score(y_test.values, y_prob, classes)
    for cls, score in brier_scores.items():
        print(f"  {cls}: {score:.6f}")
    
    # 4. Reliability Diagrams
    print("\n=== Generating Reliability Diagrams ===")
    create_reliability_diagram(y_test.values, y_prob, classes, save_dir=fig_dir)
    print(f"  Saved to: {fig_dir}/calibration_reliability_diagram.png")
    
    # 5. Confidence Distribution Analysis
    print("\n=== Confidence Distribution Analysis ===")
    conf_stats = analyze_confidence_distribution(y_test.values, y_pred, y_prob, classes, save_dir=fig_dir)
    print(f"  Mean confidence (correct): {conf_stats['mean_confidence_correct']:.4f}")
    if conf_stats['mean_confidence_incorrect'] is not None:
        print(f"  Mean confidence (incorrect): {conf_stats['mean_confidence_incorrect']:.4f}")
    else:
        print(f"  Mean confidence (incorrect): N/A (no incorrect predictions)")
    
    # 6. Per-Class Average Confidence
    print("\n=== Per-Class Average Confidence ===")
    for i, cls in enumerate(classes):
        cls_mask = (y_test.values == cls)
        avg_conf = y_prob[cls_mask, i].mean()
        print(f"  {cls}: {avg_conf:.4f}")
    
    # 7. Generate Calibration Report
    report_content = f"""# Calibration Analysis Report

## Overview
This report evaluates whether the model's probability predictions are trustworthy.
A well-calibrated model means when it predicts "80% SEVERE", that event actually occurs ~80% of the time.

## Brier Score (Lower = Better, 0.0 = Perfect)
| Class | Brier Score |
|-------|-------------|
"""
    for cls, score in brier_scores.items():
        report_content += f"| {cls} | {score:.6f} |\n"
    
    report_content += f"""
### Interpretation
- **0.0 - 0.1**: Excellent calibration
- **0.1 - 0.2**: Good calibration  
- **0.2 - 0.3**: Moderate calibration
- **0.3+**: Poor calibration

## Confidence Distribution Statistics
| Metric | Value |
|--------|-------|
| Mean confidence (correct predictions) | {conf_stats['mean_confidence_correct']:.4f} |
| Mean confidence (incorrect predictions) | {conf_stats['mean_confidence_incorrect'] if conf_stats['mean_confidence_incorrect'] is not None else 'N/A (no errors)'} |
| Min confidence (correct) | {conf_stats['min_confidence_correct']:.4f} |
| Max confidence (incorrect) | {conf_stats['max_confidence_incorrect'] if conf_stats['max_confidence_incorrect'] is not None else 'N/A (no errors)'} |

## Visualizations
- **Reliability Diagram**: `reports/figures/calibration_reliability_diagram.png`
- **Confidence Distribution**: `reports/figures/confidence_distribution.png`

## Key Findings
- The Brier scores indicate {"excellent" if brier_scores['macro_avg'] < 0.1 else "good" if brier_scores['macro_avg'] < 0.2 else "moderate" if brier_scores['macro_avg'] < 0.3 else "poor"} overall calibration.
- {"The model is highly confident in its predictions, but this confidence should be validated against real-world outcomes." if conf_stats['mean_confidence_correct'] > 0.95 else "The model shows appropriate confidence levels."}

## Implications for Disaster Response
- **High Brier scores** mean probability outputs cannot be trusted for decision-making
- **Well-calibrated probabilities** allow threshold-based dispatch decisions
- If probabilities are unreliable, the system should rely on hard classifications only
"""
    
    report_path = os.path.join(base_dir, "reports", "calibration_report.md")
    with open(report_path, "w") as f:
        f.write(report_content)
    print(f"\nCalibration report saved to: {report_path}")
    
    print("--- Calibration Analysis Complete ---")
    return brier_scores, conf_stats


if __name__ == "__main__":
    run_calibration_analysis()
