# Cross-Validation Report

## Overview
This report presents 5-fold stratified cross-validation results.
Cross-validation provides more reliable performance estimates than a single train/test split.

## Methodology
- **Folds**: 5-fold stratified
- **Shuffling**: Enabled (random_state=42)
- **Stratification**: Ensures proportional class representation in each fold
- **Pipeline**: Same as ML Engineer (Imputer, Scaler, RandomForest)

## Results Summary
| Metric | Mean | Std | 95% CI | Overfitting Gap |
|--------|------|-----|--------|-----------------|
| accuracy | 1.0000 | 0.0001 | [0.9999, 1.0001] | 0.0000 |
| precision_macro | 0.9917 | 0.0167 | [0.9590, 1.0243] | 0.0083 |
| recall_macro | 1.0000 | 0.0000 | [0.9999, 1.0001] | 0.0000 |
| f1_macro | 0.9955 | 0.0089 | [0.9781, 1.0130] | 0.0045 |

## Per-Fold Accuracy
| Fold | Accuracy |
|------|----------|
| 1 | 1.0000 |
| 2 | 1.0000 |
| 3 | 0.9999 |
| 4 | 1.0000 |
| 5 | 1.0000 |

## Stability Analysis
- **Mean Accuracy**: 1.0000
- **Std Deviation**: 0.0001
- **Coefficient of Variation**: 0.0001
- **Stability Score**: 0.9999 (1.0 = perfectly stable)

## Visualizations
- **Cross-Validation Results**: `reports/figures/cross_validation_results.png`

## Key Findings
1. **Performance Consistency**: Excellent - metrics vary by only 0.0001 across folds
2. **Overfitting Check**: No significant overfitting (train-test gap: 0.0000)
3. **95% Confidence Interval**: True accuracy is between 0.9999 and 1.0001

## Comparison with Single Split
The single train/test split showed accuracy of 1.0000.
Cross-validation confirms this is reliable 
with a 95% CI of [0.9999, 1.0001].

## Implications for Disaster Response
- **High Confidence**: Model performance is consistent and reliable
- **Low Variance**: Predictions are stable across different data splits
- **Ready for Deployment**: Cross-validation confirms model generalizes well
