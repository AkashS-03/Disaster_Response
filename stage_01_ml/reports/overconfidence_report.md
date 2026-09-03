# Overconfidence Detection Report

## Overview
This report identifies cases where the model is confident but wrong.
In disaster response, these are the **most dangerous predictions** - the system believes 
it's right but is actually making a wrong decision.

## Key Metrics
| Metric | Value |
|--------|-------|
| Total Predictions | 7028 |
| Total Incorrect | 0 |
| Overconfident Wrong (>80% confidence) | 0 |
| Error Rate | 0.00% |

## Error Rate by Confidence Level
| Confidence Bucket | Error Rate | Incorrect/Total |
|-------------------|------------|-----------------|
| (0.8, 0.9] | 0.00% | 0/2 |
| (0.9, 0.95] | 0.00% | 0/7 |
| (0.95, 1.0] | 0.00% | 0/7019 |

## Per-Class Overconfidence
| Class | Total | Wrong | Overconfident Wrong | Overconfidence Rate |
|-------|-------|-------|---------------------|---------------------|
| LOW | 3082 | 0 | 0 | 0.00% |
| MODERATE | 7 | 0 | 0 | 0.00% |
| SEVERE | 3939 | 0 | 0 | 0.00% |

## Visualizations
- **Overconfidence Analysis**: `reports/figures/overconfidence_analysis.png`

## Key Findings
- No overconfident wrong predictions detected. Model is well-calibrated.
- The model shows appropriate uncertainty when making errors.

## Risk Assessment for Disaster Response
- **Low Risk**: Model is uncertain when wrong (can be caught by confidence threshold)
- **Medium Risk**: Some errors have moderate confidence (need human review)
- **High Risk**: Errors with >90% confidence (critical failures)
- **Critical**: Errors with >95% confidence (system must be retrained)

## Recommendations
1. No immediate action needed - model confidence is well-calibrated.
2. Continue monitoring during deployment.
3. Consider adding uncertainty estimation to the API responses.
