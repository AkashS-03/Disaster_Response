# Calibration Analysis Report

## Overview
This report evaluates whether the model's probability predictions are trustworthy.
A well-calibrated model means when it predicts "80% SEVERE", that event actually occurs ~80% of the time.

## Brier Score (Lower = Better, 0.0 = Perfect)
| Class | Brier Score |
|-------|-------------|
| LOW | 0.000005 |
| MODERATE | 0.000009 |
| SEVERE | 0.000004 |
| macro_avg | 0.000006 |

### Interpretation
- **0.0 - 0.1**: Excellent calibration
- **0.1 - 0.2**: Good calibration  
- **0.2 - 0.3**: Moderate calibration
- **0.3+**: Poor calibration

## Confidence Distribution Statistics
| Metric | Value |
|--------|-------|
| Mean confidence (correct predictions) | 0.9999 |
| Mean confidence (incorrect predictions) | N/A (no errors) |
| Min confidence (correct) | 0.8800 |
| Max confidence (incorrect) | N/A (no errors) |

## Visualizations
- **Reliability Diagram**: `reports/figures/calibration_reliability_diagram.png`
- **Confidence Distribution**: `reports/figures/confidence_distribution.png`

## Key Findings
- The Brier scores indicate excellent overall calibration.
- The model is highly confident in its predictions, but this confidence should be validated against real-world outcomes.

## Implications for Disaster Response
- **High Brier scores** mean probability outputs cannot be trusted for decision-making
- **Well-calibrated probabilities** allow threshold-based dispatch decisions
- If probabilities are unreliable, the system should rely on hard classifications only
