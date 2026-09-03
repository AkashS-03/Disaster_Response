# Edge Case Testing Report

## Overview
This report tests the model's behavior at critical decision boundaries.
These are the exact thresholds where risk classification changes.

## Decision Rules Being Tested
- **SEVERE**: river_level >= 4.5 OR (rainfall_72h >= 150 AND river_level >= 3.5)
- **MODERATE**: river_level >= 3.0 OR rainfall_72h >= 80 OR emergency_calls >= 100
- **LOW**: everything else

## Results Summary
| Metric | Value |
|--------|-------|
| Total Scenarios | 17 |
| Correct Predictions | 9 |
| Edge Case Accuracy | 52.94% |

## Category Breakdown
| Category | Passed | Total | Accuracy |
|----------|--------|-------|----------|
| SEVERE Boundary | 2 | 8 | 25.00% |
| MODERATE Boundary | 6 | 8 | 75.00% |
| Zone Variation | 1 | 4 | 25.00% |

## Failed Cases

### River at SEVERE threshold (4.5m)
- **Expected**: SEVERE
- **Predicted**: MODERATE
- **Confidence**: 67.00%
- **Probabilities**: {'LOW': 0.09, 'MODERATE': 0.67, 'SEVERE': 0.24}

### River just above SEVERE threshold (4.6m)
- **Expected**: SEVERE
- **Predicted**: MODERATE
- **Confidence**: 61.00%
- **Probabilities**: {'LOW': 0.08, 'MODERATE': 0.61, 'SEVERE': 0.31}

### Combined at SEVERE threshold (river=3.5, rain72=150)
- **Expected**: SEVERE
- **Predicted**: MODERATE
- **Confidence**: 85.00%
- **Probabilities**: {'LOW': 0.14, 'MODERATE': 0.85, 'SEVERE': 0.01}

### Rain72 just below MODERATE threshold (79mm)
- **Expected**: LOW
- **Predicted**: MODERATE
- **Confidence**: 60.00%
- **Probabilities**: {'LOW': 0.08, 'MODERATE': 0.6, 'SEVERE': 0.32}

### Calls just below MODERATE threshold (99)
- **Expected**: LOW
- **Predicted**: MODERATE
- **Confidence**: 51.00%
- **Probabilities**: {'LOW': 0.17, 'MODERATE': 0.51, 'SEVERE': 0.32}

### SEVERE in Zone_B (river=4.5m)
- **Expected**: SEVERE
- **Predicted**: MODERATE
- **Confidence**: 65.00%
- **Probabilities**: {'LOW': 0.1, 'MODERATE': 0.65, 'SEVERE': 0.25}

### SEVERE in Zone_C (river=4.5m)
- **Expected**: SEVERE
- **Predicted**: MODERATE
- **Confidence**: 66.00%
- **Probabilities**: {'LOW': 0.1, 'MODERATE': 0.66, 'SEVERE': 0.24}

### SEVERE in Zone_D (river=4.5m)
- **Expected**: SEVERE
- **Predicted**: MODERATE
- **Confidence**: 66.00%
- **Probabilities**: {'LOW': 0.1, 'MODERATE': 0.66, 'SEVERE': 0.24}

## Visualizations
- **Edge Case Results**: `reports/figures/edge_case_results.png`

## Key Findings
- 8 edge cases failed, indicating potential issues at decision boundaries.
- Zone variations show inconsistent predictions.

## Implications for Disaster Response
- Decision boundaries are critical for emergency dispatch
- Failures at boundaries could lead to under/over-response
- Zone consistency ensures fair treatment across all areas
