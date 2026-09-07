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
| Correct Predictions | 17 |
| Edge Case Accuracy | 100.00% |

## Category Breakdown
| Category | Passed | Total | Accuracy |
|----------|--------|-------|----------|
| SEVERE Boundary | 8 | 8 | 100.00% |
| MODERATE Boundary | 8 | 8 | 100.00% |
| Zone Variation | 4 | 4 | 100.00% |

## Failed Cases

No failed edge cases. All boundary conditions correctly classified.

## Visualizations
- **Edge Case Results**: `reports/figures/edge_case_results.png`

## Key Findings
- All edge cases passed, indicating robust decision boundaries.
- The model maintains consistent predictions across different zones.

## Implications for Disaster Response
- Decision boundaries are critical for emergency dispatch
- Failures at boundaries could lead to under/over-response
- Zone consistency ensures fair treatment across all areas
