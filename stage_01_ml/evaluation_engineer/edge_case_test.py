import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
def create_edge_case_scenarios():
    """
    Create borderline scenarios at decision boundaries.
    
    Decision Rules (from build_master_dataset.py):
    - SEVERE: river_level >= 4.5 OR (rainfall_72h >= 150 AND river_level >= 3.5)
    - MODERATE: river_level >= 3.0 OR rainfall_72h >= 80 OR emergency_calls >= 100
    - LOW: everything else
    
    We test just below and just above each boundary.
    """
    base_features = {
        'river_level': 2.0,
        'rainfall': 0.0,
        'emergency_call_volume': 10,
        'road_closures': 0,
        'bridge_closures': 0,
        'historical_flood_probability': 0.3,
        'river_level_rolling_72h_avg': 2.0,
        'rainfall_rolling_72h_sum': 50.0,
        'emergency_calls_24h_sum': 100.0,
        'total_infrastructure_closures': 0,
        'river_level_trend': 0.0,
        'zone_id_Zone_B': 0,
        'zone_id_Zone_C': 0,
        'zone_id_Zone_D': 0,
    }
    
    scenarios = []
    
    # --- SEVERE Boundary Tests ---
    # Rule: river_level >= 4.5 OR (rainfall_72h >= 150 AND river_level >= 3.5)
    
    # Test 1: river_level at 4.4 (should be MODERATE)
    s = base_features.copy()
    s['river_level'] = 4.4
    scenarios.append({**s, 'expected': 'MODERATE', 'description': 'River just below SEVERE threshold (4.4m)'})
    
    # Test 2: river_level at 4.5 (should be SEVERE)
    s = base_features.copy()
    s['river_level'] = 4.5
    scenarios.append({**s, 'expected': 'SEVERE', 'description': 'River at SEVERE threshold (4.5m)'})
    
    # Test 3: river_level at 4.6 (should be SEVERE)
    s = base_features.copy()
    s['river_level'] = 4.6
    scenarios.append({**s, 'expected': 'SEVERE', 'description': 'River just above SEVERE threshold (4.6m)'})
    
    # Test 4: Combined threshold - rainfall_72h=149, river=3.4 (should be MODERATE)
    s = base_features.copy()
    s['river_level'] = 3.4
    s['rainfall_rolling_72h_sum'] = 149.0
    scenarios.append({**s, 'expected': 'MODERATE', 'description': 'Combined just below SEVERE (river=3.4, rain72=149)'})
    
    # Test 5: Combined threshold - rainfall_72h=150, river=3.5 (should be SEVERE)
    s = base_features.copy()
    s['river_level'] = 3.5
    s['rainfall_rolling_72h_sum'] = 150.0
    scenarios.append({**s, 'expected': 'SEVERE', 'description': 'Combined at SEVERE threshold (river=3.5, rain72=150)'})
    
    # Test 6: Combined threshold - rainfall_72h=150, river=3.4 (should be MODERATE)
    s = base_features.copy()
    s['river_level'] = 3.4
    s['rainfall_rolling_72h_sum'] = 150.0
    scenarios.append({**s, 'expected': 'MODERATE', 'description': 'Rain meets but river doesn\'t (river=3.4, rain72=150)'})
    
    # --- MODERATE Boundary Tests ---
    # Rule: river_level >= 3.0 OR rainfall_72h >= 80 OR emergency_calls >= 100
    
    # Test 7: river_level at 2.9 (should be LOW)
    s = base_features.copy()
    s['river_level'] = 2.9
    scenarios.append({**s, 'expected': 'LOW', 'description': 'River just below MODERATE threshold (2.9m)'})
    
    # Test 8: river_level at 3.0 (should be MODERATE)
    s = base_features.copy()
    s['river_level'] = 3.0
    scenarios.append({**s, 'expected': 'MODERATE', 'description': 'River at MODERATE threshold (3.0m)'})
    
    # Test 9: river_level at 3.1 (should be MODERATE)
    s = base_features.copy()
    s['river_level'] = 3.1
    scenarios.append({**s, 'expected': 'MODERATE', 'description': 'River just above MODERATE threshold (3.1m)'})
    
    # Test 10: rainfall_72h at 79 (should be LOW)
    s = base_features.copy()
    s['rainfall_rolling_72h_sum'] = 79.0
    scenarios.append({**s, 'expected': 'LOW', 'description': 'Rain72 just below MODERATE threshold (79mm)'})
    
    # Test 11: rainfall_72h at 80 (should be MODERATE)
    s = base_features.copy()
    s['rainfall_rolling_72h_sum'] = 80.0
    scenarios.append({**s, 'expected': 'MODERATE', 'description': 'Rain72 at MODERATE threshold (80mm)'})
    
    # Test 12: emergency_calls at 99 (should be LOW)
    s = base_features.copy()
    s['emergency_call_volume'] = 99
    scenarios.append({**s, 'expected': 'LOW', 'description': 'Calls just below MODERATE threshold (99)'})
    
    # Test 13: emergency_calls at 100 (should be MODERATE)
    s = base_features.copy()
    s['emergency_call_volume'] = 100
    scenarios.append({**s, 'expected': 'MODERATE', 'description': 'Calls at MODERATE threshold (100)'})
    
    # --- Zone Variation Tests ---
    # Test 14: Same conditions, different zones
    for zone in ['Zone_B', 'Zone_C', 'Zone_D']:
        s = base_features.copy()
        s['river_level'] = 4.5
        s[f'zone_id_{zone}'] = 1
        scenarios.append({**s, 'expected': 'SEVERE', 'description': f'SEVERE in {zone} (river=4.5m)'})
    
    # Test 15: Zone_C with high historical probability
    s = base_features.copy()
    s['river_level'] = 3.2
    s['historical_flood_probability'] = 0.9
    s['zone_id_Zone_C'] = 1
    scenarios.append({**s, 'expected': 'MODERATE', 'description': 'Zone_C moderate with high historical prob'})
    
    return scenarios


def run_edge_case_tests():
    """
    Main edge case testing pipeline.
    Tests model behavior at decision boundaries.
    """
    print("--- Starting Edge Case Testing ---")
    
    import sys
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
    import safety_guard  # noqa: F401 – needed so joblib can unpickle GuardRailedPredictor
    model_path = os.path.join(base_dir, "models", "risk_model.joblib")
    fig_dir = os.path.join(base_dir, "reports", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    # 1. Load Model
    print("Loading model...")
    model = joblib.load(model_path)
    classes = model.classes_
    feature_names = model.feature_names_in_
    
    # 2. Create Edge Case Scenarios
    scenarios = create_edge_case_scenarios()
    print(f"Created {len(scenarios)} edge case scenarios")
    
    # 3. Run Predictions
    results = []
    for scenario in scenarios:
        description = scenario.pop('description')
        expected = scenario.pop('expected')
        
        # Create input DataFrame in the correct feature order
        input_df = pd.DataFrame([scenario])
        input_df = input_df.reindex(columns=feature_names)
        
        # Predict
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        confidence = float(np.max(probabilities))
        
        # Check if prediction matches expected
        correct = prediction == expected
        
        results.append({
            'description': description,
            'expected': expected,
            'predicted': prediction,
            'correct': correct,
            'confidence': confidence,
            'probabilities': {cls: float(prob) for cls, prob in zip(classes, probabilities)}
        })
    
    # 4. Analyze Results
    total = len(results)
    correct_count = sum(1 for r in results if r['correct'])
    accuracy = correct_count / total
    
    print(f"\n=== Edge Case Results ===")
    print(f"Total scenarios: {total}")
    print(f"Correct predictions: {correct_count}")
    print(f"Edge case accuracy: {accuracy:.2%}")
    
    # Group by boundary type
    severe_tests = [r for r in results if 'SEVERE' in r['description'] or 'severe' in r['description'].lower()]
    moderate_tests = [r for r in results if 'MODERATE' in r['description'] or 'moderate' in r['description'].lower()]
    zone_tests = [r for r in results if 'Zone' in r['description']]
    
    print(f"\nSEVERE boundary tests: {sum(1 for r in severe_tests if r['correct'])}/{len(severe_tests)} correct")
    print(f"MODERATE boundary tests: {sum(1 for r in moderate_tests if r['correct'])}/{len(moderate_tests)} correct")
    print(f"Zone variation tests: {sum(1 for r in zone_tests if r['correct'])}/{len(zone_tests)} correct")
    
    # 5. Print Failed Cases
    failed = [r for r in results if not r['correct']]
    if failed:
        print(f"\n=== FAILED EDGE CASES ({len(failed)}) ===")
        for f in failed:
            print(f"  {f['description']}")
            print(f"    Expected: {f['expected']}, Got: {f['predicted']} (Confidence: {f['confidence']:.2%})")
    else:
        print("\n=== ALL EDGE CASES PASSED ===")
    
    # 6. Visualization: Edge Case Results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Pass/Fail by category
    categories = ['SEVERE Boundary', 'MODERATE Boundary', 'Zone Variation']
    passed = [
        sum(1 for r in severe_tests if r['correct']),
        sum(1 for r in moderate_tests if r['correct']),
        sum(1 for r in zone_tests if r['correct'])
    ]
    failed_counts = [
        len(severe_tests) - passed[0],
        len(moderate_tests) - passed[1],
        len(zone_tests) - passed[2]
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    axes[0].bar(x - width/2, passed, width, label='Passed', color='green')
    axes[0].bar(x + width/2, failed_counts, width, label='Failed', color='red')
    axes[0].set_xlabel('Test Category')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Edge Case Pass/Fail by Category')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories, rotation=15)
    axes[0].legend()
    
    # Plot 2: Confidence distribution for edge cases
    confidences = [r['confidence'] for r in results]
    axes[1].hist(confidences, bins=15, color='steelblue', edgecolor='black')
    axes[1].set_xlabel('Prediction Confidence')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Confidence Distribution on Edge Cases')
    axes[1].axvline(x=0.5, color='red', linestyle='--', label='50% threshold')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'edge_case_results.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nVisualization saved to: {fig_dir}/edge_case_results.png")
    
    # 7. Generate Report
    report_content = f"""# Edge Case Testing Report

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
| Total Scenarios | {total} |
| Correct Predictions | {correct_count} |
| Edge Case Accuracy | {accuracy:.2%} |

## Category Breakdown
| Category | Passed | Total | Accuracy |
|----------|--------|-------|----------|
| SEVERE Boundary | {sum(1 for r in severe_tests if r['correct'])} | {len(severe_tests)} | {sum(1 for r in severe_tests if r['correct'])/len(severe_tests):.2%} |
| MODERATE Boundary | {sum(1 for r in moderate_tests if r['correct'])} | {len(moderate_tests)} | {sum(1 for r in moderate_tests if r['correct'])/len(moderate_tests):.2%} |
| Zone Variation | {sum(1 for r in zone_tests if r['correct'])} | {len(zone_tests)} | {sum(1 for r in zone_tests if r['correct'])/len(zone_tests):.2%} |

## Failed Cases
"""
    if failed:
        for f in failed:
            report_content += f"""
### {f['description']}
- **Expected**: {f['expected']}
- **Predicted**: {f['predicted']}
- **Confidence**: {f['confidence']:.2%}
- **Probabilities**: {f['probabilities']}
"""
    else:
        report_content += "\nNo failed edge cases. All boundary conditions correctly classified.\n"
    
    report_content += f"""
## Visualizations
- **Edge Case Results**: `reports/figures/edge_case_results.png`

## Key Findings
- {"All edge cases passed, indicating robust decision boundaries." if not failed else f"{len(failed)} edge cases failed, indicating potential issues at decision boundaries."}
- {"The model maintains consistent predictions across different zones." if len(zone_tests) == sum(1 for r in zone_tests if r['correct']) else "Zone variations show inconsistent predictions."}

## Implications for Disaster Response
- Decision boundaries are critical for emergency dispatch
- Failures at boundaries could lead to under/over-response
- Zone consistency ensures fair treatment across all areas
"""
    
    report_path = os.path.join(base_dir, "reports", "edge_case_report.md")
    with open(report_path, "w") as f:
        f.write(report_content)
    print(f"Edge case report saved to: {report_path}")
    
    print("--- Edge Case Testing Complete ---")
    return results


if __name__ == "__main__":
    run_edge_case_tests()
