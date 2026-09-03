import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation_engineer.calibration_test import calculate_brier_score
from evaluation_engineer.edge_case_test import create_edge_case_scenarios


class TestBrierScore:
    """Test Brier Score calculation."""
    
    def test_perfect_calibration(self):
        """Perfect predictions should have Brier Score = 0."""
        y_true = np.array(['LOW', 'LOW', 'SEVERE', 'SEVERE'])
        y_prob = np.array([
            [1.0, 0.0, 0.0],  # LOW with 100% confidence
            [1.0, 0.0, 0.0],  # LOW with 100% confidence
            [0.0, 0.0, 1.0],  # SEVERE with 100% confidence
            [0.0, 0.0, 1.0],  # SEVERE with 100% confidence
        ])
        classes = np.array(['LOW', 'MODERATE', 'SEVERE'])
        
        scores = calculate_brier_score(y_true, y_prob, classes)
        
        # Perfect predictions should have Brier Score = 0
        assert scores['LOW'] == 0.0
        assert scores['SEVERE'] == 0.0
        assert scores['macro_avg'] == 0.0
    
    def test_worst_calibration(self):
        """Worst predictions should have Brier Score = 1.0."""
        y_true = np.array(['LOW', 'LOW', 'SEVERE', 'SEVERE'])
        y_prob = np.array([
            [0.0, 0.0, 1.0],  # LOW predicted as SEVERE
            [0.0, 0.0, 1.0],  # LOW predicted as SEVERE
            [1.0, 0.0, 0.0],  # SEVERE predicted as LOW
            [1.0, 0.0, 0.0],  # SEVERE predicted as LOW
        ])
        classes = np.array(['LOW', 'MODERATE', 'SEVERE'])
        
        scores = calculate_brier_score(y_true, y_prob, classes)
        
        # All wrong with high confidence should be close to 1.0
        assert scores['LOW'] == 1.0
        assert scores['SEVERE'] == 1.0
    
    def test_bounded_between_0_and_1(self):
        """Brier Score should always be between 0 and 1."""
        y_true = np.array(['LOW', 'MODERATE', 'SEVERE'])
        y_prob = np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.6, 0.3],
            [0.2, 0.3, 0.5],
        ])
        classes = np.array(['LOW', 'MODERATE', 'SEVERE'])
        
        scores = calculate_brier_score(y_true, y_prob, classes)
        
        for cls, score in scores.items():
            assert 0.0 <= score <= 1.0, f"Brier Score for {cls} out of range: {score}"


class TestEdgeCases:
    """Test edge case scenario generation."""
    
    def test_scenarios_created(self):
        """Should create multiple edge case scenarios."""
        scenarios = create_edge_case_scenarios()
        assert len(scenarios) > 0
    
    def test_scenarios_have_required_keys(self):
        """Each scenario should have expected and description."""
        scenarios = create_edge_case_scenarios()
        for scenario in scenarios:
            assert 'expected' in scenario, "Scenario missing 'expected' key"
            assert 'description' in scenario, "Scenario missing 'description' key"
    
    def test_scenarios_have_valid_labels(self):
        """Expected labels should be LOW, MODERATE, or SEVERE."""
        valid_labels = {'LOW', 'MODERATE', 'SEVERE'}
        scenarios = create_edge_case_scenarios()
        for scenario in scenarios:
            assert scenario['expected'] in valid_labels, \
                f"Invalid label: {scenario['expected']}"
    
    def test_boundary_values_present(self):
        """Should include scenarios at decision boundaries."""
        scenarios = create_edge_case_scenarios()
        descriptions = [s['description'] for s in scenarios]
        
        # Check for key boundary scenarios
        assert any('4.4' in d or '4.5' in d for d in descriptions), \
            "Missing river_level boundary tests"
        assert any('3.0' in d for d in descriptions), \
            "Missing MODERATE threshold test"
    
    def test_zone_coverage(self):
        """Should test multiple zones."""
        scenarios = create_edge_case_scenarios()
        zone_tests = [s for s in scenarios if 'Zone' in s['description']]
        assert len(zone_tests) >= 3, "Should test at least 3 zones"


class TestFileStructure:
    """Test that all required files exist."""
    
    def test_evaluation_files_exist(self):
        """All evaluation module files should exist."""
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        eval_dir = os.path.join(base_dir, "evaluation_engineer")
        
        required_files = [
            '__init__.py',
            'evaluate_model.py',
            'calibration_test.py',
            'edge_case_test.py',
            'overconfidence_detector.py',
            'cross_validation.py',
        ]
        
        for file in required_files:
            file_path = os.path.join(eval_dir, file)
            assert os.path.exists(file_path), f"Missing file: {file}"
    
    def test_report_figures_directory(self):
        """Reports and figures directories should exist."""
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        reports_dir = os.path.join(base_dir, "reports")
        figures_dir = os.path.join(reports_dir, "figures")
        
        assert os.path.exists(reports_dir), "Reports directory missing"
        assert os.path.exists(figures_dir), "Figures directory missing"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
