import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HISTORICAL_SEEDS_PATH = os.path.join(os.path.dirname(__file__), "historical_seeds.json")
MASTER_DATASET_PATH = os.path.join(os.path.dirname(BASE_DIR), "stage_01_ml", "data", "processed", "master_dataset.csv")

ZONES = {
    "Zone_A": {"label": "South Mumbai (Low Risk Coastal Drainage)", "prob": 0.15},
    "Zone_B": {"label": "Bandra / Khar (Moderate Risk Catchment)", "prob": 0.65},
    "Zone_C": {"label": "Kurla / Sion (Critical Low-Lying Basin)", "prob": 0.85},
    "Zone_D": {"label": "Borivali / Dahisar (Suburban Stream Corridor)", "prob": 0.40}
}


class BaselineDistributions:
    """Computes empirical distributions, covariance structures, and extreme value parameters

    from historical disaster data to serve as statistical priors for generative synthesis.
    """

    def __init__(self, seeds_path: str = HISTORICAL_SEEDS_PATH, dataset_path: str = MASTER_DATASET_PATH):
        self.seeds_path = seeds_path
        self.dataset_path = dataset_path
        self.seeds_data = self._load_seeds()
        self.stats = self._compute_or_load_statistics()

    def _load_seeds(self) -> Dict[str, Any]:
        if os.path.exists(self.seeds_path):
            with open(self.seeds_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"seeds": {}, "distribution_priors": {}}

    def _compute_or_load_statistics(self) -> Dict[str, Any]:
        """Loads master_dataset.csv to calculate real empirical means, stds, and covariance."""
        stats = {
            "by_severity": {},
            "covariance_matrix": None,
            "feature_cols": [
                'river_level', 'rainfall', 'emergency_call_volume', 'road_closures',
                'bridge_closures', 'historical_flood_probability', 'river_level_rolling_72h_avg',
                'rainfall_rolling_72h_sum', 'emergency_calls_24h_sum', 'total_infrastructure_closures',
                'river_level_trend'
            ]
        }

        if os.path.exists(self.dataset_path):
            try:
                df = pd.read_csv(self.dataset_path)
                for sev in ["LOW", "MODERATE", "SEVERE"]:
                    sub = df[df['risk_label'] == sev]
                    if len(sub) > 0:
                        stats["by_severity"][sev] = {
                            "river_mean": float(sub['river_level'].mean()),
                            "river_std": float(sub['river_level'].std()),
                            "rain_mean": float(sub['rainfall'].mean()),
                            "rain_std": float(sub['rainfall'].std()),
                            "calls_mean": float(sub['emergency_call_volume'].mean()),
                            "calls_std": float(sub['emergency_call_volume'].std()),
                            "road_mean": float(sub['road_closures'].mean()),
                            "bridge_mean": float(sub['bridge_closures'].mean())
                        }
                numeric_cols = [c for c in stats["feature_cols"] if c in df.columns]
                cov = df[numeric_cols].cov().to_numpy()
                stats["covariance_matrix"] = cov
                return stats
            except Exception:
                pass

        # Robust parametric fallback from historical_seeds.json
        priors = self.seeds_data.get("distribution_priors", {})
        stats["by_severity"] = {
            "LOW": {
                "river_mean": priors.get("river_level", {}).get("normal_mean", 2.1),
                "river_std": priors.get("river_level", {}).get("normal_std", 0.5),
                "rain_mean": priors.get("rainfall", {}).get("normal_mean", 12.0),
                "rain_std": priors.get("rainfall", {}).get("normal_std", 8.0),
                "calls_mean": priors.get("emergency_call_volume", {}).get("normal_mean", 25),
                "calls_std": priors.get("emergency_call_volume", {}).get("normal_std", 10),
                "road_mean": 0.5, "bridge_mean": 0.1
            },
            "MODERATE": {
                "river_mean": priors.get("river_level", {}).get("moderate_mean", 3.6),
                "river_std": priors.get("river_level", {}).get("moderate_std", 0.4),
                "rain_mean": priors.get("rainfall", {}).get("moderate_mean", 65.0),
                "rain_std": priors.get("rainfall", {}).get("moderate_std", 15.0),
                "calls_mean": priors.get("emergency_call_volume", {}).get("moderate_mean", 110),
                "calls_std": priors.get("emergency_call_volume", {}).get("moderate_std", 25),
                "road_mean": 4.0, "bridge_mean": 1.0
            },
            "SEVERE": {
                "river_mean": priors.get("river_level", {}).get("severe_mean", 4.9),
                "river_std": priors.get("river_level", {}).get("severe_std", 0.6),
                "rain_mean": priors.get("rainfall", {}).get("severe_mean", 135.0),
                "rain_std": priors.get("rainfall", {}).get("severe_std", 30.0),
                "calls_mean": priors.get("emergency_call_volume", {}).get("severe_mean", 320),
                "calls_std": priors.get("emergency_call_volume", {}).get("severe_std", 60),
                "road_mean": 12.0, "bridge_mean": 3.5
            }
        }
        return stats

    def get_seed(self, seed_key: str) -> Optional[Dict[str, Any]]:
        return self.seeds_data.get("seeds", {}).get(seed_key)

    def list_seeds(self) -> Dict[str, str]:
        return {k: v.get("name", k) for k, v in self.seeds_data.get("seeds", {}).items()}

    def get_priors(self, severity: str = "SEVERE") -> Dict[str, Any]:
        sev = severity.upper()
        return self.stats["by_severity"].get(sev, self.stats["by_severity"].get("SEVERE"))
