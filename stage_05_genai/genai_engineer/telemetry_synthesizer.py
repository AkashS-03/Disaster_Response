import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

ZONES = {
    "Zone_A": {"label": "South Mumbai (Low Risk Coastal Drainage)", "prob": 0.15},
    "Zone_B": {"label": "Bandra / Khar (Moderate Risk Catchment)", "prob": 0.65},
    "Zone_C": {"label": "Kurla / Sion (Critical Low-Lying Basin)", "prob": 0.85},
    "Zone_D": {"label": "Borivali / Dahisar (Suburban Stream Corridor)", "prob": 0.40}
}


class TelemetrySynthesizer:
    """Generates physically consistent multi-point sensor readings and 48-hour

    time-series trajectories matching the exact input schemas of Stage 01 ML and
    Stage 02 DL LSTM, with optional adversarial corruption (e.g. dead gauge).
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def generate_tabular_12_features(
        self,
        zone_id: str = "Zone_C",
        severity: str = "SEVERE",
        compound_blackout: bool = False,
        tidal_surge: bool = False,
        sensor_health: str = "Healthy"
    ) -> pd.DataFrame:
        """Synthesizes the exact 12 features required by Stage 01 ML GuardRailedPredictor."""
        sev = severity.upper()
        zone_prob = ZONES.get(zone_id, ZONES["Zone_C"])["prob"]

        # Physics-constrained continuous parameters by severity tier
        if sev == "LOW":
            river = float(np.clip(self.rng.normal(1.8, 0.3), 0.8, 2.5))
            rain = float(np.clip(self.rng.exponential(8.0), 0.0, 35.0))
            calls = int(np.clip(self.rng.poisson(22), 5, 55))
            roads = int(self.rng.choice([0, 1], p=[0.85, 0.15]))
            bridges = 0
            trend = float(np.clip(self.rng.normal(0.02, 0.08), -0.5, 0.3))
        elif sev == "MODERATE":
            river = float(np.clip(self.rng.normal(3.5, 0.35), 3.0, 4.4))
            rain = float(np.clip(self.rng.normal(68.0, 14.0), 40.0, 95.0))
            calls = int(np.clip(self.rng.normal(115, 20), 70, 180))
            roads = int(self.rng.integers(2, 7))
            bridges = int(self.rng.integers(0, 3))
            trend = float(np.clip(self.rng.normal(0.45, 0.15), 0.1, 0.9))
        elif sev == "SEVERE":
            river = float(np.clip(self.rng.normal(5.1, 0.45), 4.5, 6.5))
            rain = float(np.clip(self.rng.normal(142.0, 25.0), 100.0, 210.0))
            calls = int(np.clip(self.rng.normal(340, 50), 220, 580))
            roads = int(self.rng.integers(8, 22))
            bridges = int(self.rng.integers(2, 6))
            trend = float(np.clip(self.rng.normal(1.20, 0.35), 0.6, 2.2))
        else:  # WILDCARD / CATASTROPHIC
            river = float(np.clip(self.rng.normal(5.8, 0.5), 5.2, 7.2))
            rain = float(np.clip(self.rng.normal(185.0, 20.0), 150.0, 250.0))
            calls = int(np.clip(self.rng.normal(480, 40), 380, 750))
            roads = int(self.rng.integers(18, 28))
            bridges = int(self.rng.integers(4, 8))
            trend = float(np.clip(self.rng.normal(1.85, 0.25), 1.2, 2.8))

        # Compound multipliers
        if tidal_surge:
            river += 0.45
            trend += 0.30
        if compound_blackout:
            calls = int(calls * 1.35)
            roads += 4

        # Cumulative rolling hydrology calculations (Physical consistency)
        rain_72 = float(np.clip(rain * self.rng.uniform(2.6, 3.8), 20.0, 750.0))
        calls_24 = float(np.clip(calls * self.rng.uniform(11.0, 16.0), 50.0, 2500.0))
        river_avg72 = float(max(0.8, river - self.rng.uniform(0.1, 0.5)))
        total_closures = roads + bridges

        # Adversarial Sensor Mode: Submerged dead gauge test
        true_river = river
        if sensor_health == "Corrupted_Zero":
            river = 0.0  # Physically broken gauge in deep water!
        elif sensor_health == "Frozen_Gauge":
            river = 1.5  # Stuck at baseline reading despite storm

        row = {
            'zone_id': zone_id,
            'river_level': float(river),
            'rainfall': float(rain),
            'emergency_call_volume': int(calls),
            'road_closures': int(roads),
            'bridge_closures': int(bridges),
            'historical_flood_probability': float(zone_prob),
            'river_level_rolling_72h_avg': float(river_avg72),
            'rainfall_rolling_72h_sum': float(rain_72),
            'emergency_calls_24h_sum': float(calls_24),
            'total_infrastructure_closures': int(total_closures),
            'river_level_trend': float(trend)
        }

        df = pd.DataFrame([row])
        df.attrs["true_river_level"] = true_river
        df.attrs["sensor_health"] = sensor_health
        return df

    def generate_timeseries_48h(
        self,
        base_river: float = 4.2,
        base_rain: float = 85.0,
        base_calls: int = 150,
        closures: int = 4,
        scenario_type: str = "surge"
    ) -> np.ndarray:
        """Generates a 48-hour multivariate sequence [river, rain, calls, closures]

        scaled to feed directly into Stage 02 PyTorch FloodLSTM.
        """
        timesteps = 48
        t = np.linspace(0, 1, timesteps)

        if scenario_type == "surge" or scenario_type == "flash_flood":
            # S-curve surge over 48 hours
            surge_profile = 1.0 / (1.0 + np.exp(-10 * (t - 0.6)))
            seq_river = (base_river * 0.4) + (base_river * 0.6 * surge_profile) + self.rng.normal(0, 0.04, timesteps)
            seq_rain = (base_rain * 0.2) + (base_rain * 0.8 * np.sin(t * np.pi)) + self.rng.normal(0, 3.0, timesteps)
            seq_calls = (base_calls * 0.3) + (base_calls * 0.7 * surge_profile) + self.rng.normal(0, 5.0, timesteps)
        elif scenario_type == "steady_elevated":
            seq_river = base_river + self.rng.normal(0, 0.08, timesteps)
            seq_rain = base_rain + self.rng.normal(0, 5.0, timesteps)
            seq_calls = base_calls + self.rng.normal(0, 8.0, timesteps)
        else:  # receding or low
            seq_river = np.linspace(base_river * 1.2, base_river * 0.7, timesteps) + self.rng.normal(0, 0.04, timesteps)
            seq_rain = np.maximum(0, np.linspace(base_rain, 5.0, timesteps) + self.rng.normal(0, 2.0, timesteps))
            seq_calls = np.linspace(base_calls, 20.0, timesteps) + self.rng.normal(0, 4.0, timesteps)

        seq_river = np.clip(seq_river, 0.5, 12.0)
        seq_rain = np.clip(seq_rain, 0.0, 250.0)
        seq_calls = np.clip(seq_calls, 0.0, 1000.0)
        seq_infra = np.full(timesteps, float(closures))

        return np.column_stack([seq_river, seq_rain, seq_calls, seq_infra])
