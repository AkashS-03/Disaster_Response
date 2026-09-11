import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

from .telemetry_synthesizer import TelemetrySynthesizer
from .text_synthesizer import TextSynthesizer
from ..data_engineer.baseline_distributions import BaselineDistributions

ZONES = {
    "Zone_A": {"label": "South Mumbai (Low Risk Coastal Drainage)", "prob": 0.15},
    "Zone_B": {"label": "Bandra / Khar (Moderate Risk Catchment)", "prob": 0.65},
    "Zone_C": {"label": "Kurla / Sion (Critical Low-Lying Basin)", "prob": 0.85},
    "Zone_D": {"label": "Borivali / Dahisar (Suburban Stream Corridor)", "prob": 0.40}
}


class ScenarioGenerator:
    """Master Generative AI Scenario Orchestrator.

    Produces physically consistent, multi-zone compound disaster scenarios
    combining multi-point sensor values, 48h time-series sequences, civilian SOS
    text, and multi-unit incident logs to battle-test the entire pipeline.
    """

    def __init__(self, seed: Optional[int] = 42):
        self.telemetry = TelemetrySynthesizer(seed=seed)
        self.text_gen = TextSynthesizer(seed=seed)
        self.baseline = BaselineDistributions()
        self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    def generate_scenario(
        self,
        scenario_name: str,
        hazard_type: str,
        severity: str = "SEVERE",
        zone_id: str = "Zone_C",
        compound_blackout: bool = False,
        tidal_surge: bool = False,
        sensor_health: str = "Healthy",
        people_impact: int = 35,
        scenario_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generates a complete multi-modal compound disaster scenario."""
        sev = severity.upper()
        if not scenario_id:
            scenario_id = f"GEN-{np.random.randint(1000, 9999)}"

        # 1. Telemetry Synthesis (12 Tabular features for Stage 01 ML)
        tab_df = self.telemetry.generate_tabular_12_features(
            zone_id=zone_id,
            severity=sev,
            compound_blackout=compound_blackout,
            tidal_surge=tidal_surge,
            sensor_health=sensor_health
        )

        # 2. Time-Series Synthesis (48h multivariate for Stage 02 DL LSTM)
        river_val = float(tab_df['river_level'].iloc[0])
        rain_val = float(tab_df['rainfall'].iloc[0])
        calls_val = int(tab_df['emergency_call_volume'].iloc[0])
        closures = int(tab_df['total_infrastructure_closures'].iloc[0])

        ts_mode = "surge" if (sev in ["SEVERE", "WILDCARD"]) else ("steady_elevated" if sev == "MODERATE" else "receding")
        ts_48h = self.telemetry.generate_timeseries_48h(
            base_river=max(river_val, 1.5),
            base_rain=rain_val,
            base_calls=calls_val,
            closures=closures,
            scenario_type=ts_mode
        )

        # 3. Emergency Text Synthesis (Civilian SOS for Stage 03 NLP)
        sos_text = self.text_gen.generate_sos_message(
            zone_id=zone_id,
            severity=sev,
            compound_blackout=compound_blackout,
            people_count=people_impact
        )

        # 4. Tactical Dispatch Synthesis (Multi-unit log for Stage 04 SLM)
        dispatch_log = self.text_gen.generate_tactical_dispatch(
            zone_id=zone_id,
            severity=sev,
            compound_blackout=compound_blackout,
            people_count=people_impact
        )

        # 5. Compound Modifiers List
        compound_flags = []
        if compound_blackout:
            compound_flags.append("Total Municipal Grid Blackout")
        if tidal_surge:
            compound_flags.append("4.8m Astronomical Spring Tide Lock")
        if sensor_health == "Corrupted_Zero":
            compound_flags.append("Submerged Broken Gauge (0m Telemetry)")
        elif sensor_health == "Frozen_Gauge":
            compound_flags.append("Sensor Telemetry Freeze (Stuck 1.5m)")

        # 6. Physical Narrative Synthesis
        loc_label = ZONES.get(zone_id, {}).get("label", zone_id)
        if sev == "LOW":
            narrative = f"Nominal operations across {loc_label}. Minor precipitation detected ({rain_val:.1f} mm/hr), drainage corridors running clear with stable river gauge at {river_val:.2f}m."
        elif sev == "MODERATE":
            narrative = f"Sustained heavy precipitation ({rain_val:.1f} mm/hr) across {loc_label}. River gauge elevated to {river_val:.2f}m. Precautionary municipal pumps deployed and {closures} infrastructure closures active."
        elif sev == "SEVERE":
            compound_desc = f" coupled with {', '.join(compound_flags)}" if compound_flags else ""
            narrative = f"Severe cloudburst ({rain_val:.1f} mm/hr) triggering rapid river swell ({river_val:.2f}m){compound_desc} across {loc_label}. Floodwaters threatening ~{people_impact} residents with over {calls_val} emergency calls/hr."
        else:  # WILDCARD
            narrative = (
                f"CATASTROPHIC WILDCARD: Unprecedented {rain_val:.1f} mm/hr cloudburst coinciding with {', '.join(compound_flags)} "
                f"across {loc_label}. Municipal electrical substation inundated, leaving hospital emergency ICU generators submerged under water. "
                f"River gauge at {river_val:.2f}m with emergency calls surging past {calls_val}/hr."
            )

        drone_state = "FLOODED" if (sev in ["SEVERE", "WILDCARD"] or (sev == "MODERATE" and river_val >= 3.8)) else "CLEAR"

        return {
            "scenario_id": scenario_id,
            "scenario_name": scenario_name,
            "hazard_type": hazard_type,
            "severity": sev,
            "zone_id": zone_id,
            "zone_label": loc_label,
            "compound_modifiers": compound_flags,
            "narrative": narrative,
            "tabular_df": tab_df,
            "timeseries_48h": ts_48h,
            "sos_message": sos_text,
            "tactical_dispatch": dispatch_log,
            "people_impact": people_impact,
            "drone_recon_status": drone_state,
            "sensor_health": sensor_health,
            "expected_factors": {
                "location": loc_label.split(":")[1].split("(")[0].strip() if ":" in loc_label else zone_id,
                "people_count": people_impact,
                "risk_level": "SEVERE" if sev in ["SEVERE", "WILDCARD"] else sev
            }
        }
