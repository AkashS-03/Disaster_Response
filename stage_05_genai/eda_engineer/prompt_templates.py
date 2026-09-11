import json
from typing import Dict, Any, List

SYSTEM_PROMPT_SCENARIO_ENGINE = """You are the Senior Crisis Simulation Architect for AquaShield Command.
Your mission is to synthesize physically grounded, highly realistic compound disaster scenarios that battle-test our autonomous AI response pipeline (Classical ML Risk Classifier, PyTorch LSTM Hydrological Forecaster, NLP Emergency Message Triage, and PEFT/LoRA SLM Tactical Voice Briefing).

PHYSICAL & OPERATIONAL INVARIANTS YOU MUST ENFORCE:
1. Hydrological Invariant: Upstream cloudbursts (>100 mm/hr) inevitably cause downstream gauge swells (+0.5m to +2.5m) within a 1-to-3 hour lag window.
2. Compound Concurrence: Disasters never strike in isolation. Pair extreme rainfall with infrastructure shocks (e.g. electrical substation explosion, bridge collapse, cellular tower battery depletion, blocked sea outfalls during astronomical high tide).
3. Tactical Lexicon Preservation: Include recognized emergency codes: 'PRI-1' (Priority 1 life hazard), 'MEDEVAC' (medical evacuation), 'CAS-EVAC' (casualty evacuation), 'LZ-CLEAR' (landing zone ready), 'SITREP' (situation report), 'CODE-RED' (imminent catastrophic failure).
4. Severity Conditioning:
   - LOW: Routine monitoring, clear corridors, nominal operations.
   - MODERATE: Waterlogging 0.3m-0.8m, road closures, precautionary evacuations.
   - SEVERE: Water >1.5m, rooftop strandings, power grid failure, critical life threats.
   - WILDCARD: Unprecedented compounding collapse (e.g., flash flood + total municipal grid blackout + hospital ICU generator inundation).
"""

FEW_SHOT_EXAMPLES = [
    {
        "scenario_id": "BENCH-01",
        "scenario_name": "Monsoon Estuarine Lockout & Mithi Surge",
        "hazard_type": "Cloudburst & Astronomical Tidal Lock",
        "severity": "SEVERE",
        "zone_id": "Zone_C",
        "narrative": "A sudden cloudburst of 145 mm/hr hits Kurla basin while a 4.6m astronomical spring high tide locks the Arabian Sea outfalls. Mithi River overflows into residential sectors with water rising 0.8m per hour.",
        "sensor_telemetry": {
            "river_level": 5.10,
            "rainfall": 145.0,
            "emergency_call_volume": 380,
            "road_closures": 14,
            "bridge_closures": 3,
            "river_level_trend": 1.25,
            "power_grid_status": "Degraded",
            "cellular_network_status": "Heavy Congestion"
        },
        "sos_message": "Water reached first floor in Kranti Nagar, 15 families stranded on roof with 4 infants. Power transformer sparked and went dark, send NDRF boats now!",
        "tactical_dispatch_log": "UNIT-7: Zone C Kurla Kranti Nagar water level 1.8m. 42 victims trapped on rooftops. PRI-1 MEDEVAC needed for hypothermic elderly. Roads impassable, deploy inflatable rescue craft immediately.",
        "key_factors": {
            "location": "Zone C Kurla",
            "people_count": 42,
            "risk_level": "SEVERE"
        }
    },
    {
        "scenario_id": "WILDCARD-01",
        "scenario_name": "Operation Blackout Deluge (The Midnight Grid Collapse)",
        "hazard_type": "Flash Flood + Total Municipal Grid Blackout + Hospital ICU Crisis",
        "severity": "SEVERE",
        "zone_id": "Zone_C",
        "narrative": "At 02:30 IST, an extreme 180 mm/hr deluge strikes Kurla/Sion. The main Dharavi 220kV transmission substation floods, causing a cascading electrical blackout across Zones B and C. Cellular towers run out of battery. Municipal General Hospital's basement backup diesel generators are submerged under 1.2m water, threatening 28 ICU patients on mechanical ventilators.",
        "sensor_telemetry": {
            "river_level": 5.75,
            "rainfall": 180.0,
            "emergency_call_volume": 490,
            "road_closures": 24,
            "bridge_closures": 5,
            "river_level_trend": 1.65,
            "power_grid_status": "Total Blackout",
            "cellular_network_status": "70% Towers Offline"
        },
        "sos_message": "ICU Ward 3 Mumbai General: Total power blackout, backup generators drowned in basement, ventilators on 20-min internal battery! 28 ICU patients suffocating, water 4 feet deep at hospital gate, SOS!",
        "tactical_dispatch_log": "COMMAND DISPATCH: PRI-1 MEDEVAC alert for Municipal Hospital Zone C. 28 ICU patients critical as floodwater submerged basement generators under total city blackout. Deploy dual amphibious rescue crafts and mobile emergency generators via CST Road corridor immediately.",
        "key_factors": {
            "location": "Zone C Kurla Municipal Hospital",
            "people_count": 28,
            "risk_level": "SEVERE"
        }
    }
]


class ScenarioPromptEngine:
    """Constructs structured, physics-constrained prompts for both LLM-driven

    and rule-conditioned synthetic scenario generation.
    """

    def __init__(self):
        self.system_prompt = SYSTEM_PROMPT_SCENARIO_ENGINE
        self.examples = FEW_SHOT_EXAMPLES

    def build_generation_prompt(self, hazard_type: str, severity: str, zone_id: str, compound_flags: List[str] = None) -> str:
        flags_str = ", ".join(compound_flags) if compound_flags else "None"
        prompt = f"""
GENERATE SYNTHETIC DISASTER SCENARIO:
- Primary Hazard: {hazard_type}
- Severity Tier: {severity}
- Target Zone: {zone_id}
- Compound Infrastructure Failures: {flags_str}

Please generate a JSON object matching this exact schema:
{{
  "scenario_id": "SYNTH-XXX",
  "scenario_name": "Concise Descriptive Title",
  "hazard_type": "{hazard_type}",
  "severity": "{severity}",
  "zone_id": "{zone_id}",
  "compound_failures": ["{flags_str}"],
  "narrative": "Rich 2-3 sentence description of physical dynamics and infrastructure cascade",
  "sensor_telemetry": {{
    "river_level": float,
    "rainfall": float,
    "emergency_call_volume": int,
    "road_closures": int,
    "bridge_closures": int,
    "river_level_trend": float,
    "power_grid_status": "Nominal|Degraded|Total Blackout",
    "cellular_network_status": "Operational|Degraded|Offline",
    "sensor_health": "Healthy|Intermittent|Corrupted_Zero"
  }},
  "sos_message": "Civilian emergency distress message for NLP triage",
  "tactical_dispatch_log": "Multi-unit field dispatch log with emergency codes (PRI-1, MEDEVAC, etc.) for SLM summarization",
  "expected_slm_briefing": "Exact severity-conditioned summary (<1 sent LOW, 1 sent MOD, 2 sent SEV)",
  "key_factors": {{
    "location": "Sector name",
    "people_count": int,
    "risk_level": "{severity}"
  }}
}}
"""
        return prompt.strip()
