import os
from typing import Dict, Any, List, Optional
from ..genai_engineer.scenario_generator import ScenarioGenerator


BENCHMARK_20_SPECS = [
    # Regime 1: Extreme Tail Events (500-Year Return Period)
    {
        "id": "STRESS-01",
        "name": "500-Year Atmospheric River Cloudburst",
        "regime": "Extreme Tail Events",
        "hazard_type": "Hyper-Cloudburst (210 mm/hr)",
        "severity": "SEVERE",
        "zone_id": "Zone_C",
        "compound_blackout": False,
        "tidal_surge": False,
        "sensor_health": "Healthy",
        "people_impact": 120
    },
    {
        "id": "STRESS-02",
        "name": "Upstream Glacial Sieve / Dam Overtopping",
        "regime": "Extreme Tail Events",
        "hazard_type": "Catastrophic Dam Spillway Surge",
        "severity": "SEVERE",
        "zone_id": "Zone_D",
        "compound_blackout": False,
        "tidal_surge": False,
        "sensor_health": "Healthy",
        "people_impact": 85
    },
    {
        "id": "STRESS-03",
        "name": "Record Estuarine River Crest (6.2m)",
        "regime": "Extreme Tail Events",
        "hazard_type": "Historical River Gauge Overtopping",
        "severity": "SEVERE",
        "zone_id": "Zone_C",
        "compound_blackout": False,
        "tidal_surge": False,
        "sensor_health": "Healthy",
        "people_impact": 140
    },
    {
        "id": "STRESS-04",
        "name": "Super-Monsoon Prolonged Gale & Saturation",
        "regime": "Extreme Tail Events",
        "hazard_type": "Tropical Cyclone Inflow Deluge",
        "severity": "SEVERE",
        "zone_id": "Zone_B",
        "compound_blackout": False,
        "tidal_surge": False,
        "sensor_health": "Healthy",
        "people_impact": 65
    },
    {
        "id": "STRESS-05",
        "name": "Dual-Basin Concurrent Inundation",
        "regime": "Extreme Tail Events",
        "hazard_type": "Multi-Ward Synchronous Crest",
        "severity": "SEVERE",
        "zone_id": "Zone_C",
        "compound_blackout": False,
        "tidal_surge": False,
        "sensor_health": "Healthy",
        "people_impact": 180
    },

    # Regime 2: Multi-Zone Compound Cascades
    {
        "id": "STRESS-06",
        "name": "Cloudburst with Municipal Power Grid Collapse",
        "regime": "Compound Cascades",
        "hazard_type": "Flood + Total Electrical Blackout",
        "severity": "SEVERE",
        "zone_id": "Zone_C",
        "compound_blackout": True,
        "tidal_surge": False,
        "sensor_health": "Healthy",
        "people_impact": 95
    },
    {
        "id": "STRESS-07",
        "name": "Astronomical Spring Tide Lockout",
        "regime": "Compound Cascades",
        "hazard_type": "4.9m Spring Tide + Outfall Blockage",
        "severity": "SEVERE",
        "zone_id": "Zone_A",
        "compound_blackout": False,
        "tidal_surge": True,
        "sensor_health": "Healthy",
        "people_impact": 50
    },
    {
        "id": "STRESS-08",
        "name": "Arterial Highway Bridge Scour & Cutoff",
        "regime": "Compound Cascades",
        "hazard_type": "Structural Bridge Failure + High Water",
        "severity": "SEVERE",
        "zone_id": "Zone_B",
        "compound_blackout": False,
        "tidal_surge": False,
        "sensor_health": "Healthy",
        "people_impact": 40
    },
    {
        "id": "STRESS-09",
        "name": "Industrial Runoff & Chemical Containment Breach",
        "regime": "Compound Cascades",
        "hazard_type": "Toxic Flood Runoff Contamination",
        "severity": "SEVERE",
        "zone_id": "Zone_D",
        "compound_blackout": False,
        "tidal_surge": False,
        "sensor_health": "Healthy",
        "people_impact": 75
    },
    {
        "id": "STRESS-10",
        "name": "Triple Infrastructure Domino Cascade",
        "regime": "Compound Cascades",
        "hazard_type": "Blackout + Bridge Collapse + Flood",
        "severity": "SEVERE",
        "zone_id": "Zone_C",
        "compound_blackout": True,
        "tidal_surge": True,
        "sensor_health": "Healthy",
        "people_impact": 210
    },

    # Regime 3: Adversarial & Telemetry Degraded Stress
    {
        "id": "STRESS-11",
        "name": "Submerged Dead Gauge (0.0m Telemetry)",
        "regime": "Adversarial Telemetry",
        "hazard_type": "Gauge Short-Circuit During Peak Deluge",
        "severity": "SEVERE",
        "zone_id": "Zone_C",
        "compound_blackout": False,
        "tidal_surge": False,
        "sensor_health": "Corrupted_Zero",
        "people_impact": 110
    },
    {
        "id": "STRESS-12",
        "name": "Frozen Sensor Telemetry (Stuck 1.5m)",
        "regime": "Adversarial Telemetry",
        "hazard_type": "Gauging Station Stale Telemetry Loop",
        "severity": "SEVERE",
        "zone_id": "Zone_B",
        "compound_blackout": False,
        "tidal_surge": False,
        "sensor_health": "Frozen_Gauge",
        "people_impact": 80
    },
    {
        "id": "STRESS-13",
        "name": "Emergency 911 Call Swarm Saturation",
        "regime": "Adversarial Telemetry",
        "hazard_type": "600+ Calls/hr Network Overload",
        "severity": "SEVERE",
        "zone_id": "Zone_C",
        "compound_blackout": True,
        "tidal_surge": False,
        "sensor_health": "Healthy",
        "people_impact": 130
    },
    {
        "id": "STRESS-14",
        "name": "Upstream/Downstream Gauge Inversion",
        "regime": "Adversarial Telemetry",
        "hazard_type": "Deceptive Upstream Dry Reading with Local Surge",
        "severity": "MODERATE",
        "zone_id": "Zone_B",
        "compound_blackout": False,
        "tidal_surge": False,
        "sensor_health": "Frozen_Gauge",
        "people_impact": 35
    },
    {
        "id": "STRESS-15",
        "name": "Adversarial Low-Water Sensor in Severe Call Flood",
        "regime": "Adversarial Telemetry",
        "hazard_type": "Sensor Desynchronization",
        "severity": "SEVERE",
        "zone_id": "Zone_C",
        "compound_blackout": False,
        "tidal_surge": False,
        "sensor_health": "Corrupted_Zero",
        "people_impact": 90
    },

    # Regime 4: Rapid Dynamic Escalation & Edge Conditions
    {
        "id": "STRESS-16",
        "name": "Flash Inundation (+2.5m River Surge in 45m)",
        "regime": "Dynamic Escalation",
        "hazard_type": "Violent Cloudburst Runoff Spike",
        "severity": "SEVERE",
        "zone_id": "Zone_C",
        "compound_blackout": False,
        "tidal_surge": False,
        "sensor_health": "Healthy",
        "people_impact": 70
    },
    {
        "id": "STRESS-17",
        "name": "Midnight Slum Encroachment Breach",
        "regime": "Dynamic Escalation",
        "hazard_type": "Informal Settlement Levee Overtopping",
        "severity": "SEVERE",
        "zone_id": "Zone_C",
        "compound_blackout": True,
        "tidal_surge": False,
        "sensor_health": "Healthy",
        "people_impact": 250
    },
    {
        "id": "STRESS-18",
        "name": "Deceptive Eye-of-Storm Receding Trap",
        "regime": "Dynamic Escalation",
        "hazard_type": "Temporary Gauge Drop Prior to Surge Front",
        "severity": "MODERATE",
        "zone_id": "Zone_B",
        "compound_blackout": False,
        "tidal_surge": False,
        "sensor_health": "Healthy",
        "people_impact": 25
    },
    {
        "id": "STRESS-19",
        "name": "Coastal Ward Lowland Back-Siphonage",
        "regime": "Dynamic Escalation",
        "hazard_type": "Seawall Siphon Reversal Inflow",
        "severity": "MODERATE",
        "zone_id": "Zone_A",
        "compound_blackout": False,
        "tidal_surge": True,
        "sensor_health": "Healthy",
        "people_impact": 45
    },
    {
        "id": "STRESS-20",
        "name": "Critical Metro / Railway Culvert Submersion",
        "regime": "Dynamic Escalation",
        "hazard_type": "Transit Hub Complete Flooding",
        "severity": "SEVERE",
        "zone_id": "Zone_C",
        "compound_blackout": True,
        "tidal_surge": False,
        "sensor_health": "Healthy",
        "people_impact": 160
    }
]

WILDCARD_CAPSTONE_SPEC = {
    "id": "WILDCARD-CAPSTONE",
    "name": "Operation Blackout Deluge (The Midnight Grid Failure & ICU Crisis)",
    "regime": "Capstone Wildcard",
    "hazard_type": "Flash Flood + Total Municipal Grid Blackout + Hospital ICU Crisis",
    "severity": "WILDCARD",
    "zone_id": "Zone_C",
    "compound_blackout": True,
    "tidal_surge": True,
    "sensor_health": "Corrupted_Zero",  # Gauge broken by transformer surge!
    "people_impact": 28
}


class StressTestBattery:
    """Manages the generation and cataloging of the 20 benchmark stress-test

    events and the Wildcard Capstone challenge.
    """

    def __init__(self):
        self.generator = ScenarioGenerator(seed=1337)
        self.benchmark_specs = BENCHMARK_20_SPECS
        self.wildcard_spec = WILDCARD_CAPSTONE_SPEC

    def generate_all_benchmark_events(self) -> List[Dict[str, Any]]:
        events = []
        for spec in self.benchmark_specs:
            scen = self.generator.generate_scenario(
                scenario_name=spec["name"],
                hazard_type=spec["hazard_type"],
                severity=spec["severity"],
                zone_id=spec["zone_id"],
                compound_blackout=spec["compound_blackout"],
                tidal_surge=spec["tidal_surge"],
                sensor_health=spec["sensor_health"],
                people_impact=spec["people_impact"],
                scenario_id=spec["id"]
            )
            scen["regime"] = spec["regime"]
            events.append(scen)
        return events

    def generate_wildcard_event(self) -> Dict[str, Any]:
        scen = self.generator.generate_scenario(
            scenario_name=self.wildcard_spec["name"],
            hazard_type=self.wildcard_spec["hazard_type"],
            severity=self.wildcard_spec["severity"],
            zone_id=self.wildcard_spec["zone_id"],
            compound_blackout=self.wildcard_spec["compound_blackout"],
            tidal_surge=self.wildcard_spec["tidal_surge"],
            sensor_health=self.wildcard_spec["sensor_health"],
            people_impact=self.wildcard_spec["people_impact"],
            scenario_id=self.wildcard_spec["id"]
        )
        scen["regime"] = self.wildcard_spec["regime"]
        return scen
