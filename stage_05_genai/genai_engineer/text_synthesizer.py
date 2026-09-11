import random
from typing import Dict, Any, Optional

TACTICAL_CODES = {
    "radio": ["SITREP", "PRI-1", "ROGER", "CODE-RED", "10-4"],
    "rescue": ["MEDEVAC", "CAS-EVAC", "EVAC-ORDER", "LZ-CLEAR", "SEARCH-RESCUE"],
    "logistics": ["WATER-PT", "RATION-DEP", "SHELTER-OPEN", "AMPHIB-UNIT"]
}

ZONE_LOCATIONS = {
    "Zone_A": ["South Mumbai Colaba Corridor", "Marine Drive Sea Wall", "Nariman Point Outfall"],
    "Zone_B": ["Bandra East Government Colony", "Khar Danda Slum Cluster", "BKC Bridge Lowland"],
    "Zone_C": ["Kurla Kranti Nagar Basin", "Sion Railway Culvert", "Mithi River Chunabhatti Bank"],
    "Zone_D": ["Borivali Dahisar Riverbank", "National Park Stream Ward", "Kandivali Link Drainage"]
}


class TextSynthesizer:
    """Synthesizes paired crisis text logs:

    1. Civilian SOS messages tailored to battle-test Stage 03 NLP Triage.
    2. Multi-unit tactical dispatch logs embedded with domain dictionary codes
       tailored for Stage 04 PEFT/LoRA SLM voice briefing synthesis.
    """

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

    def generate_sos_message(
        self,
        zone_id: str = "Zone_C",
        severity: str = "SEVERE",
        compound_blackout: bool = False,
        people_count: int = 25
    ) -> str:
        """Generates realistic civilian distress text for Stage 03 NLP Triage."""
        loc = random.choice(ZONE_LOCATIONS.get(zone_id, ZONE_LOCATIONS["Zone_C"]))
        sev = severity.upper()

        if sev == "LOW":
            templates = [
                f"Market reopened in {loc}, storm drains are working normally and buses are running on schedule.",
                f"Minor puddles near {loc}, light rain has stopped and shops are open as usual.",
                f"Traffic moving smoothly past {loc}, rainfall was light overnight and no waterlogging observed."
            ]
            return random.choice(templates)

        elif sev == "MODERATE":
            templates = [
                f"Flood water entered ground floor homes near {loc}. Municipal shelter opened, residents evacuating slowly.",
                f"Waterlogging of 2 feet along main road in {loc}. Traffic blocked, pumps active, need municipal assistance.",
                f"Monsoon runoff overflowing storm drain at {loc}. Approximately {people_count} residents moving valuables upstairs."
            ]
            return random.choice(templates)

        elif sev == "SEVERE":
            blackout_phrase = " Transformer exploded, complete darkness and flood water rising past chest level." if compound_blackout else " Flood water reached ceiling level and current is extremely violent."
            templates = [
                f"CRITICAL: {people_count} people trapped on roof near {loc}. Water rising fast, two children injured and need urgent rescue!{blackout_phrase} Send emergency boats immediately!",
                f"Emergency SOS! Ground floor completely submerged in {loc}, {people_count} residents stranded without drinking water.{blackout_phrase} Wall collapsed nearby, need NDRF search and rescue now!",
                f"Urgent help needed at {loc}! {people_count} victims screaming on terrace as flood waters breach second floor.{blackout_phrase} Elderly patient having breathing trouble, send medical team!"
            ]
            return random.choice(templates)

        else:  # WILDCARD: Operation Blackout Deluge
            return (
                f"MAYDAY MAYDAY! {loc} General Hospital ICU basement flooded under 4 feet of dark storm water! "
                f"Complete city blackout, backup diesel generators completely submerged and failed! "
                f"{people_count} ICU patients on mechanical ventilators have only 15 minutes of internal battery reserve! "
                f"Doctors hand-pumping oxygen in pitch black water, send amphibious rescue and portable generators NOW!"
            )

    def generate_tactical_dispatch(
        self,
        zone_id: str = "Zone_C",
        severity: str = "SEVERE",
        compound_blackout: bool = False,
        people_count: int = 25
    ) -> str:
        """Generates structured multi-unit incident dispatch log for Stage 04 SLM."""
        loc = random.choice(ZONE_LOCATIONS.get(zone_id, ZONE_LOCATIONS["Zone_C"]))
        sev = severity.upper()

        if sev == "LOW":
            return (
                f"UNIT-1: Patrol report from {loc}. Gauge reading 1.8m, storm drainage nominal. "
                f"SITREP: All roads clear, zero structural damage. 10-4, returning to standard sector watch."
            )

        elif sev == "MODERATE":
            return (
                f"UNIT-4: Staging near {loc}. Road closure in effect due to 0.6m street inundation. "
                f"SITREP: Approximately {people_count} residents relocating to primary municipal shelter. "
                f"WATER-PT established at sector hall, pumps deployed, monitoring runoff levels."
            )

        elif sev == "SEVERE":
            blackout_line = " Substation outage confirmed, total grid blackout across sector." if compound_blackout else ""
            return (
                f"UNIT-9: Emergency dispatch to {loc}. Mithi River overflow breached levee barrier.{blackout_line} "
                f"SITREP: {people_count} civilians trapped on building rooftops with water at 2.1 meters. "
                f"PRI-1 MEDEVAC required for 3 hypothermic casualties. Roads impassable, deploy motorized rescue rafts and LZ-CLEAR at railway overpass immediately."
            )

        else:  # WILDCARD: Operation Blackout Deluge
            return (
                f"COMMAND DISPATCH: CODE-RED disaster declaration for {loc}. "
                f"Primary electrical substation flooded causing total municipal blackout. "
                f"Hospital basement inundated, emergency generators submerged under 1.2m water. "
                f"PRI-1 MEDEVAC: {people_count} ICU patients facing immediate ventilator failure. "
                f"Direct directive: Dispatch amphibious rescue units with mobile generator packs to CST Road immediately."
            )
