"""
Stage 04 - SLM | Integration Engineer: Tactical Briefing Assistant Wrapper
==========================================================================
Exposes the fine-tuned TacticalBriefingSLM to the incident commander dashboard.
Provides:
  - generate_briefing(incident_log): 2-sentence tactical summary with latency tracking
  - extract_tactical_chips(text): identifies tactical codes for UI HUD badges
  - compute_time_savings(log_text, briefing_text): Team Huddle reading time calculator
  - Curated crisis incident presets for live edge demonstration
"""

import os
import sys
import time
import torch

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from data_engineer.slm_utils import (  # noqa: E402
    load_meta, text_to_ids, ids_to_text, pad_sequence, tokenize,
    get_domain_tokens, load_domain_dictionary, format_two_sentences,
    MAX_SRC_LEN, MAX_TGT_LEN
)
from dl_engineer.slm_model import build_model  # noqa: E402

MODELS_DIR = os.path.join(base_dir, "models")

# Field Presets for Instant Incident Commander Demo (Dense Multi-Report Logs ~140-160 words)
DEMO_PRESETS = {
    "🌊 Flash Flood Isolation (Sector 3 Delta)": (
        "=== INCIDENT LOG DISPATCH | Sector 3 Delta | GRID: GR-482/DELTA ===\n"
        "[T+025m / Field-Unit-01 / SEV:SEVERE]: Rapid flood waters rising over 2 meters near Delta Bridge; 18 civilians stranded on church rooftop.\n"
        "[T+030m / Field-Unit-02 / SEV:SEVERE]: Current is too swift for conventional rubber dinghies; water entering residential floor levels rapidly.\n"
        "[T+035m / Field-Unit-03 / SEV:MODERATE]: Local highway access cut off; primary roadway completely inundated with flash debris and downed power lines.\n"
        "[T+040m / Field-Unit-04 / SEV:SEVERE]: Urgent medical evacuation needed for elderly casualties suffering hypothermia, shock, and dehydration.\n"
        "[T+045m / Field-Unit-05 / SEV:MODERATE]: Potable drinking water supply contaminated by river silt; sanitation supplies exhausted across ward.\n"
        "[T+050m / Field-Unit-06 / SEV:SEVERE]: Flash flood surge continuing downstream toward school shelter; immediate aerial extraction requested."
    ),
    "🏚️ Post-Quake Structural Collapse (Sector 1 North)": (
        "=== INCIDENT LOG DISPATCH | Sector 1 North | GRID: GR-719/ALPHA ===\n"
        "[T+045m / Field-Unit-01 / SEV:SEVERE]: 3-story concrete residential structure collapsed; rhythmic sounds of tapping heard from ground basement level.\n"
        "[T+050m / Field-Unit-02 / SEV:SEVERE]: Estimated 12 victims trapped under secondary rubble beams; building structural integrity severely compromised.\n"
        "[T+055m / Field-Unit-03 / SEV:MODERATE]: High-pressure natural gas leak detected in adjoining alleyway; heavy smell of methane and airborne dust.\n"
        "[T+060m / Field-Unit-04 / SEV:SEVERE]: Specialized heavy lifting cranes, pneumatic shoring struts, and acoustic search teams required on site immediately.\n"
        "[T+065m / Field-Unit-05 / SEV:SEVERE]: Secondary aftershocks measuring magnitude 4.8 triggered minor wall collapses on eastern perimeter.\n"
        "[T+070m / Field-Unit-06 / SEV:MODERATE]: Local clinic damaged; temporary triage staging post requested on open soccer field north of sector."
    ),
    "☣️ Chemical Hazmat Ingress (Coastal Ward)": (
        "=== INCIDENT LOG DISPATCH | Coastal Ward | GRID: GR-933/CHARLIE ===\n"
        "[T+010m / Field-Unit-01 / SEV:SEVERE]: Industrial chemical storage tank ruptured following storm surge; yellow-green vapor cloud drifting southeast.\n"
        "[T+015m / Field-Unit-02 / SEV:SEVERE]: Multiple residents reporting acute respiratory distress, severe ocular burning, nausea, and disorientation.\n"
        "[T+020m / Field-Unit-03 / SEV:MODERATE]: Wind speed 15 knots blowing directly toward coastal civilian evacuation shelter housing 400 evacuees.\n"
        "[T+025m / Field-Unit-04 / SEV:SEVERE]: Immediate 2-kilometer perimeter cordon required; deploy hazmat decontamination units and vertical shelter directives.\n"
        "[T+030m / Field-Unit-05 / SEV:MODERATE]: Local access avenue blocked by stalled transport vehicles; traffic control units dispatched.\n"
        "[T+035m / Field-Unit-06 / SEV:SEVERE]: Toxic vapor concentration rising near drainage canal; mandatory respirator directive issued for all personnel."
    )
}


class TacticalBriefingAssistant:
    """Lazy-loaded Tactical Briefing SLM engine for edge command deployment."""

    def __init__(self, models_dir=MODELS_DIR):
        self.models_dir = models_dir
        self.model = None
        self.meta = None
        self.src_vocab = None
        self.tgt_vocab = None
        self.inv_tgt = None
        self.domain_tokens = set(get_domain_tokens())
        self.domain_dict = load_domain_dictionary()

    def _load(self):
        if self.model is not None:
            return
        meta_path = os.path.join(self.models_dir, "slm_briefing_meta.json")
        weights_path = os.path.join(self.models_dir, "slm_briefing.pth")
        
        if not (os.path.exists(meta_path) and os.path.exists(weights_path)):
            raise FileNotFoundError("Tactical Briefing SLM weights not found. Run slm_train.py first.")
            
        self.meta = load_meta(meta_path)
        self.src_vocab = self.meta["src_vocab"]
        self.tgt_vocab = self.meta["tgt_vocab"]
        self.inv_tgt = {v: k for k, v in self.tgt_vocab.items()}
        
        self.model = build_model(self.meta, weights_path)
        self.model.eval()

    @staticmethod
    def available(models_dir=MODELS_DIR):
        return (os.path.exists(os.path.join(models_dir, "slm_briefing.pth"))
                and os.path.exists(os.path.join(models_dir, "slm_briefing_meta.json")))

    def generate_briefing(self, incident_log):
        """Generates a 2-sentence tactical briefing from a dense incident log in <100ms."""
        self._load()
        
        t0 = time.perf_counter()
        src_ids = text_to_ids(incident_log, self.src_vocab, max_len=MAX_SRC_LEN, add_sos=True, add_eos=True)
        src_tensor = torch.tensor([pad_sequence(src_ids, MAX_SRC_LEN)], dtype=torch.long)
        
        gen_ids = self.model.generate(
            src_tensor, max_len=MAX_TGT_LEN,
            src_vocab=self.src_vocab, tgt_vocab=self.tgt_vocab
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        
        raw_text = ids_to_text(gen_ids, self.inv_tgt)
        formatted = format_two_sentences(raw_text)
            
        chips = self.extract_tactical_chips(formatted)
        time_stats = self.compute_time_savings(incident_log, formatted)
        
        return {
            "briefing": formatted,
            "latency_ms": round(latency_ms, 1),
            "tactical_chips": chips,
            "time_stats": time_stats
        }

    def extract_tactical_chips(self, text):
        """Identifies recognized domain dictionary codes to display as tactical badges."""
        toks = tokenize(text)
        found = []
        for t in toks:
            if t in self.domain_tokens and t not in [f[0] for f in found]:
                desc = "Tactical Code"
                for cat, codes in self.domain_dict.items():
                    if t in codes:
                        desc = codes[t]
                        break
                found.append((t, desc))
        return found

    def compute_time_savings(self, log_text, briefing_text):
        """Calculates exact reading duration and percentage reduction (Team Huddle metric)."""
        wpm = 140.0  # Technical reading speed in words per minute
        log_words = len(log_text.split())
        briefing_words = len(briefing_text.split())
        
        log_read_sec = log_words / (wpm / 60.0)
        briefing_sec = briefing_words / (wpm / 60.0)
        
        reduction = (1.0 - (briefing_words / max(log_words, 1))) * 100.0
        
        return {
            "log_words": log_words,
            "briefing_words": briefing_words,
            "log_seconds": round(log_read_sec, 1),
            "briefing_seconds": round(briefing_sec, 1),
            "reduction_pct": round(reduction, 1),
            "passed_80pct_gate": reduction >= 80.0
        }


# Backward-compatible aliases
SlmAssistant = TacticalBriefingAssistant
load_slm_assistant = TacticalBriefingAssistant


def load_briefing_assistant(models_dir=MODELS_DIR):
    return TacticalBriefingAssistant(models_dir=models_dir)


if __name__ == "__main__":
    if not TacticalBriefingAssistant.available():
        print("SLM briefing weights not found. Run slm_train.py first.")
        sys.exit(1)
        
    asst = load_briefing_assistant()
    sample = DEMO_PRESETS["🌊 Flash Flood Isolation (Sector 3 Delta)"]
    res = asst.generate_briefing(sample)
    print("Briefing:", res["briefing"])
    print("Latency :", res["latency_ms"], "ms")
    print("Chips   :", res["tactical_chips"])
    print("Savings :", res["time_stats"])