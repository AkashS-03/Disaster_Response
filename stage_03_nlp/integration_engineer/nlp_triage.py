"""
Stage 03 - NLP | Integration Engineer
=====================================
Production wrapper for the NLP triage agent: loads the baseline-gated
shipped model (and optionally the deep BiLSTM for a live comparison),
applies the deterministic keyword guard rail, and supports the
human-in-the-loop abstention ("REVIEW") decision.

Flow for one message:
  1. statistical model -> class + confidence
  2. if confidence < threshold  -> REVIEW (human triage), stop
  3. else apply keyword guard rail floor (can only escalate, never downgrade)
  4. return final class + confidence (+ which keyword fired, if any)
"""

import os
import sys
import json

import joblib
import numpy as np
import torch

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from dl_engineer.nlp_utils import clean_text, tokenize, CLASSES  # noqa: E402
from dl_engineer.nlp_trainer import BiLSTMSeverity, MAX_LEN  # noqa: E402

SEVERE_FLOOR = [
    "trapped", "drowning", "drown", "rescue", "rescuing", "stranded",
    "injured", "dying", "unconscious", "bleeding", "buried", "rubble",
    "swept", "missing people", "no contact", "urgent", "emergency",
    "please help", "need help", "help us", "help immediately", "critical",
    "sos", "flee", "fleeing", "dead", "corpse", "pregnant", "baby",
]
MODERATE_FLOOR = [
    "flood", "flooded", "flooding", "storm", "hurricane", "cyclone",
    "typhoon", "earthquake", "landslide", "fire", "heavy rain",
    "torrential", "water rising", "water level", "evacuat", "shelter",
    "homeless", "shortage", "no electricity", "power cut", "power out",
    "road closed", "bridge", "aid", "supplies",
]


class NlpTriage:
    def __init__(self, model_dir, default_threshold=0.50):
        self.model_dir = model_dir
        self.threshold = default_threshold
        with open(os.path.join(model_dir, "winner.txt")) as f:
            self.shown_name = f.read().strip()
        self.stat = joblib.load(os.path.join(model_dir, self.shown_name))
        self.classes = list(self.stat.classes_)
        self.deep = None
        self.deep_vocab = None

    def load_deep(self):
        meta = json.load(open(os.path.join(self.model_dir, "bilstm_vocab.json")))
        model = BiLSTMSeverity(len(meta["vocab"]))
        model.load_state_dict(torch.load(
            os.path.join(self.model_dir, "bilstm_severity.pth"),
            map_location="cpu"))
        model.eval()
        self.deep = model
        self.deep_vocab = meta["vocab"]

    @staticmethod
    def _guard_hit(text):
        low = " " + clean_text(text) + " "
        hits = [k for k in SEVERE_FLOOR if k in low]
        if hits:
            return "SEVERE", hits
        hits = [k for k in MODERATE_FLOOR if k in low]
        if hits:
            return "MODERATE", hits
        return None, None

    def _stat_proba(self, text):
        return self.stat.predict_proba([text])[0]

    def _deep_proba(self, text):
        from dl_engineer.nlp_utils import encode
        x = torch.tensor(np.array(
            encode([text], self.deep_vocab, MAX_LEN), dtype=np.int64))
        with torch.no_grad():
            logits, _ = self.deep(x, (x != 0).float())
        return torch.softmax(logits, dim=1)[0].numpy()

    def triage(self, text, use_deep=False):
        """Returns final decision dict for one message.

        Order of operations (safety-first):
          1. deterministic keyword guard rail fires -> its answer wins ALWAYS
          2. otherwise, uncertain statistical model -> REVIEW (human)
          3. otherwise -> statistical model prediction
        The guard rail can never be overruled by an unsure estimate.
        """
        result = {"text": text, "model": "deep" if use_deep else "classical"}
        if use_deep and self.deep is not None:
            proba = self._deep_proba(text)
        else:
            proba = self._stat_proba(text)
        max_p = float(proba.max())
        stat_cls = self.classes[int(proba.argmax())]
        result["confidence"] = max_p
        result["stat_class"] = stat_cls

        guard_cls, guard_hits = self._guard_hit(text)
        result["guard_hits"] = guard_hits

        if guard_cls == "SEVERE":
            result["prediction"] = "SEVERE"
            result["reason"] = "guard"
            return result
        if guard_cls == "MODERATE":
            result["prediction"] = (stat_cls if stat_cls == "SEVERE" else "MODERATE")
            result["reason"] = "guard" if result["prediction"] != stat_cls else "model"
            return result
        if max_p < self.threshold:
            result["prediction"] = "REVIEW"
            result["reason"] = "low_confidence"
            return result
        result["prediction"] = stat_cls
        result["reason"] = "model"
        return result

    def triage_batch(self, texts, use_deep=False):
        """Batch wrapper returning arrays: pred, conf, reason, guard_hits."""
        preds, confs, reasons, hits = [], [], [], []
        for t in texts:
            r = self.triage(t, use_deep=use_deep)
            preds.append(r["prediction"])
            confs.append(r["confidence"])
            reasons.append(r["reason"])
            hits.append(r["guard_hits"])
        return np.array(preds), np.array(confs), np.array(reasons), hits


def build_triage(model_dir, threshold=0.50, with_deep=True):
    triage = NlpTriage(model_dir, default_threshold=threshold)
    if with_deep:
        try:
            triage.load_deep()
        except Exception as e:
            print(f"  (deep model not loaded: {e})")
    return triage


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    demo = build_triage(os.path.join(base_dir, "models"))
    samples = [
        "people are trapped on the roof and the water is rising please help",
        "flood water entering homes in the eastern ward, shelter opened",
        "sunny day, markets open as usual",
        "the building collapsed and a child is injured, rescue needed now",
    ]
    for s in samples:
        print(demo.triage(s))