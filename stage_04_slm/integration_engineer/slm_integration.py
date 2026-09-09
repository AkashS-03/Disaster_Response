"""
Stage 04 - SLM | Integration Engineer: Copilot wrapper
=======================================================
Exposes the trained Small Language Model (SLM) to the dashboard as an
ASSISTIVE tool. The SLM only makes suggestions - it never overrides the
shipped classifier and never invents data.

Everything is real-data-only:
  - `perplexity(text)`  faithfulness/domain-fit gauge (real-probability)
  - `complete(text)`     next-word suggestions, clearly labelled "SLM guess"
  - `refine(text)`       deterministic text hygiene, NO generated content
"""

import os
import sys

import torch
import torch.nn.functional as F

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from data_engineer.slm_utils import SOS, EOS, PAD, SPECIAL, UNK, MAX_SEQ, load_meta, tokenize  # noqa: E402
from dl_engineer.slm_model import SlmLm  # noqa: E402

MODELS_DIR = os.path.join(base_dir, "models")
SOS_ID = SPECIAL[SOS]
EOS_ID = SPECIAL[EOS]
PAD_ID = SPECIAL[PAD]
UNK_ID = SPECIAL[UNK]

# Pure function words / URL remnants add no drafting value as hints. We still
# use the model's REAL probabilities - we only choose which words to SHOW.
_COMMON_WORDS = frozenset("""
the and to of in a for is on that with at by from as we i you it be are was were
have has had this there so but or an not all if can will more some what when one
two our their my your do no he she her him his its up out about than then after
over while during into which who whom whose t co http com www also very just
been being get go got like would could should may might still such only other
many much most any each few both these those them they us me don't it's i'm we're
""".split())


class SlmAssistant:
    """Lazy-loaded SLM wrapper used by the dashboard (Copilot panel)."""

    def __init__(self, models_dir=MODELS_DIR):
        self.models_dir = models_dir
        self.model = None
        self.vocab = None
        self._inv = None

    def _load(self):
        if self.model is not None:
            return
        meta = load_meta(os.path.join(self.models_dir, "slm_lm_meta.json"))
        self.vocab = meta["vocab"]
        cfg = meta["config"]
        self.model = SlmLm(vocab_size=len(self.vocab), emb=cfg["emb"],
                           hidden=cfg["hidden"], layers=cfg["layers"], dropout=0.0)
        self.model.load_state_dict(torch.load(
            os.path.join(self.models_dir, "slm_lstm.pth"), map_location="cpu"))
        self.model.eval()
        self._inv = {v: k for k, v in self.vocab.items()}

    @staticmethod
    def available(models_dir=MODELS_DIR):
        return (os.path.exists(os.path.join(models_dir, "slm_lstm.pth"))
                and os.path.exists(os.path.join(models_dir, "slm_lm_meta.json")))

    def _ids(self, text, with_eos=False, max_seq=MAX_SEQ):
        ids = [self.vocab.get(w, UNK_ID) for w in tokenize(text)]
        if with_eos:
            ids = ids + [EOS_ID]
        return ids[:max_seq]

    def perplexity(self, text):
        """Domain-fit gauge: exp(avg neg-log-prob of REAL next words)."""
        self._load()
        ids = [SOS_ID] + self._ids(text, with_eos=True, max_seq=MAX_SEQ - 1)
        x = torch.tensor([ids], dtype=torch.long)
        t = torch.tensor([ids[1:] + [PAD_ID]], dtype=torch.long)
        with torch.no_grad():
            logits, _ = self.model(x)
        logp = F.log_softmax(logits, dim=-1)
        nll = []
        for i in range(t.size(1)):
            target = t[0, i].item()
            if target in (PAD_ID,):
                continue
            nll.append(-logp[0, i, target].item())
        if not nll:
            return float("inf")
        return float(torch.exp(torch.tensor(sum(nll) / len(nll))))

    def complete(self, text, k=5):
        """Most likely NEXT word(s), common filler words hidden.

        Always read as an SLM guess, not data. The probabilities are the
        model's real softmax outputs - we merely hide pure function words so
        the hints are actually useful (\"the / and / to\" are never helpful).
        """
        self._load()
        ids = [SOS_ID] + self._ids(text, with_eos=False, max_seq=MAX_SEQ - 1)
        x = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            logits, _ = self.model(x)
        logits = logits[0, -1, :]
        for spec in (SOS_ID, EOS_ID, PAD_ID, UNK_ID):
            logits[spec] = -1e9
        probs = F.softmax(logits, dim=-1)
        ordered = torch.argsort(probs, descending=True)
        picks = [i.item() for i in ordered if self._inv[i.item()] not in _COMMON_WORDS]
        picks = picks[:k] or [ordered[0].item()]
        return [(self._inv[i], round(probs[i].item() * 100, 1))
                for i in picks if i in self._inv]

    def refine(self, text):
        """Deterministic hygiene only - never generates or rewrites content."""
        cleaned = " ".join(str(text).split())
        if cleaned and cleaned[-1] not in ".!?":
            cleaned += "."
        return cleaned


def load_slm_assistant(models_dir=MODELS_DIR):
    return SlmAssistant(models_dir=models_dir)


if __name__ == "__main__":
    if not SlmAssistant.available():
        print("SLM weights not found - run slm_train.py first.")
        sys.exit(1)
    a = SlmAssistant()
    sample = "need food and water after the flood in our village"
    print("perplexity:", round(a.perplexity(sample), 2))
    print("complete:", a.complete(sample, k=5))
    messy = "  need   water   immediately   "
    print("refine:", repr(a.refine(messy)))
    print("refine(empty):", repr(a.refine("")))