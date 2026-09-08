"""
Stage 03 - NLP | shared text utilities
======================================
Tokeniser + vocabulary builder used by both the DL trainer and the
integration wrapper so a message is always preprocessed identically.
"""

import re
from collections import Counter

_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_HASHTAG = re.compile(r"#(\w+)")
_NOISE = re.compile(r"[^a-zA-Z0-9'\s]")

CLASSES = ["LOW", "MODERATE", "SEVERE"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def clean_text(t):
    t = _URL.sub(" ", str(t))
    t = _MENTION.sub(" ", t)
    t = _HASHTAG.sub(r"\1", t)
    t = _NOISE.sub(" ", t)
    return " ".join(t.lower().split())


def tokenize(t):
    return [w for w in clean_text(t).split(" ") if len(w) > 1]


def build_vocab(texts, min_freq=2, max_size=50000):
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    words = sorted((w for w, n in counter.items() if n >= min_freq),
                   key=lambda w: counter[w], reverse=True)[:max_size]
    return {"<pad>": 0, "<unk>": 1, **{w: i + 2 for i, w in enumerate(words)}}


def encode(texts, vocab, max_len=64):
    """Return padded token-id tensor (list of lists)."""
    out = []
    for t in texts:
        ids = [vocab.get(w, 1) for w in tokenize(t)][:max_len]
        if not ids:
            ids = [1]  # guarantee >=1 real token so attention mask is never all-zero
        ids = ids + [0] * (max_len - len(ids))
        out.append(ids)
    return out