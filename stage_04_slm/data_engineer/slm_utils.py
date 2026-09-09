"""
Stage 04 - SLM | Data Engineer: shared utilities
================================================
Tokeniser + vocabulary builder used by the Small Language Model (SLM)
training and its integration wrapper. Real data only - no synthetic text
is ever created or injected into any pipeline.
"""

import json
import re
from collections import Counter

_TOK = re.compile(r"[a-z0-9']+")

SOS = "<s>"
EOS = "</s>"
PAD = "<pad>"
UNK = "<unk>"

SPECIAL = {SOS: 0, EOS: 1, PAD: 2, UNK: 3}

MAX_SEQ = 24


def tokenize(text):
    return [t for t in _TOK.findall(str(text).lower())]


def build_vocab(texts, max_size=16000, min_freq=2):
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    words = [w for w, n in counter.items() if n >= min_freq]
    words.sort(key=lambda w: counter[w], reverse=True)
    words = words[:max_size]
    return {**SPECIAL, **{w: i + len(SPECIAL) for i, w in enumerate(words)}}


def save_meta(path, vocab, config):
    payload = {"vocab": vocab, "config": config}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def load_meta(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_windows(texts, vocab, seq_len=MAX_SEQ):
    """Real message windows for next-word training: (input_ids, target_ids).

    Each real message becomes <s> + words + </s>; every position predicts
    the next word. Nothing here is generated - the windows come straight
    from the real dataset.
    """
    inputs, targets = [], []
    for t in texts:
        ids = [vocab.get(w, vocab[UNK]) for w in tokenize(t)]
        if not ids:
            continue
        seq = [vocab[SOS]] + ids + [vocab[EOS]]
        for i in range(0, len(seq) - 1, seq_len):
            chunk = seq[i:i + seq_len + 1]
            if len(chunk) < 2:
                continue
            x = chunk[:-1]
            y = chunk[1:]
            x = x + [vocab[PAD]] * (seq_len - len(x))
            y = y + [vocab[PAD]] * (seq_len - len(y))
            inputs.append(x)
            targets.append(y)
    return inputs, targets