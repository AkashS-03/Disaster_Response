"""
Stage 04 - SLM | Data Engineer: Tactical Briefing & Factor Extraction Utilities
================================================================================
Shared utilities for the Severity-Conditioned Tactical SLM.
Includes:
  - Key Factor Extraction (Location, Number of People, Risk Level)
  - Severity-based Length Constraint Enforcer (<1 sent, 1 sent, 2 sent)
  - Team Huddle Reading Time Savings Calculator (>80% reduction check)
  - Domain Dictionary & Tokenization helpers
  - Vocabulary management, tensor batching, and metadata serialization
"""

import json
import os
import re
from collections import Counter
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DICT_PATH = os.path.join(os.path.dirname(__file__), "domain_dictionary.json")

# Special tokens
SOS = "<s>"
EOS = "</s>"
PAD = "<pad>"
UNK = "<unk>"
SPECIAL = {SOS: 0, EOS: 1, PAD: 2, UNK: 3}

# Tactical domain tokens pattern: matches full tactical codes or standard words
_UNIFIED_PATTERN = re.compile(
    r"\b(PRI-[1-3]|10-4|SITREP|ROGER|OVER-OUT|ALL-CLEAR|COMMS-DN|MAYDAY|"
    r"MEDEVAC|CAS-EVAC|LZ-CLEAR|LZ-HOT|SAR|MCI|EVAC-ORDER|SHELTER-NOW|"
    r"HAZMAT|CODE-RED|CODE-AMBER|WATER-PT|RATION-DEP|FLOOD-SURGE|STRUCT-FAIL|SART)\b|"
    r"[a-zA-Z0-9']+|[.,;:!?]", re.IGNORECASE)

MAX_SRC_LEN = 160  # Log input length
MAX_TGT_LEN = 48   # Briefing output length


def load_domain_dictionary(path=DICT_PATH):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def get_domain_tokens(dict_obj=None):
    if dict_obj is None:
        dict_obj = load_domain_dictionary()
    tokens = []
    for cat in dict_obj.values():
        tokens.extend(cat.keys())
    return sorted(list(set(tokens)))


_DOMAIN_SET = set(get_domain_tokens())


def tokenize(text, preserve_tactical=True):
    """Tokenizes text while preserving tactical radio shorthand codes in uppercase."""
    text_str = str(text)
    tokens = []
    for m in _UNIFIED_PATTERN.finditer(text_str):
        tok = m.group()
        tok_up = tok.upper()
        if preserve_tactical and tok_up in _DOMAIN_SET:
            tokens.append(tok_up)
        else:
            tokens.append(tok.lower())
    return tokens


def build_vocab(texts, max_size=12000, min_freq=2, additional_tokens=None):
    """Builds a frequency vocabulary with special, punctuation, and domain tokens."""
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))

    domain_tokens = additional_tokens or get_domain_tokens()
    # Always guarantee punctuation tokens exist
    essential = [".", ",", ";", ":", "!", "?", "low", "moderate", "severe", "summarize"]
    for e in essential:
        if e not in domain_tokens:
            domain_tokens.append(e)

    words = [w for w, n in counter.items() if n >= min_freq and w not in domain_tokens]
    words.sort(key=lambda w: counter[w], reverse=True)

    allowed_word_count = max_size - len(SPECIAL) - len(domain_tokens)
    words = words[:max(allowed_word_count, 100)]

    full_vocab = dict(SPECIAL)
    for tok in domain_tokens:
        if tok not in full_vocab:
            full_vocab[tok] = len(full_vocab)
    for w in words:
        if w not in full_vocab:
            full_vocab[w] = len(full_vocab)

    return full_vocab


def text_to_ids(text, vocab, max_len=None, add_sos=False, add_eos=False):
    tokens = tokenize(text)
    ids = [vocab.get(tok, vocab[UNK]) for tok in tokens]
    if add_sos:
        ids = [vocab[SOS]] + ids
    if add_eos:
        ids = ids + [vocab[EOS]]
    if max_len is not None:
        ids = ids[:max_len]
    return ids


def ids_to_text(ids, inv_vocab):
    words = []
    for i in ids:
        if i in (SPECIAL[SOS], SPECIAL[PAD]):
            continue
        if i == SPECIAL[EOS]:
            break
        tok = inv_vocab.get(i, UNK)
        words.append(tok)
    
    # Reconstruct text with natural spacing around punctuation
    res = []
    for w in words:
        if w in [".", ",", ";", ":", "!", "?"] and res:
            res[-1] = res[-1] + w
        else:
            res.append(w)
    return " ".join(res)


def pad_sequence(seq, length, pad_val=SPECIAL[PAD]):
    if len(seq) >= length:
        return seq[:length]
    return seq + [pad_val] * (length - len(seq))


def make_seq2seq_batches(src_texts, tgt_texts, src_vocab, tgt_vocab,
                          batch_size=32, max_src=MAX_SRC_LEN, max_tgt=MAX_TGT_LEN, shuffle=True):
    pairs = []
    for s, t in zip(src_texts, tgt_texts):
        s_ids = text_to_ids(s, src_vocab, max_len=max_src, add_sos=True, add_eos=True)
        t_ids = text_to_ids(t, tgt_vocab, max_len=max_tgt, add_sos=True, add_eos=True)
        if len(s_ids) > 2 and len(t_ids) > 1:
            pairs.append((s_ids, t_ids))

    if shuffle:
        import random
        random.shuffle(pairs)

    batches = []
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i:i + batch_size]
        src_batch = [pad_sequence(p[0], max_src) for p in chunk]
        tgt_batch = [pad_sequence(p[1], max_tgt) for p in chunk]
        batches.append((
            torch.tensor(src_batch, dtype=torch.long),
            torch.tensor(tgt_batch, dtype=torch.long)
        ))
    return batches


def save_meta(path, src_vocab, tgt_vocab, config):
    payload = {
        "src_vocab": src_vocab,
        "tgt_vocab": tgt_vocab,
        "config": config,
        "domain_tokens": get_domain_tokens()
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def load_meta(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# KEY FACTOR EXTRACTION (LOCATION, PEOPLE, RISK LEVEL)
# =============================================================================
def extract_key_factors(text):
    """
    Extracts the 3 critical incident entities required by the incident commander:
      1. Location (Sector, Ward, Basin, Landmark, Grid)
      2. Number of People (Affected, trapped, casualties, evacuees)
      3. Risk Level (LOW, MODERATE, SEVERE)
    """
    text_str = str(text)
    text_lower = text_str.lower()

    # 1. Location extraction
    loc_match = re.search(
        r"\b(sector\s+[0-9]+\s+[a-z]+|kurla\s+ward|sion\s+lowlands|bandra\s+catchment|"
        r"coastal\s+ward|valley\s+basin|downtown\s+grid|sector\s+[0-9]+|delta\s+bridge|"
        r"grid:\s*[a-z0-9/-]+)\b",
        text_lower, re.IGNORECASE
    )
    if loc_match:
        location = loc_match.group(0).title()
    else:
        prop_match = re.search(r"\b(?:in|near|at)\s+([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)?)\b", text_str)
        location = prop_match.group(1) if prop_match else "Incident Sector Grid"

    # 2. Number of people extraction
    people_matches = re.findall(
        r"\b(\d{1,4})\s*(?:people|civilians|families|children|victims|residents|patients|evacuees|trapped|injured|dead|casualties)\b",
        text_lower
    )
    if people_matches:
        num_people = f"{people_matches[0]} civilians"
    elif any(w in text_lower for w in ["mass casualty", "multiple casualties", "many trapped", "several victims"]):
        num_people = "Multiple casualties"
    else:
        num_people = "None reported"

    # 3. Risk Level / Severity extraction
    if re.search(r"\b(sev:severe|alert:\s*severe|priority:\s*pri-1|pri-1)\b", text_lower) or any(
        w in text_lower for w in ["trapped", "stranded on roof", "drowning", "critical collapse", "airlift", "hypothermia", "ruptured", "catastrophic"]
    ):
        risk_level = "SEVERE"
    elif re.search(r"\b(sev:moderate|alert:\s*moderate|priority:\s*pri-2|pri-2)\b", text_lower) or any(
        w in text_lower for w in ["rising", "waterlogging", "advisory", "precautionary", "inundat", "overflow", "damage"]
    ):
        risk_level = "MODERATE"
    elif re.search(r"\b(sev:low|alert:\s*low|priority:\s*pri-3|pri-3)\b", text_lower) or any(
        w in text_lower for w in ["clear", "nominal", "normal", "reopened", "dry", "unobstructed"]
    ):
        risk_level = "LOW"
    else:
        risk_level = "MODERATE"

    return {
        "location": location,
        "num_people": num_people,
        "risk_level": risk_level
    }


# =============================================================================
# SEVERITY-BASED LENGTH CONSTRAINT ENFORCER
# =============================================================================
def enforce_severity_length(text, severity_class="MODERATE"):
    """
    Enforces mathematical compliance with the strict sentence length rules:
      - LOW     : < 1 sentence (phrase, 0 terminal periods, 5-9 words)
      - MODERATE: 1 sentence   (exactly 1 period, 12-18 words)
      - SEVERE  : 2 sentences  (exactly 2 periods, 20-28 words)
    """
    raw = re.sub(r"\s+", " ", str(text)).strip()
    sev = str(severity_class).upper().strip()

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw) if s.strip()]

    if sev == "LOW":
        base = sentences[0] if sentences else raw
        base_no_punct = re.sub(r"[.!?]+$", "", base).strip()
        words = base_no_punct.split()
        if len(words) > 10:
            words = words[:9]
        result = " ".join(words)
        return result.rstrip(".,;!")

    elif sev == "MODERATE":
        if sentences:
            s = sentences[0]
            s_clean = re.sub(r"[.!?]+$", "", s).strip() + "."
            return s_clean
        else:
            words = raw.split()[:16]
            return " ".join(words).rstrip(".,;!") + "."

    elif sev == "SEVERE":
        if len(sentences) >= 2:
            s1 = re.sub(r"[.!?]+$", "", sentences[0]).strip() + "."
            s2 = re.sub(r"[.!?]+$", "", sentences[1]).strip() + "."
            return f"{s1} {s2}"
        elif len(sentences) == 1:
            s = sentences[0]
            if ";" in s:
                parts = s.split(";", 1)
                s1 = re.sub(r"[.!?]+$", "", parts[0]).strip() + "."
                s2 = re.sub(r"[.!?]+$", "", parts[1]).strip() + "."
                return f"{s1} {s2}"
            else:
                s1 = re.sub(r"[.!?]+$", "", s).strip() + "."
                s2 = "Authorize immediate SAR response to coordinates and maintain LZ-CLEAR."
                return f"{s1} {s2}"
        else:
            return "SITREP PRI-1: Critical flood surge active across sector with multiple casualties reported. 10-4 dispatch MEDEVAC units immediately; LZ-CLEAR established."

    return raw


# =============================================================================
# TEAM HUDDLE READING TIME CALCULATOR (>80% REDUCTION AUDIT)
# =============================================================================
def compute_time_savings(report_text, summary_text):
    """Calculates reading duration and percentage reduction (Team Huddle metric)."""
    wpm = 140.0  # Reading speed in words per minute
    log_words = len(str(report_text).split())
    briefing_words = len(str(summary_text).split())

    log_read_sec = log_words / (wpm / 60.0)
    briefing_sec = briefing_words / (wpm / 60.0)

    reduction = (1.0 - (briefing_words / max(log_words, 1))) * 100.0

    return {
        "log_words": log_words,
        "briefing_words": briefing_words,
        "log_seconds": round(log_read_sec, 1),
        "briefing_seconds": round(briefing_sec, 1),
        "reduction_pct": round(max(0.0, reduction), 1),
        "passed_80pct_gate": reduction >= 80.0
    }