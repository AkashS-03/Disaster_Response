"""
Stage 04 - SLM | Data Engineer: Tactical Briefing Utilities
===========================================================
Shared utilities for the 5-Second Tactical Voice Briefing SLM.
Includes Domain Dictionary loading, tactical tokenization, vocabulary
management, and sequence-to-sequence batch generation.
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
    r"[a-zA-Z0-9']+", re.IGNORECASE)

MAX_SRC_LEN = 120  # Log input length
MAX_TGT_LEN = 36   # Briefing output length (2 crisp sentences)


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
    """Builds a frequency vocabulary with special and domain tokens always included."""
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    
    # Always include domain dictionary tokens
    domain_tokens = additional_tokens or get_domain_tokens()
    
    words = [w for w, n in counter.items() if n >= min_freq and w not in domain_tokens]
    words.sort(key=lambda w: counter[w], reverse=True)
    
    # Reserved space for specials and domain tokens
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
        words.append(inv_vocab.get(i, UNK))
    return " ".join(words)


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
        if len(s_ids) > 2 and len(t_ids) > 2:
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


def format_two_sentences(text_or_tokens):
    """Formats generated tokens into strictly 2 crisp, actionable sentences."""
    if isinstance(text_or_tokens, str):
        words = text_or_tokens.split()
    else:
        words = list(text_or_tokens)
        
    if not words:
        return "SITREP PRI-2: Emergency response active. Authorize SAR deployment."
        
    # Standardize domain tokens in uppercase
    norm_words = []
    for w in words:
        w_up = w.upper()
        if w_up in _DOMAIN_SET:
            norm_words.append(w_up)
        else:
            norm_words.append(w.lower())
            
    # Find boundary between Sentence 1 (Threat/SITREP) and Sentence 2 (Action/Directive)
    split_idx = None
    action_triggers = {"10-4", "AUTHORIZE", "ENFORCE", "ACTIVATE", "DISPATCH"}
    for i, w in enumerate(norm_words):
        if i >= 4 and w.upper() in action_triggers:
            split_idx = i
            break
            
    if split_idx is None:
        # Fallback to natural midpoint
        split_idx = max(len(norm_words) // 2, 4)
        
    s1_words = norm_words[:split_idx]
    s2_words = norm_words[split_idx:]
    
    s1 = " ".join(s1_words).strip()
    s2 = " ".join(s2_words).strip()
    
    s1 = s1[:1].upper() + s1[1:] if s1 else "SITREP: Incident reported"
    s2 = s2[:1].upper() + s2[1:] if s2 else "Authorize emergency teams"
    
    if not s1.endswith((".", "!", "?")):
        s1 += "."
    if not s2.endswith((".", "!", "?")):
        s2 += "."
        
    return f"{s1} {s2}"


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