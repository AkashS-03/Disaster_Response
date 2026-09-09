"""
Stage 04 - SLM | DL Engineer: Small Language Model (SLM) architecture
=====================================================================
A compact 2-layer LSTM language model trained on the real disaster-message
corpus. It learns to predict the NEXT word given the words so far - that
single skill gives us: a real encoder (its hidden states), a perplexity
(domain-fit) score, and word-level suggestions.

Real data only: the training text is exclusively the master text dataset.
"""

import torch
import torch.nn as nn

from data_engineer.slm_utils import PAD, SPECIAL, load_meta

PAD_IDX = SPECIAL[PAD]


class SlmLm(nn.Module):
    def __init__(self, vocab_size, emb=128, hidden=128, layers=2, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.rnn = nn.LSTM(emb, hidden, layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x, hidden=None):
        e = self.emb(x)
        out, hidden = self.rnn(e, hidden)
        logits = self.fc(out)
        return logits, hidden

    def encode(self, x):
        """Real contextual encoding: hidden state at the last real token."""
        e = self.emb(x)
        out, _ = self.rnn(e)
        lengths = (x != PAD_IDX).sum(dim=1) - 1
        idx = lengths.clamp(min=0)
        pooled = out[torch.arange(out.size(0)), idx]
        return pooled


def build_slm_from_meta(meta, state_dict_path=None, device="cpu"):
    vocab = meta["vocab"]
    cfg = meta["config"]
    model = SlmLm(vocab_size=len(vocab), emb=cfg["emb"], hidden=cfg["hidden"],
                  layers=cfg["layers"], dropout=0.0)
    if state_dict_path:
        model.load_state_dict(torch.load(state_dict_path, map_location=device))
    model.eval()
    return model, vocab


def load_slm(model_dir, device="cpu"):
    meta = load_meta(f"{model_dir}/slm_lm_meta.json")
    model, vocab = build_slm_from_meta(meta, f"{model_dir}/slm_lstm.pth", device)
    return model, vocab