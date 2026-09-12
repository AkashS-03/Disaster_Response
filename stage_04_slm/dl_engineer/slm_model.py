"""
Stage 04 - SLM | DL Engineer: Transformer SLM with PEFT / LoRA
==============================================================
Modern Encoder-Decoder Transformer architecture equipped with Low-Rank Adaptation
(LoRA) parameter-efficient fine-tuning layers.

Features:
  - Pure Transformer: Multi-Head Self-Attention + Multi-Head Cross-Attention
  - PEFT / LoRA: Freezes base Transformer projections (W_0) and trains only low-rank
    adapters: h = W_0*x + (alpha/r) * B*A*x (rank r=8, alpha=16)
  - Severity-conditioned decoding:
      * LOW     -> < 1 sentence (phrase, 0 periods)
      * MODERATE-> 1 sentence   (stops at 1st period)
      * SEVERE  -> 2 sentences  (stops at 2nd period)
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

try:
    from stage_04_slm.data_engineer.slm_utils import SPECIAL, SOS, EOS, PAD, UNK, text_to_ids, ids_to_text, tokenize  # noqa: E402
except (ImportError, ModuleNotFoundError):
    from data_engineer.slm_utils import SPECIAL, SOS, EOS, PAD, UNK, text_to_ids, ids_to_text, tokenize  # noqa: E402

PAD_IDX = SPECIAL[PAD]
SOS_IDX = SPECIAL[SOS]
EOS_IDX = SPECIAL[EOS]


# =============================================================================
# PEFT / LoRA LINEAR LAYER
# =============================================================================
class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation (LoRA) layer following Hu et al. (2021).
    W = W_0 + (alpha / r) * B @ A
    Where W_0 is frozen, A ~ N(0, sigma), B = 0 at initialization.
    """
    def __init__(self, in_features, out_features, r=8, lora_alpha=16, lora_dropout=0.05, freeze_base=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0

        # Base frozen projection
        self.base_linear = nn.Linear(in_features, out_features)
        if freeze_base:
            for p in self.base_linear.parameters():
                p.requires_grad = False

        # Trainable LoRA adapter matrices
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            self.dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
            self.reset_parameters()
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def reset_parameters(self):
        if self.r > 0:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base_out = self.base_linear(x)
        if self.r > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T * self.scaling
            return base_out + lora_out
        return base_out

    def merge_weights(self):
        """Merges LoRA adapter into base weights for zero-latency deployment."""
        if self.r > 0:
            delta = (self.lora_B @ self.lora_A) * self.scaling
            self.base_linear.weight.data += delta
            self.r = 0


# =============================================================================
# POSITIONAL ENCODINGS
# =============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# =============================================================================
# TRANSFORMER WITH PEFT / LoRA ATTENTION
# =============================================================================
class TransformerLoRA(nn.Module):
    """
    Encoder-Decoder Transformer with LoRA parameter-efficient fine-tuning.
    Base Transformer is frozen; only LoRA adapters are trained.
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, nhead=4,
                 num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=512,
                 dropout=0.1, lora_r=8, lora_alpha=16, freeze_base=True):
        super().__init__()
        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha

        # Separate Source & Target token embeddings with Positional Encoding
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=PAD_IDX)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_encoder = PositionalEncoding(d_model, max_len=512, dropout=dropout)

        # Standard Transformer Core
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # Output projection head with LoRA
        self.fc_out = LoRALinear(d_model, tgt_vocab_size, r=lora_r, lora_alpha=lora_alpha, freeze_base=freeze_base)

        # Freeze Base Transformer parameters if freeze_base is True
        if freeze_base:
            for p in self.src_embedding.parameters():
                p.requires_grad = False
            for p in self.tgt_embedding.parameters():
                p.requires_grad = False
            for p in self.transformer.parameters():
                p.requires_grad = False

    def get_lora_parameter_counts(self):
        """Returns (total_params, trainable_lora_params, lora_pct)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        pct = (trainable / total) * 100.0 if total > 0 else 0.0
        return total, trainable, round(pct, 2)

    def make_src_mask(self, src):
        return (src == PAD_IDX)

    def make_tgt_mask(self, tgt):
        sz = tgt.size(1)
        mask = torch.triu(torch.full((sz, sz), float('-inf'), device=tgt.device), diagonal=1)
        return mask

    def forward(self, src, tgt):
        src_key_padding_mask = self.make_src_mask(src)
        tgt_key_padding_mask = self.make_src_mask(tgt)
        tgt_mask = self.make_tgt_mask(tgt)

        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))

        out = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        logits = self.fc_out(out)
        return logits

    @torch.no_grad()
    def generate(self, src, max_len=48, src_vocab=None, tgt_vocab=None, severity="MODERATE"):
        """
        Autoregressively decodes summary tokens with severity-constrained stopping:
          - LOW     : stops at max 10 tokens or 1st period (< 1 sentence)
          - MODERATE: stops after 1st period (1 sentence)
          - SEVERE  : stops after 2nd period (2 sentences)
        """
        self.eval()
        device = next(self.parameters()).device
        src = src.to(device)
        src_key_padding_mask = self.make_src_mask(src)

        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

        ys = torch.ones(1, 1).fill_(SOS_IDX).type(torch.long).to(device)

        period_id = tgt_vocab.get(".", None) if tgt_vocab else None
        periods_seen = 0
        max_periods = 0 if severity == "LOW" else (1 if severity == "MODERATE" else 2)

        for _ in range(max_len):
            tgt_mask = self.make_tgt_mask(ys)
            tgt_emb = self.pos_encoder(self.tgt_embedding(ys) * math.sqrt(self.d_model))
            out = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            logits = self.fc_out(out[:, -1])
            next_token = torch.argmax(logits, dim=1).item()

            if next_token == EOS_IDX:
                break

            ys = torch.cat([ys, torch.ones(1, 1).type_as(ys.data).fill_(next_token)], dim=1)

            if period_id is not None and next_token == period_id:
                periods_seen += 1
                if periods_seen >= max_periods and max_periods > 0:
                    break

            if severity == "LOW" and ys.size(1) >= 10:
                break

        return ys.squeeze(0).tolist()


def build_model(meta, weights_path=None):
    cfg = meta.get("config", {})
    src_vocab_size = len(meta["src_vocab"])
    tgt_vocab_size = len(meta["tgt_vocab"])

    model = TransformerLoRA(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=cfg.get("d_model", 256),
        nhead=cfg.get("nhead", 4),
        num_encoder_layers=cfg.get("num_encoder_layers", 2),
        num_decoder_layers=cfg.get("num_decoder_layers", 2),
        dim_feedforward=cfg.get("dim_feedforward", 512),
        dropout=cfg.get("dropout", 0.1),
        lora_r=cfg.get("lora_r", 8),
        lora_alpha=cfg.get("lora_alpha", 16),
        freeze_base=False
    )
    if weights_path and os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    return model