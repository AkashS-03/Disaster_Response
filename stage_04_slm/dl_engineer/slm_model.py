"""
Stage 04 - SLM | DL Engineer: Tactical Briefing SLM Architecture
================================================================
Compact Sequence-to-Sequence neural architecture with Bahdanau Attention
fine-tuned for edge devices (offline laptop CPU).

Condenses dense multi-page crisis logs into exactly 2 crisp, actionable sentences
in <100ms.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from data_engineer.slm_utils import SPECIAL, SOS, EOS, PAD, UNK, text_to_ids, ids_to_text, tokenize  # noqa: E402

PAD_IDX = SPECIAL[PAD]
SOS_IDX = SPECIAL[SOS]
EOS_IDX = SPECIAL[EOS]


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=128, n_layers=1, dropout=0.15):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, num_layers=n_layers,
                           bidirectional=True, batch_first=True,
                           dropout=dropout if n_layers > 1 else 0.0)
        self.fc_h = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.fc_c = nn.Linear(hidden_dim * 2, hidden_dim * 2)

    def forward(self, src):
        # src: [batch_size, src_len]
        embedded = self.embedding(src)  # [batch_size, src_len, emb_dim]
        outputs, (h, c) = self.rnn(embedded)  # outputs: [batch_size, src_len, hidden_dim*2]
        
        # Concatenate forward and backward final hidden states
        h_cat = torch.cat((h[-2, :, :], h[-1, :, :]), dim=1)
        c_cat = torch.cat((c[-2, :, :], c[-1, :, :]), dim=1)
        
        h_init = torch.tanh(self.fc_h(h_cat)).unsqueeze(0)  # [1, batch_size, hidden_dim*2]
        c_init = torch.tanh(self.fc_c(c_cat)).unsqueeze(0)
        
        return outputs, (h_init, c_init)


class Attention(nn.Module):
    def __init__(self, enc_dim=256, dec_dim=256):
        super().__init__()
        self.attn = nn.Linear(enc_dim + dec_dim, dec_dim)
        self.v = nn.Linear(dec_dim, 1, bias=False)

    def forward(self, dec_hidden, enc_outputs, mask=None):
        # dec_hidden: [batch_size, dec_dim]
        # enc_outputs: [batch_size, src_len, enc_dim]
        src_len = enc_outputs.size(1)
        dec_hidden_rep = dec_hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        energy = torch.tanh(self.attn(torch.cat((dec_hidden_rep, enc_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)  # [batch_size, src_len]
        
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)
            
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, enc_dim=256, dec_dim=256, dropout=0.15):
        super().__init__()
        self.vocab_size = vocab_size
        self.attention = Attention(enc_dim, dec_dim)
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.rnn = nn.LSTM(emb_dim + enc_dim, dec_dim, batch_first=True)
        self.fc_out = nn.Linear(emb_dim + dec_dim + enc_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_tok, dec_state, enc_outputs, mask=None):
        # input_tok: [batch_size]
        # dec_state: (h, c) where h is [1, batch_size, dec_dim]
        h, c = dec_state
        embedded = self.dropout(self.embedding(input_tok.unsqueeze(1)))  # [batch_size, 1, emb_dim]
        
        # Calculate attention weights using top hidden layer
        a = self.attention(h[-1], enc_outputs, mask)  # [batch_size, src_len]
        a_weighted = a.unsqueeze(1)  # [batch_size, 1, src_len]
        
        # Context vector
        context = torch.bmm(a_weighted, enc_outputs)  # [batch_size, 1, enc_dim]
        
        rnn_input = torch.cat((embedded, context), dim=2)
        out, dec_state_new = self.rnn(rnn_input, dec_state)
        
        # Output projection
        pred_input = torch.cat((out.squeeze(1), context.squeeze(1), embedded.squeeze(1)), dim=1)
        logits = self.fc_out(pred_input)  # [batch_size, vocab_size]
        
        return logits, dec_state_new, a


class TacticalBriefingSLM(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size,
                 emb_dim=128, enc_hidden=128, dec_hidden=256, dropout=0.15):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, emb_dim, enc_hidden, n_layers=1, dropout=dropout)
        self.decoder = Decoder(tgt_vocab_size, emb_dim, enc_dim=enc_hidden * 2,
                               dec_dim=dec_hidden, dropout=dropout)
        self.tgt_vocab_size = tgt_vocab_size

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        # src: [batch_size, src_len], tgt: [batch_size, tgt_len]
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        
        outputs = torch.zeros(batch_size, tgt_len, self.tgt_vocab_size, device=src.device)
        mask = (src != PAD_IDX).float()
        
        enc_outputs, dec_state = self.encoder(src)
        
        # First input token is SOS
        input_tok = tgt[:, 0]
        
        for t in range(1, tgt_len):
            logits, dec_state, _ = self.decoder(input_tok, dec_state, enc_outputs, mask)
            outputs[:, t, :] = logits
            
            teacher_force = (torch.rand(1).item() < teacher_forcing_ratio)
            top1 = logits.argmax(1)
            input_tok = tgt[:, t] if teacher_force else top1
            
        return outputs

    @torch.no_grad()
    def generate(self, src, max_len=36, src_vocab=None, tgt_vocab=None):
        """Greedy autoregressive generation optimized for CPU edge deployment."""
        self.eval()
        mask = (src != PAD_IDX).float()
        enc_outputs, dec_state = self.encoder(src)
        
        batch_size = src.size(0)
        curr_tok = torch.full((batch_size,), SOS_IDX, dtype=torch.long, device=src.device)
        
        generated_ids = []
        period_count = 0
        
        period_id = tgt_vocab.get(".", None) if tgt_vocab else None
        
        for _ in range(max_len):
            logits, dec_state, _ = self.decoder(curr_tok, dec_state, enc_outputs, mask)
            # Suppress special tokens from being picked except EOS
            logits[:, PAD_IDX] = -1e9
            logits[:, UNK_IDX := SPECIAL[UNK]] = -1e9
            logits[:, SOS_IDX] = -1e9
            
            top1 = logits.argmax(1)
            tok_id = top1.item()
            
            if tok_id == EOS_IDX:
                break
                
            generated_ids.append(tok_id)
            curr_tok = top1
            
            if period_id is not None and tok_id == period_id:
                period_count += 1
                if period_count >= 2:
                    break
                    
        return generated_ids


def build_model(meta, weights_path=None, device="cpu"):
    cfg = meta["config"]
    src_vocab = meta["src_vocab"]
    tgt_vocab = meta["tgt_vocab"]
    
    model = TacticalBriefingSLM(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        emb_dim=cfg.get("emb_dim", 128),
        enc_hidden=cfg.get("enc_hidden", 128),
        dec_hidden=cfg.get("dec_hidden", 256),
        dropout=0.0
    )
    if weights_path and os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model