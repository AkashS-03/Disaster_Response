"""
Stage 04 - SLM | DL Engineer: Tactical Briefing SLM Training
============================================================
Fine-tunes the compact Sequence-to-Sequence model with Attention on the curated
incident log-to-tactical summary dataset. Runs entirely on CPU in ~2-3 minutes.

Artifacts written to stage_04_slm/models/:
  - slm_briefing.pth        fine-tuned Seq2Seq model weights
  - slm_briefing_meta.json  vocabularies and architecture configuration
"""

import os
import sys
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from data_engineer.slm_utils import (  # noqa: E402
    build_vocab, make_seq2seq_batches, save_meta, SPECIAL, PAD, SOS, EOS,
    ids_to_text, pad_sequence, text_to_ids, MAX_SRC_LEN, MAX_TGT_LEN
)
from dl_engineer.slm_model import TacticalBriefingSLM  # noqa: E402

PAD_IDX = SPECIAL[PAD]
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


def train_epoch(model, batches, optimizer, criterion, clip=1.0, teacher_forcing_ratio=0.5):
    model.train()
    epoch_loss = 0
    total_tokens = 0
    
    for src, tgt in batches:
        optimizer.zero_grad()
        # tgt: [batch_size, tgt_len]
        output = model(src, tgt, teacher_forcing_ratio=teacher_forcing_ratio)
        # output: [batch_size, tgt_len, vocab_size]
        
        output_dim = output.shape[-1]
        # Ignore first token (<sos>) when calculating loss
        output = output[:, 1:].reshape(-1, output_dim)
        target = tgt[:, 1:].reshape(-1)
        
        loss = criterion(output, target)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        non_pad = (target != PAD_IDX).sum().item()
        epoch_loss += loss.item() * non_pad
        total_tokens += non_pad
        
    return epoch_loss / max(total_tokens, 1)


@torch.no_grad()
def evaluate_loss(model, batches, criterion):
    model.eval()
    epoch_loss = 0
    total_tokens = 0
    
    for src, tgt in batches:
        output = model(src, tgt, teacher_forcing_ratio=0.0)
        output_dim = output.shape[-1]
        
        output = output[:, 1:].reshape(-1, output_dim)
        target = tgt[:, 1:].reshape(-1)
        
        loss = criterion(output, target)
        non_pad = (target != PAD_IDX).sum().item()
        epoch_loss += loss.item() * non_pad
        total_tokens += non_pad
        
    return epoch_loss / max(total_tokens, 1)


def main():
    print("=== Stage 04 SLM | DL Engineer: Fine-Tuning Tactical Briefing SLM ===")
    data_path = os.path.join(base_dir, "data", "briefing_dataset.csv")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}")
        sys.exit(1)
        
    df = pd.read_csv(data_path)
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    
    print(f"Loaded {len(train_df)} training and {len(val_df)} validation pairs.")
    
    # 1. Build source and target vocabularies
    print("Building domain-aware vocabularies...")
    src_vocab = build_vocab(train_df["incident_log"], max_size=8000, min_freq=2)
    tgt_vocab = build_vocab(train_df["tactical_summary"], max_size=2000, min_freq=1)
    
    print(f"Source Vocabulary Size: {len(src_vocab)}")
    print(f"Target Vocabulary Size: {len(tgt_vocab)}")
    
    # 2. Create Batches
    train_batches = make_seq2seq_batches(
        train_df["incident_log"].tolist(), train_df["tactical_summary"].tolist(),
        src_vocab, tgt_vocab, batch_size=32, shuffle=True
    )
    val_batches = make_seq2seq_batches(
        val_df["incident_log"].tolist(), val_df["tactical_summary"].tolist(),
        src_vocab, tgt_vocab, batch_size=32, shuffle=False
    )
    print(f"Batches: {len(train_batches)} train, {len(val_batches)} val.")
    
    # 3. Model Architecture
    cfg = {
        "emb_dim": 128,
        "enc_hidden": 128,
        "dec_hidden": 256,
        "dropout": 0.15,
        "max_src": MAX_SRC_LEN,
        "max_tgt": MAX_TGT_LEN
    }
    
    model = TacticalBriefingSLM(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        emb_dim=cfg["emb_dim"],
        enc_hidden=cfg["enc_hidden"],
        dec_hidden=cfg["dec_hidden"],
        dropout=cfg["dropout"]
    )
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {total_params:,} (~{total_params * 4 / (1024*1024):.1f} MB fp32)")
    
    # 4. Training Loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    
    epochs = 8
    best_val_loss = float("inf")
    weights_path = os.path.join(models_dir, "slm_briefing.pth")
    meta_path = os.path.join(models_dir, "slm_briefing_meta.json")
    
    print("\nStarting fine-tuning...")
    t0 = time.time()
    
    for ep in range(1, epochs + 1):
        t_ep = time.time()
        tf_ratio = max(0.6 - (ep * 0.05), 0.2)
        tr_loss = train_epoch(model, train_batches, optimizer, criterion, teacher_forcing_ratio=tf_ratio)
        val_loss = evaluate_loss(model, val_batches, criterion)
        
        val_ppl = np.exp(min(val_loss, 20.0))
        print(f"  Epoch {ep:02d}/{epochs:02d} | Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f} (PPL: {val_ppl:.1f}) | {time.time() - t_ep:.1f}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), weights_path)
            
    print(f"\nFine-tuning completed in {time.time() - t0:.1f}s. Best Val Loss: {best_val_loss:.4f}")
    
    # 5. Save metadata
    save_meta(meta_path, src_vocab, tgt_vocab, cfg)
    print(f"Saved artifacts to {weights_path} and {meta_path}")
    
    # 6. Sample Generation Test
    inv_tgt = {v: k for k, v in tgt_vocab.items()}
    sample_log = val_df.iloc[0]["incident_log"]
    ground_truth = val_df.iloc[0]["tactical_summary"]
    
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    
    src_ids = text_to_ids(sample_log, src_vocab, max_len=MAX_SRC_LEN, add_sos=True, add_eos=True)
    src_tensor = torch.tensor([pad_sequence(src_ids, MAX_SRC_LEN)], dtype=torch.long)
    
    gen_ids = model.generate(src_tensor, max_len=MAX_TGT_LEN, src_vocab=src_vocab, tgt_vocab=tgt_vocab)
    generated_briefing = ids_to_text(gen_ids, inv_tgt)
    
    print("\n--- Field Test Validation ---")
    print("Log (sample first 100 chars):", sample_log[:100], "...")
    print("Ground Truth :", ground_truth)
    print("SLM Generated:", generated_briefing)


if __name__ == "__main__":
    main()