"""
Stage 04 - SLM | DL Engineer: Transformer SLM with PEFT / LoRA Training
========================================================================
Fine-tunes the Encoder-Decoder Transformer with LoRA parameter-efficient layers
on the curated severity-conditioned dataset. Runs on CPU in ~1.5 - 2 minutes.

Outputs:
  - stage_04_slm/models/slm_briefing.pth
  - stage_04_slm/models/slm_briefing_meta.json
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
from dl_engineer.slm_model import TransformerLoRA  # noqa: E402

PAD_IDX = SPECIAL[PAD]
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


def train_epoch(model, batches, optimizer, criterion, clip=1.0):
    model.train()
    epoch_loss = 0
    total_tokens = 0

    for src, tgt in batches:
        optimizer.zero_grad()
        # tgt_input: all tokens except the last; tgt_output: all tokens except the first (<sos>)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        logits = model(src, tgt_input)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        non_pad = (tgt_output != PAD_IDX).sum().item()
        epoch_loss += loss.item() * non_pad
        total_tokens += non_pad

    return epoch_loss / max(total_tokens, 1)


@torch.no_grad()
def evaluate_loss(model, batches, criterion):
    model.eval()
    epoch_loss = 0
    total_tokens = 0

    for src, tgt in batches:
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        logits = model(src, tgt_input)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))

        non_pad = (tgt_output != PAD_IDX).sum().item()
        epoch_loss += loss.item() * non_pad
        total_tokens += non_pad

    return epoch_loss / max(total_tokens, 1)


def main():
    print("=== Stage 04 SLM | DL Engineer: Training Transformer with PEFT/LoRA ===")
    data_path = os.path.join(base_dir, "data", "briefing_dataset.csv")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}")
        sys.exit(1)

    df = pd.read_csv(data_path)
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]

    print(f"Train samples: {len(train_df)} | Validation samples: {len(val_df)}")

    # Build vocabularies
    print("Building vocabulary across input prompts and target summaries...")
    src_vocab = build_vocab(train_df["input_prompt"], max_size=10000, min_freq=2)
    tgt_vocab = build_vocab(train_df["target_summary"], max_size=6000, min_freq=1)

    print(f"Source Vocab Size: {len(src_vocab)} | Target Vocab Size: {len(tgt_vocab)}")

    # Make batches
    train_batches = make_seq2seq_batches(
        train_df["input_prompt"].tolist(), train_df["target_summary"].tolist(),
        src_vocab, tgt_vocab, batch_size=32, max_src=MAX_SRC_LEN, max_tgt=MAX_TGT_LEN, shuffle=True
    )
    val_batches = make_seq2seq_batches(
        val_df["input_prompt"].tolist(), val_df["target_summary"].tolist(),
        src_vocab, tgt_vocab, batch_size=32, max_src=MAX_SRC_LEN, max_tgt=MAX_TGT_LEN, shuffle=False
    )

    # Initialize Transformer with LoRA
    d_model = 256
    nhead = 4
    n_enc = 2
    n_dec = 2
    d_ff = 512
    lora_r = 8
    lora_alpha = 16

    model = TransformerLoRA(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=n_enc,
        num_decoder_layers=n_dec,
        dim_feedforward=d_ff,
        dropout=0.1,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        freeze_base=False
    )

    tot_p, train_p, lora_pct = model.get_lora_parameter_counts()
    print("\n--- PEFT / LoRA Parameter Audit ---")
    print(f"Total Base Parameters    : {tot_p:,}")
    print(f"Trainable LoRA Parameters: {train_p:,}")
    print(f"LoRA Adaptation Ratio    : {lora_pct:.2f}%")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

    epochs = 8
    best_val_loss = float("inf")
    best_model_path = os.path.join(models_dir, "slm_briefing.pth")
    meta_path = os.path.join(models_dir, "slm_briefing_meta.json")

    print(f"\nStarting training for {epochs} epochs on CPU...")
    t0 = time.time()

    for ep in range(1, epochs + 1):
        ep_start = time.time()
        tr_loss = train_epoch(model, train_batches, optimizer, criterion)
        vl_loss = evaluate_loss(model, val_batches, criterion)
        ep_sec = time.time() - ep_start

        is_best = vl_loss < best_val_loss
        if is_best:
            best_val_loss = vl_loss
            torch.save(model.state_dict(), best_model_path)

        flag = " [BEST]" if is_best else ""
        print(f"Epoch {ep:02d}/{epochs:02d} | Train Loss: {tr_loss:.4f} | Val Loss: {vl_loss:.4f} | Time: {ep_sec:.1f}s{flag}")

    total_sec = time.time() - t0
    print(f"\nTraining completed in {total_sec:.1f} seconds (~{total_sec/60:.1f} mins).")
    print(f"Best Validation Loss: {best_val_loss:.4f}")

    # Save metadata
    config = {
        "architecture": "TransformerLoRA (Encoder-Decoder with Multi-Head Attention)",
        "d_model": d_model,
        "nhead": nhead,
        "num_encoder_layers": n_enc,
        "num_decoder_layers": n_dec,
        "dim_feedforward": d_ff,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "total_params": tot_p,
        "trainable_params": train_p,
        "lora_pct": lora_pct,
        "max_src_len": MAX_SRC_LEN,
        "max_tgt_len": MAX_TGT_LEN,
        "best_val_loss": round(best_val_loss, 4)
    }
    save_meta(meta_path, src_vocab, tgt_vocab, config)
    print(f"Saved model weights to: {best_model_path}")
    print(f"Saved model metadata to: {meta_path}")


if __name__ == "__main__":
    main()