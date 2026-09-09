# Stage 04 SLM - SLM Track C report (real data only)

The SLM (Small Language Model) is a compact 2-layer LSTM that learned the NEXT-WORD distribution of the real disaster-message corpus. Track C is a small classifier head trained on TOP of the SLM's frozen encoding to see whether the same SLM can also triage messages. Everything here is trained and tested on REAL data.

## What was trained

- **Language model (LM):** `models/slm_lstm.pth` + `models/slm_lm_meta.json`, 2-layer LSTM (emb 128 / hidden 128), vocabulary 16004 tokens, trained on all 33791 real messages (next-word prediction, 12 epochs, average loss 7.62 -> 5.72; random guessing would be ~9.68).
- **Track C head:** `models/slm_head.pth`, Linear 128->ReLU->Dropout->Linear 3, trained on the frozen SLM encoding of the REAL train split. Features standardised (fit on train only) and class-weighted because SEVERE is only 10.5% of the data.

## Evaluation (real untouched test split)

| Metric | SLM Track C | Stat (LogReg) | Deep (BiLSTM) |
| :--- | :---: | :---: | :---: |
| Macro-F1 | 0.3523 | **0.4242** | 0.4012 |
| SEVERE recall | 0.6746 | 0.3211 | 0.5915 |

## Verdict

> **Track C does NOT ship.** Baseline gating: it needed to clearly beat 0.4242 macro-F1 on the same real test set; it reached 0.3523 (deterministic eval mode). The verdict is reported honestly, not massaged.

Notable observation for the debate: the SLM's SEVERE recall (0.6746) beats the shipped models (stat 0.3211, deep 0.5915). That is the single most important class in disaster triage, yet the head's overall discrimination is too weak to clear the quality gate. This is exactly what stage gating is for: a headline number on one class does not earn a production slot.

The SLM itself still ships as a LANGUAGE MODEL (next-word suggestions + perplexity domain-fit), where its role is assistive and it never overrides the shipped classifier.
