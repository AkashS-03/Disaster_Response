# Role 8: SLM Engineer

## What I Own
I built our own **Small Language Model (SLM)** — a compact word-prediction
neural network trained **entirely on the real disaster corpus** (no synthetic
text anywhere). It powers the **Copilot panel** in Tab 4 of the dashboard:
deterministic text hygiene, top-5 **next-word guesses** (clearly labelled
"SLM guess"), and a **DomainFit (perplexity)** gauge that warns when a message
uses unusual wording. The SLM is **assistive only** — it never overrides the
triaged verdict, the guard rail, or a human REVIEW.

I also attempted a **Track C** classifier (the SLM reused to triage messages),
evaluated it honestly with the same **baseline gating** as every other track,
and the evidence says it **does NOT ship**. I report that openly.

## Problem
Given a partial message, help the coordinator type faster and judge whether a
message "reads like" a real disaster report — without fabricating anything.

## Why a self-built SLM instead of a downloaded LLM?
- **Offline**: floods kill networks; a big model needs the internet to run
  (or a giant download). Ours is 17 MB, on-device.
- **CPU-only**: plain laptop, PyTorch only, no `transformers`/`sentencepiece`.
- **Explainable**: every layer is our own code — perfect for the debate.
- **Trust + honesty**: trained on OUR data, we know exactly what it saw.

## Data (real data only)
- **Training text:** the same master text dataset as the triage model —
  **33,791 real messages** (Figure Eight 26,180 + Kaggle 7,611).
- **No synthetic text** is generated, injected, or used for augmentation.
  Suggestions in the app are labelled **"SLM guess"** and never fed back.
- Vocabulary: **16,004** tokens = 16,000 most frequent words (min frequency 2)
  + 4 specials `<s>/</s>/<pad>/<unk>`.

## The model (self-built)
| Parameter | Value |
| :--- | :---: |
| Architecture | 2-layer **LSTM** language model |
| Embedding | 128-dim, learned |
| Hidden | 128 units per layer |
| Sequence length | 24 tokens (windows from real messages) |
| Loss | CrossEntropy (predict next word), pads ignored |
| Optimizer / LR | Adam / 1e-3, batch 512, 12 epochs |
| Training windows | 48,244 (from real messages only) |
| **Parameters** | ~4.38M (≈ 17.5 MB fp32 on disk) |
| **Files** | `stage_04_slm/models/slm_lstm.pth` (~17.5 MB) + `stage_04_slm/models/slm_lm_meta.json` |
| Training loss | 7.62 → **5.72** (random guess would be ~9.68 = ln 16004) |

**Architecture spec (count it in the code):** Embedding `16,004 × 128`
(2,048,512 params) → 2×LSTM hidden 128 (264,192) → `Linear(128 → 16,004)`
(2,064,516) = **4,377,220** total. Tokeniser `[a-z0-9']+`; window
`<s> words </s>` ≤ 24 tokens; `encode(x)` = hidden state at the last real
token (128-dim, frozen for reuse). Track C head = `Linear 128→ReLU→Dropout(0.3)
→Linear 3` (16,899 params), class-weighted CE, train-only standardisation. All
config lives in `slm_lm_meta.json`, so the architecture is reproducible from
the artifact alone. Full beginner walkthrough: `Explainations/SLM_Explained.md`.

**Intuition to explain:** the model's only skill is "guess the next word". At
every position in every real message it is asked this, and it gets slightly
less surprised over 12 epochs — so it has learned word patterns like
*"trapped on the roof"*.

## What it powers in the dashboard (Copilot panel, Tab 4)
| Feature | Mechanism | Honesty rule |
| :--- | :--- | :--- |
| **Cleaned draft** | deterministic whitespace/capitalisation tidy-up — no new words | never rewrites/generates content |
| **Next-word guesses** | softmax over vocab at last position → top-5 with % | labelled "SLM guess" |
| **DomainFit (perplexity)** | exp(avg next-word NLL); bands <300 / 300–900 / >900 | "gauge only, never overrides the verdict" |

Perplexity is our **honesty gauge**: low = the wording is familiar to a model
trained on real disaster text; high = unusual/out-of-domain → "keep the human
in the loop". Bands are disclosed heuristics. If the weights are missing the
panel says *"run slm_train.py once"* — the app never crashes.

## Track C — the honest classifier attempt (does NOT ship)
Froze the SLM, encoded each real message (hidden state at last real token),
standardised features with train-only statistics, and trained a small classed-
weighted head (Linear 128 → ReLU → Dropout → Linear 3).

| Metric | SLM Track C | Stat (shipped) | Deep (BiLSTM) |
| :--- | :---: | :---: | :---: |
| Macro-F1 | 0.3523 | **0.4242** | 0.4012 |
| SEVERE recall | 0.6746 | 0.3211 | 0.5915 |

**Verdict: baseline-gated FAIL → Track C does not ship**
(`stage_04_slm/reports/slm_head_report.md`, reported openly, not massaged).

**Debate-worthy nuance:** Track C SEVERE recall (0.6746) beats both shipped
models — but macro-F1 is far below the 0.4242 gate. Same lesson as Track B vs
Track A: *one good number does not earn a production slot.*

## Honest limitations I state unprompted
1. Weak LM — 12 epochs on 33K messages; guesses are often common function words.
2. Overall perplexity is high (~200–1000) — good for relative gauging, not a
   quality boast.
3. It predicts words; it cannot paraphrase, answer, or "understand" facts.
4. Tokeniser `[a-z0-9']` drops non-Latin text and emojis.
5. Track C failed; we do not pretend otherwise.
6. Perplexity bands are heuristics, disclosed as such.

## Likely Viva Questions (Role 8)
1. **Why build an SLM instead of using an LLM?** — Offline, CPU-only, 17 MB,
   every layer explainable, trained only on our real data. Big models need
   internet/GPU and are unexplained black boxes even to their users.
2. **Is your SLM generating synthetic text?** — No. It only predicts the next
   word and its output is labelled "SLM guess". We deliberately declined a
   "paraphrase/rewrite" feature because that would create synthetic text.
3. **How did you train it with no labels?** — Next-word prediction is
   self-supervised: the "answer" is simply the next real word of each real
   message. 33,791 messages → 48,244 windows.
4. **What is perplexity?** — exp(average surprise). It tells us how "familiar"
   the wording is to a model trained on real disaster text; used as a gauge,
   never to change the verdict.
5. **Did Track C ship?** — No. Macro-F1 0.3523 vs the 0.4242 gate. We report
   the honest verdict. Its SEVERE recall (0.6746) is a discussion point, not a
   free pass.
6. **Does the SLM ever decide a message's severity?** — Never. It is assistive
   in Tab 4 only, behind the guard rail and human REVIEW.
7. **Why tie it to the Copilot panel at all?** — Offline next-word drafting
   speeds up report writing and its perplexity acts as a domain-fit warning,
   both useful in a coordination loop where connectivity is unreliable.