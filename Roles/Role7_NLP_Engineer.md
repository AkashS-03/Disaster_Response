# Role 7: NLP Engineer

## What I Own
I build the **Stage 03 NLP triage system** — the text agent. It reads real emergency messages (tweets, direct reports, news) and triages each one as **LOW / MODERATE / SEVERE**, so a coordinator knows which messages are urgent without reading thousands manually.

I train **two different model tracks** on the same data and let held-out evidence decide which one ships (**baseline gating** — we do not ship the flashiest model, we ship the best-evidence one). On top of the winner I add two safety layers: a **deterministic keyword guard rail** (model can never under-warn below hard rules) and **human-in-the-loop abstention** (unsure messages are sent to a human, not guessed).

## Problem
Given a free-text emergency report, produce a triage class. Classes: `["LOW", "MODERATE", "SEVERE"]`.

## Data
- **Master text dataset**: 33,791 real labelled messages.
  - Source 1: **Figure Eight** crowd-sourced disaster responses (26,180) — genres *news* (13,039), *direct* (10,745), *social* (2,394).
  - Source 2: **Kaggle "Disaster Tweets"** (7,611).
- Labels are **derived severity** via disclosed rules: SEVERE = search/rescue, medical help, death, missing people; MODERATE = floods/storms/etc + Kaggle disaster flag; LOW = everything else. We disclose this honestly — near-news text is genuinely hard to label.
- Class split: **LOW 15,908 (47.1%) / MODERATE 14,335 (42.4%) / SEVERE 3,548 (10.5%)** — SEVERE is ~4.5× rarer, an imbalanced set.
- Stratified **80/20 split** → train **27,032** / test **6,759** (both real, untouched).
- EDA (avg message ~22 words, vocab ~50K tokens) → `reports/figures/nlp_eda.png`.

## Track A (SHIPPED): TF-IDF + LogisticRegression — classical, interpretable
| Parameter | Value |
| :--- | :---: |
| Vectorizer | TF-IDF, n-grams **(1,2)**, min_df = 5, sublinear_tf, max 200K features |
| Classifier | **LogisticRegression** (won internal validation over LinearSVC) |
| class_weight | **balanced** (handles the rare SEVERE class) |
| solver / max_iter | lbfgs / 2000 |
| **Model file** | **1.4 MB** (`severity_stat.joblib`) |
| Test macro-F1 | **0.4242** |
| Test SEVERE recall | **0.3211** |

## Track B: BiLSTM + attention — deep learning
| Parameter | Value |
| :--- | :---: |
| Input | word indices up to **64 tokens** |
| Embedding | **128-dim** (learned, with pad index) |
| Recurrence | **BiLSTM**, hidden **128**, bidirectional (→ 256-dim states) |
| Attention | learned **self-attention pooling** over non-pad timesteps (gives word-level explainability) |
| Dropout / FC | 0.3 → Linear(256 → 3) |
| Loss | CrossEntropyLoss with **inverse-frequency class weights** |
| Optimizer / LR | Adam / 1e-3 |
| Epochs / Batch / Early stop | 8 / 128 / patience 3 |
| Gradient clip | max_norm 1.0 |
| **Model file** | **9.06 MB** (`bilstm_severity.pth` + `bilstm_vocab.json`) |
| Test macro-F1 | **0.4012** |
| Test SEVERE recall | **0.5915** |

## Why Track A ships (the debatable result)
- **macro-F1 prefers the simple model** (0.4242 vs 0.4012) — overall triage balance is better classical.
- **SEVERE recall prefers the deep model** (0.5915 vs 0.3211) — the BiLSTM catches more real SEVERE messages.
- Baseline-gating philosophy: the deep model must **clearly beat** the interpretable one to justify its extra complexity. It doesn't on the headline metric — so we ship interpretability + a **safety guard rail** to cover the SEVERE gap.

## Safety layers on top of the statistical model
1. **Deterministic keyword guard rail** — hard floors evaluated FIRST, can only *escalate* (never downgrade):
   - SEVERE floor: *trapped, drowning, rescue, stranded, injured, buried, missing people, need help, SOS, dead, baby*, … → forced **SEVERE**.
   - MODERATE floor: *flood, storm, earthquake, landslide, evacuat, shelter, road closed*, … → forced **MODERATE** (or SEVERE if the model already said so).
   - Fired on **2,251 / 6,759** test messages.
2. **Human-in-the-loop abstention** — if the model's top confidence is below **threshold 0.50**, it returns **REVIEW** and the message goes to a human operator instead of a guessed class. The dashboard renders this as a distinct purple "HUMAN TRIAGE REQUIRED" state.

## Results (independent evaluation, n=6,759)
| Quantity | Value |
| :--- | :---: |
| Statistical core macro-F1 / SEVERE recall | 0.4242 / 0.3211 |
| Deep track macro-F1 / SEVERE recall | 0.4012 / 0.5915 |
| Guard-rail floors fired | 2,251 / 6,759 |
| Abstention threshold | 0.50 (below → REVIEW) |
| Auto coverage (messages we answer) | **67.4%** |
| Macro-F1 on auto-answered | 0.4197 |
| Auto accuracy | 0.458 |
| **SEVERE handled (auto + human)** | **0.85** |

**Verdict: CONDITIONAL PASS.** The statistical model alone is weak — but guard-rail floors catch the dangerous keywords and abstention routes uncertainty to humans, making the system usable in the coordination loop. Label noise (derived severity) is disclosed, not hidden.

## Likely Viva Questions
1. **Why not ship the BiLSTM?** — Baseline gating: the deep model must clearly beat the simple one on the headline metric. It wins on SEVERE recall but loses macro-F1, so interpretability + guard rail win; we disclose the trade-off.
2. **What is the guard rail for text?** — Hard keyword floors (trapped/rescue/flood/…) evaluated first that can only escalate a severity class, never downgrade it.
3. **What is abstention?** — If the model is unsure (confidence < 0.50) it says REVIEW and a human decides. Better than guessing on emergency messages.
4. **What is attention in your BiLSTM?** — A learned weighting over the words: it tells us which words drove the decision — an explainability handle.
5. **How do you handle imbalance?** — `class_weight="balanced"` (classical) + inverse-frequency loss weights (deep) + over-sampling of SEVERE by the guard rail.
6. **Why CONDITIONAL PASS?** — Raw model metrics miss the auto-accuracy/macro-F1 gates; guard rail + abstention make it safe enough to ship as a triage aid, disclosed honestly.