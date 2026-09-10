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

## Entity clarification — what else I pull from a message
Triage class + confidence answers *"how urgent?"* — but a coordinator also needs *"what does this message actually describe?"*. So the NLP agent exposes a lightweight **entity extractor** (`extract_entities`, served to the dashboard as the **Detected Entities** panel):

| Entity type | Examples pulled | How |
| :--- | :--- | :--- |
| **People counts** | `150 people`, `3 children`, `30 families`, `hundreds of villagers` | regex on *number ≈ unit / approximate-word ≈ unit* |
| **Location** | `roof of the colony`, `Bandra station`, `eastern ward`, `slum near Dadar` | preposition-phrase regex (`in/at/near/on/…`) + road/ward/colony keyword + known Mumbai areas |
| **Key details** | `rescue intervention needed`, `medical help reported`, `vulnerable group present`, `supplies / aid requested`, `evacuation / shelter needed`, `infrastructure affected`, `casualties reported` | seven keyword rules → human-readable chips |

Design notes worth defending at the viva:
- **No external NER dependency** (no spaCy/transformers on a CPU-only machine) — it's deterministic regex/keyword logic, so **every extraction is explainable and reproducible** (count it with your own hands). The cost: it misses unlisted phrasings that a learned NER would catch — an honest trade-off we disclose.
- The extractor is **complementary to the model**, not part of it: the verdict never changes because of what entities were found. It's a "what + where + how many" layer on top of the "how urgent" verdict.
- Empty result is a meaningful signal: the message gives no actionable population/location/detail, so the coordinator knows not to dispatch on guesswork.

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
7. **What is your entity extractor and why not real NER?** — `extract_entities` pulls people counts, locations, and key details from a message (rendered as chips in the dashboard) using deterministic regex/keyword rules. No spaCy/transformers means no heavy CPU dependency and every extraction is explainable and reproducible; we accept that unlisted phrasings are missed and disclose that limit.

## SLM: Detailed Explanation & My Role Facts (Role 8 tie-in)

### What the SLM is (NLP view)
The SLM (Stage 04) is a **Severity-Conditioned Tactical Briefing Summarizer** that complements my Stage 03 **NLP SOS Triage Classifier**. They address two distinct operational bottlenecks in the disaster response lifecycle:
- **Stage 03 NLP (Tab 4): Citizen-to-Responder Triage.** Takes unstructured, noisy incoming SOS messages from the public, detects urgency (LOW/MODERATE/SEVERE/REVIEW), enforces safety guard rails, and extracts entities (people count, location chips).
- **Stage 04 SLM (Tab 5): Responder-to-Commander Tactical Briefing.** Takes multi-page, multi-unit incident dispatch reports and summarizes them into structured, length-constrained spoken briefs (<1 sentence for LOW, 1 sentence for MODERATE, strictly 2 sentences for SEVERE) while highlighting key situational factors.

### Operational Division of Labor
| Dimension | Stage 03: NLP Triage (Role 7) | Stage 04: Tactical SLM Briefing (Role 8) |
| :--- | :--- | :--- |
| **Input** | Single incoming SOS message / tweet (~22 words) | Multi-unit tactical dispatch report (~158 words) |
| **Task** | Classification (Severity Class + Abstention) | Sequence-to-Sequence Severity-Conditioned Summarization |
| **Model** | TF-IDF + Logistic Regression (Interpretable winner) | TransformerLoRA ($r=8, \alpha=16$, Multi-Head Attention) |
| **Output** | Triage Card + Confidence + Guard Rail + Entity Chips | Key Factors Box + Brevity Summary + Offline Voice Synthesis |
| **Dashboard Tab** | **Tab 4 ("💬 Stage 03: NLP Message Triage")** | **Tab 5 ("🎙️ Stage 04: SLM Severity Summarizer")** |

### How Stage 03 and Stage 04 Interface
1. **Severity Handoff:** In the command workflow, the severity predicted by Stage 03 triage (or aggregated across field dispatches) directly sets the severity condition token (`summarize sev <level>:`) for the Stage 04 SLM.
2. **Entity Consistency:** Both stages share domain-dictionary vocabulary rules (`PRI-1`, `MEDEVAC`, `LZ-CLEAR`), ensuring emergency codes identified during triage remain intact in the commander's voice briefing.
3. **Safety Separation:** The SLM cannot change a triage verdict; Stage 03 guard rails remain sovereign over classification.

### Likely SLM questions for the NLP Engineer
1. **"Why not use one single model for both triage classification and summarization?"** — As our earlier experiments demonstrated, multi-task models often underperform specialized architectures on critical disaster edge cases. A lightweight linear model with deterministic keyword guard rails is provably safer and more interpretable for triage (Tab 4), while a dedicated Sequence-to-Sequence Transformer with cross-attention excels at length-constrained briefing generation (Tab 5).
2. **"How do the tokenizers compare between Stage 03 and Stage 04?"** — Stage 03 uses TF-IDF word n-grams tuned for discriminative keyword frequency. Stage 04 uses dual domain-pruned vocabularies (2,500 tokens each) with atomic preservation of tactical codes and control tokens (`<s>`, `</s>`, `<pad>`, `<unk>`).
3. **"Does the SLM hallucinate details not found by Stage 03 triage?"** — No. Stage 04 is evaluated for factual fidelity with a 100% location extraction accuracy and 95% risk level accuracy on held-out test data.