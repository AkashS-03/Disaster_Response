# Role 5: Evaluation Engineer

## What I Own
I independently audit the models built by the ML and DL engineers. My job is to be **adversarial and honest**: I re-run my own tests, compute my own numbers, and give a **verdict (PASS / CONDITIONAL PASS / FAIL)** with evidence. I never trust the training reports — I re-derive everything.

## Ethos: Honesty Over Polish
Disaster-response models are safety-critical. A fabricated "PASS" could cost lives in a debate or real deployment. Where the model falls short, I **say so clearly** (e.g., CONDITIONAL PASS, 157 catastrophic underestimates, edge-case imagery pending). This honesty is our strongest debating weapon — no other team will have real, defensible evaluation.

## Stage 01 ML Evaluation

### Methods
| Tool | What it checks |
| :--- | :--- |
| `model_evaluation_report.md` | Global accuracy, per-class precision/recall/F1, confusion matrix |
| `edge_case_test.py` | Model behaviour at every decision boundary (riv 3.0 / 4.5, rain 80/150, calls 100) |
| `calibration_test.py` | Does the predicted confidence match reality? (over-confidence) |
| `overconfidence_detector.py` | Flags over-confident wrong predictions |
| `cross_validation.py` | Robustness across folds |
| `tests/test_evaluation.py` | 10 unit tests that self-check my evaluation code |

### The GuardRail — what I verified
I confirmed the **GuardRailedPredictor** pushes **edge-case accuracy from 52.94% (9/17) → 100% (17/17)**. The safety thresholds are now guaranteed by construction, not left to chance.

### Honest finding: MODERATE recall artifact
Low MODERATE recall (≈1%) is **not a model failure** — every MODERATE test row has `river_level = 20.0m`, which the decision rules call SEVERE. The test *labels* are inconsistent with the rules. I disclose this rather than hide it.

## Stage 02 DL Evaluation

### Forecast (`eval_forecasting.py`) — Residual FloodLSTM
- **Baseline gating:** re-computed naive & XGBoost; LSTM must beat them.
- **Result:** LSTM MAE **0.4462** vs naive **0.6609** (+32.5%), beats naive **72.56%**.
- **Worst-case:** 157 catastrophic underestimates when true level >6.0 under-predicted by >2.5 → flagged.
- **Multi-horizon gap:** 3h/6h models don't exist — a real gap, **noted, not faked**.
- **Verdict: CONDITIONAL PASS.**

### Vision (`eval_vision.py`) — MobileNetV2
- **Confusion matrix:** TP=650, FN=31, TN=694, FP=25 on 1,400 held-out images.
- **Flooded recall 95.45%**, clear recall **96.52%**, accuracy **96.00%**.
- **Edge-case imagery:** honest note — real wet-asphalt/puddle edge images need extra labeling; **not fabricated**.
- **Verdict: PASS.**

## Scorecard Integrity Fix
Originally the scorecard hardcoded fabricated numbers and PASS verdicts. I fixed it to **read the actual verdicts from the independent reports** — the forecast now correctly shows **CONDITIONAL PASS**, not false PASS.

## Likely Viva Questions
1. **Why not trust the training report?** — Training metrics can be inflated; independent re-derivation is the only honest signal.
2. **What's an edge-case test?** — Testing exactly at decision thresholds (river 2.99 vs 3.01) where a small error flips the label — where safety lives.
3. **Why CONDITIONAL PASS for the forecast?** — Beats naive on MAE/fraction but 157 catastrophic underestimates + missing 3h/6h horizons.
4. **Why correct fabricated edge cases?** — Fabrication would be caught in a live demo; real, defensible numbers win the debate.
5. **What is baseline gating?** — A model is only useful if it beats the trivial baseline (e.g., "predict it stays the same"); we test that explicitly.

## SLM: Detailed Explanation & My Role Facts (Role 8 tie-in)

### What the SLM is (evaluation view)
The SLM is a **Severity-Conditioned Tactical Briefing Summarizer**. My role as Evaluation Engineer is to run an **uncompromising, independent audit** on a held-out test set of 240 disaster incident reports, benchmarking summary fidelity, rule compliance, key factor extraction accuracy, and CPU inference latency under load.

### The Independent Evaluation Audit (`eval_slm.py`)
I evaluate the shipped checkpoint (`stage_04_slm/models/slm_briefing.pth`) directly in deterministic evaluation mode (`eval()`), generating briefings for 240 unseen incident logs.

### The Benchmark Scorecard (from `reports/slm_evaluation_report.md`)
| Metric | Shipped SLM Result | Benchmark Target | Verdict |
| :--- | :---: | :---: | :---: |
| **ROUGE-1 F1** | **0.4701** | > 0.4500 | **PASS** |
| **ROUGE-2 F1** | **0.2675** | > 0.2500 | **PASS** |
| **ROUGE-L F1** | **0.4453** | > 0.4000 | **PASS** |
| **BLEU-2 Score** | **0.3241** | > 0.3000 | **PASS** |
| **Sentence Length Compliance** | **100.0%** | > 95.0% | **PASS** |
| **Location Extraction Accuracy** | **100.0%** | > 80.0% | **PASS** |
| **Risk Level Accuracy** | **95.0%** | > 80.0% | **PASS** |
| **Mean CPU Latency** | **85.7 ms** | < 300 ms | **PASS** |
| **P95 Latency** | **141.0 ms** | < 500 ms | **PASS** |

### Why 100% Rule Compliance Matters
In an emergency control room, format compliance is a safety feature:
- If a commander expects a 1-sentence summary for a MODERATE incident and receives 3 rambling sentences, they lose precious time.
- If a SEVERE alert omits the immediate directive (Sentence 2), rescue assets are not mobilized.
- Our dual-layer audit proves **100% of LOW alerts contain 0 periods**, **100% of MODERATE alerts contain 1 period**, and **100% of SEVERE alerts contain exactly 2 periods**.

### Disclosed Nuances & Limitations
- Vocabulary is domain-constrained to 2,500 source and 2,500 target tokens; unlisted colloquial phrasing outside the domain vocabulary maps to `<unk>`.
- Latency is measured on a standard laptop CPU across 100 iterations (mean 85.7ms, P95 141ms). Under extreme thermal throttling, latency may peak up to ~250ms, which remains safely below the 500ms operational bound.

### Likely SLM questions for the Evaluation Engineer
1. **"How do you prevent data leakage during evaluation?"** — The 240 evaluation pairs were held out from all training epochs. The tokenizer vocabularies and scaling constants were computed strictly from the training split.
2. **"Why use ROUGE and BLEU instead of pure human ratings?"** — ROUGE and BLEU provide objective, mathematically reproducible overlap benchmarks across n-grams and longest common subsequences. We supplement them with deterministic rule tests (period counting and entity matching).
3. **"What is the final shipping verdict for Stage 04?"** — **SHIP [SUCCESS]**. The model passes all 7 gating thresholds, achieves 100% sentence compliance, and executes in 85.7 ms offline on consumer CPU hardware.

