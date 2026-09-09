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
The SLM is a language model (next-word prediction) with **two uses and two verdicts**:
1. **As a language model — PASS as an assistive tool.** Trained on real data; loss 7.62 → 5.72 vs random ≈9.68; powers drafting hints and a domain-fit gauge. It never makes decisions, so it is judged on plausibility, not safety gates.
2. **As a classifier (Track C) — FAIL, does NOT ship.** This is the part I evaluate the hardest and report un-massaged.

### Track C evaluation methodology (how fairness is guaranteed)
- **Data:** the exact same real split as the shipped triage — train 27,032 for the head, and the **untouched test split (6,759)** for grading. Nothing about Track C's training sees the test set.
- **Standardisation honesty:** features are scaled using **train-only** mean/std, then applied to the test set. This is the textbook-correct way (fit on train, transform test) and is stated in the report.
- **Class weighting:** weights are computed *per class on the training split* (`len(y)/count(class)`, normalised) so the rare SEVERE class is not starved of gradient — a standard imbalance treatment, not a re-labelling cheat.
- **Baseline gating:** ships only if it **clearly beats** 0.4242 macro-F1 on the same test set.

### The numbers table to quote (from `stage_04_slm/reports/slm_head_report.md`)
| Metric | SLM Track C | Stat (shipped) | Deep (BiLSTM) |
| :--- | :---: | :---: | :---: |
| Macro-F1 | 0.3523 | **0.4242** | 0.4012 |
| SEVERE recall | 0.6746 | 0.3211 | 0.5915 |

An independent re-audit (`stage_04_slm/evaluation_engineer/eval_slm.py` →
`stage_04_slm/reports/slm_evaluation_report.md`) recomputes every number from
the shipped weights — nothing copied from the training log, and the head is
scored in deterministic eval mode (dropout off), matching production.
| Verdict | **NOT shipped** (below gate) | shipped | not shipped |

### Why I publish the SEVERE-recall nuance instead of hiding it
The most defensible position in a debate is the one that already told the truth. Track C catches more real SEVERE messages than the shipped models (0.6746 vs 0.3211 / 0.5915) — a reviewer WILL notice. I say it myself, first, and explain why it still loses the gate: macro-F1 measures *overall* triage balance; shipping a model that only excels at one class would hurt the coordination loop on every other message. Same reasoning as Track B vs Track A. **Honesty is the weapon: nothing we say can be fact-checked against our own report file and found wrong.**

### Perplexity: a gauge, not a metric under the gate
Perplexity is *exp(average surprise)* — reported in the Copilot panel as bandwidths <300 / 300–900 / >900 (recalibrated on the real test-message distribution). It is explicitly **not part of any safety gate**: it is a disclosed heuristic for "does this read like a real disaster message". Safety stays with the guard rail and human REVIEW. I state this so nobody can quote our own gauge as a safety claim.

### Disclosed limitations (say them first)
Weak 12-epoch LM · high absolute perplexity (~200–1000) · English-only tokeniser · Track C failed · perplexity bands are heuristics.

### Likely SLM questions for the Evaluation Engineer
1. **"Your Track C report admits failure — why keep the model?"** — Because "does the model ship as a classifier?" and "is the LM useful as an assistant?" are different questions. We answer them separately and honestly.
2. **"Is SEVERE recall 0.6746 hiding something?"** — No; both numbers are in the report. One-class strength does not clear a balanced gate, and hiding the nuance would be caught instantly in a live demo.
3. **"How do we know the test set is untouched by the SLM?"** — The SLM's language training used the full master corpus including test messages *for next-word learning*; Track C's *classifier head* was then trained only on the train split and evaluated only on the untouched test split. We disclose this distinction precisely because it matters: the LM is assistive; the head is what would have shipped — and it lost.
