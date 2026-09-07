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
