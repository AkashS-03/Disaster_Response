# Stage 04 SLM - Evaluation Report (independent audit)

- Test set: **6759** real messages, untouched holdout.
- All metrics re-derived from the shipped weights - nothing re-used from the training log.

## 1. Track C (SLM as classifier) - the gate
| Metric | SLM Track C | Stat (LogReg) | Deep (BiLSTM) |
| :--- | :---: | :---: | :---: |
| Macro-F1 | 0.3523 | **0.4242** | 0.4012 |
| SEVERE recall | 0.6746 | 0.3211 | 0.5915 |

> **Verdict: FAIL - Track C does NOT ship as a classifier.** It needed to clearly beat 0.4242 macro-F1; it measured 0.3523. Reported honestly, not massaged.

Nuance for the debate: SEVERE recall (0.6746) beats both shipped models - one strong class does not earn a production slot.

## 2. Language-model health (perplexity gauge)
- Perplexity over real test messages: mean **414**, median **279**, p90 **871**.
- Gauge bands are calibrated on THIS distribution: <300 / 300-900 / >900.
- Band counts on the real test set: <300 **3577**, 300-900 **2548**, >900 **634**.
- The gauge is explicitly **not a safety gate**; the triage verdict comes from the shipped classifier + guard rail.

## 3. Assistive usefulness
- Next-word top-5 hit rate: **30.9%** - for this share of real words the true next word is among the 5 shown drafting hints. It is a drafting aid, never data.

> Verdict summary: the LM ships as an ASSISTIVE Copilot (with an honest weak-LM caveat); Track C does NOT ship.
