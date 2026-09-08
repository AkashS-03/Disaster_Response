# Stage 03 NLP - Evaluation Report (independent)

- Test set: **6759** real messages, untouched holdout.
- Shipped model: `severity_stat.joblib` (winner by macro-F1, baseline-gated).

## Track comparison (independent re-derivation)
| Track | Macro-F1 | SEVERE recall |
| :--- | :---: | :---: |
| TF-IDF + LogReg | 0.4242 | 0.3211 |
| BiLSTM + attention | 0.4012 | 0.5915 |

> Debatable result: macro-F1 prefers the simple model; SEVERE recall prefers the deep model.
> We ship the interpretable model + deterministic guard rail + human-in-the-loop abstention.


## Abstention sweep (statistical core, guard-first)
| threshold | coverage % | macro-F1 | SEVERE-R | auto-acc |
| :---: | :---: | :---: | :---: | :---: |
| 0.70 | 40.7 | 0.3768 | 0.2127 | 0.4113 |
| 0.65 | 44.8 | 0.4053 | 0.2282 | 0.4366 |
| 0.60 | 50.0 | 0.4195 | 0.2366 | 0.4527 |
| 0.55 | 57.4 | 0.4262 | 0.2662 | 0.4608 |
| 0.50 | 67.4 | 0.4197 | 0.2831 | 0.4585 |
| 0.45 | 80.4 | 0.4129 | 0.3099 | 0.4551 |
| 0.40 | 92.6 | 0.4044 | 0.3563 | 0.4452 |
| 0.35 | 99.5 | 0.3979 | 0.3817 | 0.4376 |
| 0.30 | 100.0 | 0.3972 | 0.3831 | 0.4368 |
| 0.25 | 100.0 | 0.3972 | 0.3831 | 0.4368 |

**Operating point:** threshold = 0.5 (messages below it go to human triage).

## Guard-rail effect
- Keyword floor fired on **2251/6759** test messages, forcing at least MODERATE/SEVERE.
- At threshold 0.5: auto coverage 67.4%, macro-F1(auto) 0.4197, auto-accuracy 0.458, SEVERE handled (auto+human) 0.85.

## Verdict gates
- SEVERE handled (auto+human) >= 0.70: OK
- macro-F1 (auto) >= 0.45: MISS
- auto-accuracy >= 0.55: MISS
- auto coverage >= 55%: OK

> **Final verdict: CONDITIONAL PASS** — the statistical model alone is weak, but with the guard rail + abstention it is usable in the coordination loop. Label noise (derived severity) is disclosed as a limitation.
