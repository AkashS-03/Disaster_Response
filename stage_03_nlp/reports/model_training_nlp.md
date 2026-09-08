# Stage 03 NLP - Model Training Report

## Two-track evidence comparison (held-out test, n=6759)

| Track | Model | Macro-F1 | SEVERE recall |
| :--- | :--- | :---: | :---: |
| A (classical) | TF-IDF + LogisticRegression | 0.4242 | 0.3211 |
| B (deep) | BiLSTM + attention | 0.4005 | 0.5986 |

## Decision
Winner: **LogisticRegression (TF-IDF)** (severity_stat.joblib) — selected purely on macro-F1 (SEVERE-safe).

> Philosophy: we do not ship the flashiest model by default. The deep track
> must clearly beat the simple, interpretable one; otherwise interpretability
> and speed win. This is baseline gating applied to NLP.
