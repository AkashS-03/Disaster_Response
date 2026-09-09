# Stage 04 SLM - EDA Report (real data only)

- **Corpus:** 33791 real labelled disaster messages (master text dataset).
- **Tokens/message:** mean 22.5, median 20, p90 36.
- **Vocabulary used by the SLM:** 16004 tokens (min_freq=2, capped at 16000).
- **`<unk>` rate:** 5.09% (coverage 94.91%) - rare words fall back to <unk>; acceptable because the SLM only gives drafting hints.
- **Next-word windows:** 48244 real windows, of which 31.3% padding - the 24-token window is a good fit for the message-length distribution above.
- **Imbalance:** SEVERE is 10.5% of messages (minority). We keep class-weighted heads and the deterministic guard rail, exactly like Stage 03.

> EDA takeaway: the real corpus is naturally imbalanced and short (median ~22 tokens); the SLM's tokeniser/window choices lose little to <unk> and padding. Honest numbers, no synthetic text anywhere.
