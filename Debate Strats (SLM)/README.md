# Debate Strats — SLM (Small Language Model)

Role-wise playbook for defending our SLM work and for attacking the "other
side" in the viva cross-examination. The **golden rule everywhere**: honesty.
Real, defensible numbers beat flashy claims.

## Golden numbers (memorise these)
- SLM = **2-layer LSTM**, emb 128, hidden 128, **17.5 MB**, trains in **~8 min
  on CPU**, zero extra packages.
- Trained on **33,791 real messages** → **48,244 windows**; vocab **16,004**
  (16,000 words + 4 specials), max seq **24**.
- Training loss **7.62 → 5.72** (random ≈ 9.68 = ln 16004).
- Perplexity bands: **<300** very typical, **300–900** typical, **>900**
  unusual → keep human in the loop. (Heuristics, disclosed.)
- Track C (SLM-as-classifier): macro-F1 **0.3523**, SEVERE recall **0.6746**,
  vs gate **0.4242** → **does NOT ship**. Reported honestly in
  `stage_04_slm/reports/slm_head_report.md`.
- Copilot panel: cleaned draft (deterministic), top-5 **next-word guesses**
  labelled "SLM guess", DomainFit perplexity.

---

## Role-wise attack & defense

### Data Engineer (Role 1)
**Defend:** The SLM uses the same 33,791 real messages, no synthetic text, no
augmentation. Vocab is frequency-based with min freq 2 — junk/rare tokens
dropped. Split was done before training (test is untouched).
**Attack the scare:** "Your LLM needs BIG data." — Our SLM is not an LLM; it is
a 17 MB tool tuned to *our* domain. A domain-specific SLM needs far less data
than a general-purpose model; the honest test is whether it learned real
patterns (loss fell, perplexity sane) — not whether it could write poetry.

### EDA Engineer (Role 2)
**Defend:** Avg message ~22 words → **MAX_SEQ=24** covers ~most messages with
few pads. 48,244 windows from 33,791 messages (≈1.4 windows/message). Vocab
16,004 kept the 16,000 most frequent real words (coverage of the NLP vocab).
**Attack angle:** "24 tokens truncates long messages." — True and disclosed:
long news items are cut. But the SLM is for *drafting hints*, and greedy
windows from real text still give solid next-word signals; no metric is
claimed beyond what's in the report.

### ML / Classical Engineer (Role 3)
**Defend:** Baseline gating is consistent project-wide: nothing ships without
beating the incumbent on the real test set. The stat model still ships for
triage (macro-F1 0.4242). Track C's 0.3523 FAIL proves the gate works and
catches weak models.
**Attack the other way:** "Is deep always better?" — We have *two* pieces of
evidence that no: Track B lost macro-F1 to a bag-of-words LogReg, and Track C
(deeply reusing the SLM) lost badly. Our process lets evidence decide; that is
the defensible story.

### DL Engineer (Role 4)
**Defend:** LSTM chosen over Transformer for size/simplicity/CPU inference
(no attention to run, no positional encoding, ~17 MB, explainable). Self-
supervised next-word training = no hand labels. Dropout 0.2 + 3 epochs keeps
it honest — no memorisation.
**Attack angle:** "Why not a Transformer?" — Transformers shine with hundreds
of millions of tokens on GPU; on 33K messages on a CPU they overfit and add no
explainability. LSTM gives us a working, verifiable SLM. We state the trade-off
plainly.

### Evaluation Engineer (Role 5)
**Defend:** Everything is measured on the REAL untouched test split. Track C
verdict is written down un-massaged. Limitations (weak LM, high perplexity,
English-only tokeniser, heuristic bands) are disclosed in the report and docs.
**Attack the scare:** "Perplexity bands are made up." — They are *labelled*
heuristics calibrated from the real test-message perplexity distribution
(mean ≈ 414, median ≈ 279, p90 ≈ 871) and shown as a
relative gauge, not a claim of safety. The triage verdict is decided by the
shipped classifier + guard rail, never by perplexity.

### Integration Engineer (Role 6)
**Defend:** Copilot panel lazy-loads; if weights are missing the app says
"run slm_train.py once" — no crash. Verified with the headless AppTest harness
(zero exceptions after typing SOS text, switching engines, clicking presets).
SLM lives in Tab 4 only and can't override verdicts/guard rail/REVIEW.
**Attack the scare:** "The SLM slows the app." — Inference is one tiny LSTM
forward pass on CPU (~tens of ms); weights load once, lazily.

### NLP Engineer (Role 7)
**Defend:** SLM is complementary, not competitive. Triage answers "how
urgent?"; SLM answers "what word comes next / does this read like real
disaster text?". Both trained on the SAME real master dataset. The guard rail
+ abstention still sit between any text and a human.
**Attack the other side:** "One model should do everything." — One model
doing triage, drafting, and gauging is exactly what Track C was; it failed the
gate. Separate tools with strict roles are easier to verify, debug, and defend.

### SLM Engineer (Role 8) — the headline role
**Defend the big three:**
1. **Real data only** — no synthetic text, no paraphrase feature, outputs
   labelled "SLM guess".
2. **Assistive only** — never overrides triage/guard rail/human REVIEW.
3. **Honest failure** — Track C (SLM-as-classifier) is reported as NOT shipped
   even though its SEVERE recall (0.6746) beats both shipped models. Stage
   gating held the line.
**Attack the boogeyman:** "Replacing humans with AI." — We give a *faster
typist and a warning light*, and we keep the human on every low-confidence
message. The machine assists; the human decides.

---

## Cross-examination cheat sheet (tough questions, short answers)

**Q: "Is this just a toy? 3 epochs on 33K messages."**
A: Yes, it is deliberately small: 17 MB, CPU-only, offline — exactly what a
field deployment needs. It is verified by loss/perplexity and it never makes
decisions. Bigger is not automatically more trustworthy.

**Q: "Why should I trust something that can't even beat 0.42 macro-F1?"**
A: That's the point — we *don't* ship it as a classifier. The report says so.
Our confidence comes from the honesty of the gating, not from the model.

**Q: "Your guesses are just 'the' and 'and'."**
A: True — next-word top-5 often surfaces common function words at low %. That
is a *drafting hint*, and the panel labels it "SLM guess". We make no quality
claim beyond what the numbers in the report show.

**Q: "Is perplexity a safety feature?"**
A: No — it's a domain-fit *gauge*. Safety comes from the deterministic guard
rail and human REVIEW. We never confuse the two.

**Q: "What breaks if the SLM is wrong?"**
A: Nothing safety-critical: the verdict, guard rail, and human REVIEW are
untouched. Worst case a coordinator ignores a bad suggestion.