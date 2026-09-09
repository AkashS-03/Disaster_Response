# Stage 03 — NLP (Text Triage): Explained From The Ground Up

*For beginners. No jargon assumed — every concept is built up from a simple idea.*

---

## 1. What NLP is doing in this project

NLP = making a computer understand/classify **text**. In the app there's extra noise: victims and volunteers send text reports ("water entered my house", "building collapsed, people trapped"). A coordinator can't read thousands of them. So we built an agent that reads a message and stamps it with one of three severity levels:

- **LOW** — informational, no action
- **MODERATE** — real hazard or aid need
- **SEVERE** — life-threatening, needs an urgent rescue response

That's a **classification** problem: an input (text) → one of 3 categories.

---

## 2. The data it learned from (33,791 real messages)

A model is useless without examples. We trained on real messages:
- **26,180** from **Figure Eight** — a crowd-sourced disaster-response dataset (news reports, direct help requests, social posts).
- **7,611** from the **Kaggle "Disaster Tweets"** dataset.

Each message needed a label. Two honest options: pay humans to label all 33,791 (expensive) or **derive** the labels with rules. We derived them and **disclosed** it — never hide this. Proper label engineering:

| Label | Rule (keyword-based) | Messages |
| :--- | :--- | :---: |
| SEVERE | contains rescue/medical/death/missing terms | 3,548 (10.5%) |
| MODERATE | flood/storm/etc + disaster tweet flag | 14,335 (42.4%) |
| LOW | everything else | 15,908 (47.1%) |

Notice **SEVERE is rare** (~1 in 10). That's **class imbalance**, and it's exactly the dangerous class we care most about.

We split the data **80/20** *before* any training: **27,032** messages for training, **6,759** locked away as the test set that neither model ever touches until graded. This is the only honest way to know whether a model generalises — it's the **exam**, not the **homework**. If a model cheated by seeing exam answers during training, it would score well in practice but fail in real life.

---

## 3. The awkward problem: computers don't read words

Before any model, text must become **numbers**. A human parses "flood water rising fast" instantly; a computer sees a string of letters. So there's a prep stage:

1. **Clean** — lowercase, tidy spacing ("Flood!" → "flood").
2. **Tokenise** — split into word pieces: `["flood", "water", "rising", "fast"]`.
3. **Vocabulary** — identify the ~50,000 tokens that matter and give each an ID.

Now the message is a sequence of numbers. But **how** we turn words into numbers is where the two "tracks" differ.

---

## 4. Track A — the classical model (the one we shipped)

### Step 1: TF-IDF — turning words into "how important is this word"

A simple word *count* is dumb: "the" and "a" appear constantly and carry zero information. TF-IDF fixes that:

- **TF** (Term Frequency): how often a word appears in the message — more often = somewhat more relevant.
- **IDF** (Inverse Document Frequency): a word that appears in **few** messages is rare and distinctive → it gets a **big** weight; a word in almost every message (like "the") gets ~**zero**.

**Plain English intuition:** one copy of *"trapped"* tells you more than ten copies of *"the"*. TF-IDF makes the computer agree. We used unigrams **and** bigrams (groups of 1 and 2 words), so the model sees both `"help"` and `"please help"`.

The result: every message becomes a long vector of numbers — one slot per distinctive word (up to 200,000 slots), mostly zeros, with high values where the rare/meaningful words are.

### Step 2: Logistic Regression — learning "which words point to which class"

Logistic Regression is the simplest honest classifier. Think of it as a **weighted vote**:

- Every word gets a learned weight towards each class.
- `"trapped"` pushes strongly toward SEVERE.
- `"school"` pushes toward LOW.
- The model **adds up** the contributions of all the words and outputs **three probabilities**, e.g. `{LOW: 0.10, MODERATE: 0.25, SEVERE: 0.65}`, then picks the biggest.

Why "regression"? Because inside, it computes a score and turns scores into probabilities. Why is it **interpretable**? Because we can inspect the weights: *"the word 'trapped' contributed +X toward SEVERE."* That transparency is gold in a safety-critical system. It is also tiny (a 1.4 MB file) and instant to run.

We handled the SEVERE imbalance with `class_weight="balanced"` — the model **pretends SEVERE is more common** than it is, so it tries harder on the rare class instead of lazily predicting "LOW".

**Result on the untouched test set:** macro-F1 **0.4242**, SEVERE recall **0.3211**.

---

## 5. Track B — the deep model (BiLSTM + attention)

This one thinks like a brain, word by word, *in order*.

### Word embeddings — words become coordinates

Instead of a giant 0/1 vector per word, each word gets a small **learned vector (128 numbers)** encoding *meaning* — words used in similar ways end up close together. `"flood"` sits near `"water"`, `"surge"`, `"inundation"`. The model learns these vectors from our data during training.

### LSTM — a network with a memory

Word order matters, so a "bag of words" approach misses the story. An **LSTM (Long Short-Term Memory)** is a **recurrent neural network**: it reads the message one word at a time and keeps an internal **memory state** that updates as it goes. It has "gates" that decide what to remember (the flood context) and what to forget (noise), so by the end it holds a summary of the whole sentence.

**BiLSTM** = **two** LSTMs running in parallel — one reading left→right AND one reading right→left. A message's meaning often depends on words *after* the current one ("…if you don't evacuate"). The right-to-left pass adds context the left-to-right pass misses. Both passes' outputs are joined, so each word is understood in the light of the *whole* message.

### Attention — "which words matter?"

Attention is a learned **spotlight**. The model computes a weight for every word and takes a weighted average: `"trapped"` or `"rescue"` → high weight (drives the decision); `"kindly"`, `"please"` → low weight. Two bonuses:

1. It **improves performance** — the model focuses on the decisive words.
2. It is an **explainability handle** — we can show *which words* the model looked at.

### Class weights (countering imbalance)

The loss function **penalises a wrong SEVERE prediction ~3× more** than a wrong LOW prediction (inverse-frequency weighting: SEVERE weight 3.175 vs LOW 0.708). Missing a rare SEVERE message hurts training more, so the network can't ignore the class.

**Result on the same untouched test set:** macro-F1 **0.4012**, SEVERE recall **0.5915**.

---

## 6. The debatable result — and why the simple model shipped

| Track | Model | Macro-F1 | SEVERE recall |
| :--- | :--- | :---: | :---: |
| A | TF-IDF + Logistic Regression | **0.4242** | 0.3211 |
| B | BiLSTM + attention | 0.4012 | **0.5915** |

- **Macro-F1** (average of precision & recall across all classes, equal weight) says the **simple model** is better overall.
- **SEVERE recall** (of all *real* SEVERE messages, how many we caught) says the **deep model** is better at the dangerous class (0.59 vs 0.32).

Both numbers sit in the same moderate range — this is **hard** data, no free lunches.

**Our rule: baseline gating.** The deep model must **clearly beat** the interpretable one to justify its complexity, its 9 MB file, slower inference, and its black-box explainability. It didn't win the headline metric. So we shipped **Logistic Regression** — interpretable, tiny, fast — and **disclosed** the trade-off. Differentiator: *"we don't ship the flashiest model, we ship the best-evidenced one."*

---

## 7. The safety layers (why the system is safe even though the models are mediocre)

A raw 0.42 macro-F1 in a disaster system is not acceptable to deploy naked. So we put **two guard rails** around it.

### Layer 1 — deterministic keyword floors (can only escalate)

Independent of the model, we have fixed keyword lists: text containing `trapped / drowning / rescue / baby / SOS…` → severity forced **at least** SEVERE. Text containing `flood / earthquake / evacuat / shelter…` → forced **at least** MODERATE.

This layer can **only escalate, never downgrade**. Even if the statistical model says "LOW", the keyword floor bumps it up. It mirrors Stage 01's `GuardRailedPredictor` philosophy: *the system can never under-warn below a hard rule.* It fired on **2,251 of 6,759** test messages — real work, not decoration.

*(Why "at least"? A message containing "flood" AND "baby trapped" should correctly be SEVERE even though the "flood" floor only says MODERATE — the higher of the two wins.)*

### Layer 2 — abstention (human-in-the-loop)

Sometimes the model genuinely doesn't know — a flat spread like `{0.34, 0.33, 0.33}` is a coin flip. Forcing an answer there risks emergency responses. So we set a **confidence threshold of 0.50**: if the top confidence is *below 0.50*, the system does **not** guess. It returns **REVIEW** and the message goes to a **human operator**. In the dashboard that's the purple "HUMAN TRIAGE REQUIRED" card.

**The system raises its hand instead of bluffing.**

---

## 8. What the combined system achieves (and the honest verdict)

| Quantity | Value |
| :--- | :---: |
| Messages answered automatically (coverage @ 0.50) | **67.4%** |
| Macro-F1 on the messages it *does* answer | 0.4197 |
| **SEVERE messages handled (auto + human routing)** | **0.85** |

- Coverage 67.4% = ~2 of 3 messages answered instantly; the other 1 of 3 go to a human. That's a realistic triage-aid posture.
- SEVERE handling **0.85** means that between the keyword floor, the model, and human review, 85% of genuinely dangerous messages end up treated as dangerous — well above the 0.70 gate.

**Verdict: CONDITIONAL PASS.**

Why not a clean PASS? The *raw statistical model* alone misses the quality gates (auto-accuracy 0.458 < 0.55; auto macro-F1 0.4197 < 0.45). We could have hidden that — but that's the dishonest move the Evaluation Engineer exists to prevent. CONDITIONAL PASS is the truthful statement: *"usable and safe because of the guard rail + abstention; the core model alone is weak; label noise from rule-derived labels is a disclosed limitation."*

---

## 9. What actually happens when you paste a message (the live flow)

1. You paste text → `clean_text()` + `tokenize()`.
2. Keyword **guard rail** runs first (SEVERE floor → MODERATE floor).
3. Statistical core (TF-IDF + Logistic Regression) computes probabilities + top confidence (the "Classical" radio; "Deep" runs the BiLSTM instead for a live comparison).
4. Decision logic:
   - Guard said SEVERE → **SEVERE** (the rule wins, always).
   - Guard said MODERATE → MODERATE (or SEVERE if the model already said so).
   - Else if confidence < 0.50 → **REVIEW** (human).
   - Else → the model's class.
5. The dashboard renders the coloured card + directive ("CRITICAL TRIAGE — Escalate to immediate rescue dispatch").
6. The **entity extractor** reads the same message and pulls the "what + where + how many": people counts, location phrases, and detail flags — shown as a **Detected Entities** panel of chips under the verdict.

Try it: paste *"people are trapped on the roof…"* → the statistical core is only ~41% confident, but the guard hits `trapped` + `please help` → **SEVERE**. Paste *"markets open, buses running"* → low confidence and no guard hits → **REVIEW**. That's the whole design showing off.

---

## 10. Entity clarification — the "what, where, how many" layer

Severity class and confidence each answer one question: **how urgent?** But picture the coordinator: they see "SEVERE" and instantly need follow-ups — *how many people? where? who is hurt? do they need food?* That's what the **entity extractor** (`extract_entities`) answers, automatically, from the same message.

It is a completely different technique from the two models above. The models are *learned* (data → math → decision). The entity extractor is **pure rules**: regular expressions + a dictionary of keywords. Think of it as a very smart find-and-replace that spots patterns like "a number followed by a person-word" and colour-codes the results.

Three things it pulls (the dashboard renders them as chips):

| We extract | Example | The rule |
| :--- | :--- | :--- |
| 👥 People counts | `150 people`, `3 children`, `30 families`, `hundreds of villagers` | "number + person/group word", or "dozens/hundreds of + word" |
| 📍 Location | `roof of the colony`, `Bandra station`, `eastern ward` | "in / at / near / on + a place phrase", plus a dictionary of place-words (ward, colony, road, station…) and known Mumbai areas |
| 🔖 Details | `rescue needed`, `medical help`, `food & water`, `no electricity` | seven keyword groups → readable chips |

Why build this instead of using a "real" NER library (e.g. spaCy)?

- **No new dependency.** We're CPU-only; spaCy would add hundreds of MB of install for a demo system.
- **Explainable.** A rule either fires or it doesn't — you can verify every chip by hand. A learned NER would be yet another black box to defend at the viva.
- **Deterministic and testable.** Give it the same 20 messages and you get the same 20 answers, every run.

The honest trade-off: a rule-based extractor only understands the phrasings its patterns were written for. *"30 souls are stuck upstairs"* will not count "30 people", because the message uses no known unit word; a learned model might catch it. We accept that limit and disclose it — better a transparent tool with known edges than a magical one we can't explain.

---

## 11. Where it lives in the repo

| File | What it is |
| :--- | :--- |
| `stage_03_nlp/dl_engineer/nlp_utils.py` | cleaning, tokenising, vocabulary |
| `stage_03_nlp/dl_engineer/nlp_trainer.py` | trains BOTH tracks, picks + saves the winner |
| `stage_03_nlp/models/severity_stat.joblib` | the shipped TF-IDF + Logistic Regression (1.4 MB) |
| `stage_03_nlp/models/bilstm_severity.pth` | the BiLSTM + attention (9 MB) |
| `stage_03_nlp/integration_engineer/nlp_triage.py` | guard rail + abstention + entity extractor + the dashboard's `triage()` |
| `Roles/Role7_NLP_Engineer.md` | the viva cheat-sheet for this stage |

---

## One-line summary for the viva

> *"We trained a simple interpretable model and a deep BiLSTM on 33k real disaster messages, let the evidence pick the simple one, then wrapped it in a keyword guard rail that can only escalate and an abstention threshold that routes uncertainty to humans — a CONDITIONAL PASS that's safe to use as a triage aid, honestly disclosed."*