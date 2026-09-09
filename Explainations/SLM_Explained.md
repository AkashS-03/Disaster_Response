# Stage 04 — SLM (Small Language Model): Explained From The Ground Up

*For beginners. No jargon assumed — every idea is built up from a simple one.*

---

## 1. What an SLM is (and how it differs from a chatbot LLM)

A **language model** is a computer program that has one superpower: it can
**predict the next word** when you give it some words.

- Give it:  *"people are trapped on the"*
- It answers: *"roof"* (probably) or *"roof of a"* (also probably).

That sounds small — but it is the exact same trick used by huge systems like
ChatGPT (their models are also just "predict the next word", trained on far
more text with far more parameters). We call ours an **S**mall **L**anguage
Model because it is tiny (about **17 MB**), trains in **~8 minutes on a plain
CPU**, has **zero external dependencies**, and lives **on our own machine**.

### Why build our own instead of downloading a big one?
1. **Floods kill networks.** A rescue coordinator in the field may not have
   internet. Our SLM works fully offline.
2. **Our laptop is plain CPU.** Big models need GPU/fast hardware and heavy
   packages (`transformers` is ~250 MB of downloads). Ours needs only PyTorch,
   which we already have.
3. **Explainability for the debate.** If a reviewer asks "what exactly did you
   build?", we can open the code and show every layer. A downloaded black-box
   model would be a mystery even to us.
4. **Trust.** A small model that we trained ourselves on *our own data* is
   easier to judge honestly than a giant model we don't understand.

---

## 2. It only learns from REAL data (nothing synthetic)

Our biggest rule: **no synthetic (fake/computer-generated) text anywhere.**

- Training text = the **33,791 real emergency messages** (same master dataset
  the triage model uses — Figure Eight + Kaggle).
- The SLM never *invents* rescue reports to pad its training.
- Anything the SLM says in the app is labelled **"SLM guess"** — it is a
  *drafting hint*, never treated as real data, never fed back into training.

> Why so strict? In disaster response, a made-up message could send rescue
> teams to the wrong place. The whole point of our project is honest data. A
> "paraphrase" feature that rewrites real messages would *create* new text —
> that is exactly the kind of synthetic data we refuse to make.

---

## 3. The pipeline, piece by piece

### Step 1 — Tokenising words into IDs
Computers don't read words; they read numbers. So each message is:

1. **lowercased** — "Flood!" → "flood",
2. **Tokenised** by a regex `[a-z0-9']+` — "flood, water rising" →
   `["flood", "water", "rising"]`,
3. **Mapped to IDs** via a learned **vocabulary**.

### Step 2 — Building the vocabulary
We counted every word across all 33,791 messages and kept words that appear
**at least twice** (rare/junk words are skipped), up to **16,000 words**, plus
**4 special tokens**:

| Token | ID | Meaning |
| :--- | :---: | :--- |
| `<s>` | 0 | Start of message |
| `</s>` | 1 | End of message |
| `<pad>` | 2 | Filler to make sequences equal length |
| `<unk>` | 3 | Any word NOT in the vocabulary (unknown) |

Vocabulary size = **16,004**. A message is now just a list of small numbers.

### Step 3 — The window trick (how it learns)
To "predict the next word", we cut each real message into `(input, target)`
pairs. For *"people are trapped"*:

```
input   : <s> people are trapped        predict next word
target  : people are trapped </s>
```

Every **real** message becomes `<s> words… </s>`, windows of up to **24
tokens**, and at every position the model is asked: *"what word comes next?"*.
From 33,791 messages we get **48,244** of these training windows. This is pure
echo of real messages — we create no new sentences.

### Step 4 — Embedding (turning IDs into meanings)
An **embedding** is a table: each of the 16,004 words gets a list of **128
numbers** (a "meaning vector"). Words that behave similarly end up with
similar numbers. The model *learns these numbers during training* — it's not
something we hand-write.

### Step 5 — The LSTM (the memory machine)
**LSTM** = Long Short-Term Memory. It's a special kind of neural network that
reads a sentence **left to right** and keeps a *running memory* of what it has
seen so far. Because it remembers:

- after *"people are"* it has a good guess for the next word,
- after *"people are trapped on the"* it has an even better one.

We use **2 stacked LSTM layers**, each with **128 memory units** — 2 layers
lets the model catch slightly higher-level patterns than 1. During training we
also use **dropout (0.2)**, which randomly turns off some connections to stop
the model from just memorising the training set word-for-word.

### Step 6 — The output layer
The final word-vector is pushed through a linear layer that scores **every
word in the 16,004-word vocabulary**. Those scores go through **softmax**
(which turns any list of numbers into probabilities that add up to 100%).
The **highest-scoring word is the model's guess** for the next word.

---

## 4. Training & the numbers to quote

| Quantity | Value |
| :--- | :---: |
| Training data | 33,791 real messages → 48,244 windows |
| Vocabulary | 16,004 tokens (16,000 words + 4 special) |
| Embedding size | 128 |
| LSTM | 2 layers × hidden 128 |
| Max sequence length | 24 tokens |
| Optimizer / learning rate | Adam / 1e-3 |
| Epochs / batch size | 12 (3 + 9 continuation) / 512 |
| **Final training average loss** | **5.72** |

### The architecture, spec by spec (verify the numbers yourself)

| Component | Config | Parameter count |
| :--- | :--- | ---: |
| Embedding | `nn.Embedding(16004, 128)` — one learned 128-dim vector per token | 2,048,512 |
| LSTM layer 1 | reads 128-dim emb → 128-dim hidden (4 gates: in + hidden weights + biases) | 132,096 |
| LSTM layer 2 | stacks on layer 1 → 128-dim hidden | 132,096 |
| Output head | `nn.Linear(128, 16004)` — a score for every vocab token | 2,064,516 |
| **Total** | ≈ **4.38M parameters ≈ 17.5 MB** on disk (fp32) | **4,377,220** |

- **Forward pass:** for every window position the stack is `embed → LSTM → Linear`,
  and softmax over the 16,004 scores gives "probability of each next word".
- **What is trained:** only the LM weights above. `encode(x)` = the LSTM hidden
  state at the **last real (non-pad) token** — a frozen 128-dim vector that the
  classifier/head tracks re-use.
- **Track C head** (does NOT ship): `Linear 128→ReLU→Dropout(0.3)→Linear(3)` —
  16,899 trainable params on top of the frozen encoder; class-weighted CE,
  features standardised with train-only statistics.
- All config (`emb`/`hidden`/`layers`/`max_seq`/`epochs`) is saved in
  `stage_04_slm/models/slm_lm_meta.json`, so the architecture is reproducible
  from the artifact alone.

**How to read the loss:** average loss is "how surprised the model is".
- If it were guessing randomly, each of the 16,004 words would be equally
  likely, surprise ≈ `ln(16004) ≈ 9.68`.
- We start at **7.62** and end at **5.72** — so after 12 epochs on a
  laptop, the model is genuinely far less surprised (≈ **52× more confident**)
  than random. It has learned real word patterns like *"trapped on the roof"*.

---

## 5. Perplexity — the DomainFit honesty gauge

**Perplexity** is just the average loss converted back to "how many equally
likely choices does it feel like there are". If the model is very unsure,
perplexity is high; if it confidently predicts the actual next words,
perplexity is low.

We use perplexity as an **honesty signal** in the dashboard:

- Give the SLM a real SOS message and compute its perplexity.
- **Low perplexity** (e.g. < 300) → the message *reads like* the disaster
  messages the model learned from. The wording is **in-domain**.
- **High perplexity** (e.g. > 900) → the wording is unusual/out-of-domain, so
  we flag **"keep a human in the loop"** instead of trusting the automation.

Bands we show in the app (heuristic, disclosed as such):
| Perplexity | Meaning shown to the coordinator |
| :---: | :--- |
| < 300 | very typical of disaster messages (low) |
| 300 – 900 | typical for a disaster message |
| > 900 | unusual wording — keep the human in the loop (high) |

Calibration note: the bands are set from the real held-out test-message
distribution (independently audited) — average perplexity ≈ 414, median ≈ 279,
90th percentile ≈ 871 —
so "300–900" is roughly the familiar zone, and anything far above it is
genuinely strange text. This is a **gauge only** — it never changes the
triage verdict.

---

## 6. Next-word suggestions (top-5), not a chatbot

The Copilot panel asks the SLM: *"after these words, which word is most
likely?"* We take the softmax probabilities over the whole vocabulary, remove
the special tokens, and show the **top 5** with their probabilities:

```
type:  "People are trapped on the"
SLM:   roof · 6%, and · 3%, the · 3%, of · 3%, floor · 2%   ← "SLM guess"
```

- **Top-k** just means "show the k highest-probability guesses".
- This is a *drafting hint* for whoever is typing the report — it might
  autocomplete "roof" quickly — **not** a fact and **not** generated data.
- It is clearly labelled **"SLM guesses for the most probable next word"** in
  the app.

Feel free to mention *temperature* in the debate if asked: temperature is a
slider that flattens or sharpens the probability distribution. We keep it at
greedy top-5 (top scoring words), the simplest honest setting. We deliberately
did **not** build a sentence generator: generating new sentences = creating
synthetic text, which is against our rule.

---

## 7. Track C — can the SLM also be a classifier? (honest: NO)

Since the SLM learns useful word patterns, we tried to reuse it: freeze the
SLM, take its learned **encoding** of each message (the hidden state at the
last real word), and train a small classifier **head** on top to predict
LOW / MODERATE / SEVERE. We call this **Track C**.

Fairness details so nobody can accuse us of cheating:
- Head trained only on the **real train split** (27,032 messages).
- Features **standardised** (scaled) using train-set statistics only.
- **Class weights** because SEVERE is only 10.5% of data.
- Evaluated honestly on the real untouched **test split** (6,759 messages).
- Baseline gating exactly like the token tracks: it must **clearly beat the
  shipped winner (macro-F1 0.4242)** or it does **not** ship.

Result on real test data:

| Metric | SLM Track C | Stat (shipped) | Deep (BiLSTM) |
| :--- | :---: | :---: | :---: |
| Macro-F1 | 0.3523 | **0.4242** | 0.4012 |
| SEVERE recall | 0.6746 | 0.3211 | 0.5915 |

**Verdict: Track C does NOT ship.** Macro-F1 0.3523 is far below the 0.4242
gate. We report this openly (`stage_04_slm/reports/slm_head_report.md`).

### The juicy debate point
Track C's **SEVERE recall (0.6746)** beats *both* shipped models (stat 0.3211,
deep 0.5915) — and SEVERE is the single most important class in triage. But
its overall discrimination is too weak to clear the quality gate. Exactly like
Track B vs Track A, we hold the line: **a good number on one class earns
attention, not a production slot.** The honest story is: the SLM failed at
triage but still ships as an assistive language model.

### Why is Track C weak while Track B (BiLSTM) is decent?
Track B is a 9 MB BiLSTM **tuned to the classification task** with attention
and 8 epochs of supervised training. Our SLM spent 12 epochs learning
*next words*, and its encoding is a single hidden state — classification is a
side-benefit, not its talent. That is exactly why we tested it instead of
assuming it would work.

---

## 8. Where it lives in the app (the Copilot panel)

Tab 4, below the triage panel: **"SLM COPILOT — assistive drafting hints
(SLM guesses, never data)"**. For the message you type it shows:

| Element | What it is | Honesty rule |
| :--- | :--- | :--- |
| **Cleaned draft** | squashes spaces, capitalises/ends with a period — a *deterministic* tidy-up, no new words | never rewrites or generates content |
| **Next-word guesses** | top-5 most probable next words with % | labelled **"SLM guess"** |
| **DomainFit (perplexity)** | familiar vs unusual wording gauge | has its own text: "gauge only — never overrides the triage verdict" |

It **never** overrides: the triage verdict, the guard rail, or the human
REVIEW decision. It is a keyboard helper with a warning light, nothing more.

If the SLM weights are missing, the panel politely says *"run
`stage_04_slm/dl_engineer/slm_train.py` once and it appears here"* — the app
never crashes because of it.

---

## 9. Honest limitations (say these before anyone asks)

1. **It is a weak language model.** 12 epochs on 33K messages is still tiny. Its
   guesses are often just common words ("the", "and", "to").
2. **High perplexity overall** (~200–1000 on real messages). Great for
   *relative* comparisons, not an absolute quality boast.
3. **It cannot parable; it cannot answer questions.** It predicts the next
   word; it has no knowledge or memory beyond word patterns.
4. **English characters only** (the tokeniser `[a-z0-9']` drops anything else
   like Devanagari/Hindi text or emojis).
5. **Learns the quirks of social + news text** together because that's what
   the corpus mixes; a disciplined crisis-only corpus would change its flavour.
6. **Track C failed the gate** — we must not pretend otherwise.
7. **Perplexity bands are heuristic** cut-offs, disclosed, not magic.

---

## 10. Adding / tweaking a new candidate? (how to repro everything)

```
python stage_04_slm/dl_engineer/slm_train.py
```
- Reads `stage_03_nlp/data/processed/master_text_dataset.csv` (33,791 real
  messages), builds the vocab + windows, trains the LM (12 epochs), saves
  `stage_04_slm/models/slm_lstm.pth` + `stage_04_slm/models/slm_lm_meta.json`.
- If `stage_04_slm/models/slm_lstm.pth` already exists it **reuses** it and only
  re-runs Track C.
- Writes `stage_04_slm/reports/slm_head_report.md` with the honest verdict.

Run the wrapper sanity check:
```
python stage_04_slm/integration_engineer/slm_integration.py
```
prints perplexity + top-5 guesses + a cleaned draft on a sample message.

---

## 11. File map

| File | Purpose |
| :--- | :--- |
| `stage_04_slm/data_engineer/slm_utils.py` | tokeniser, vocab builder, window maker |
| `stage_04_slm/eda_engineer/eda_slm.py` | descriptive stats of the real corpus (vocab coverage, `<unk>` rate, windows) → `reports/slm_eda_report.md` |
| `stage_04_slm/dl_engineer/slm_model.py` | the 2-layer LSTM + encoding + loaders |
| `stage_04_slm/dl_engineer/slm_train.py` | train LM + Track C head, write honest report |
| `stage_04_slm/evaluation_engineer/eval_slm.py` | independent audit: re-derives Track C + perplexity + top-5 hit rate → `reports/slm_evaluation_report.md` |
| `stage_04_slm/integration_engineer/slm_integration.py` | the Copilot wrapper (refine / complete / perplexity) |
| `stage_04_slm/models/slm_lstm.pth` | trained LM weights (~17.5 MB) |
| `stage_04_slm/models/slm_lm_meta.json` | vocabulary + architecture config |
| `stage_04_slm/models/slm_head.pth` | Track C head weights (**does not ship**) |
| `stage_04_slm/reports/slm_head_report.md` | honest Track C verdict (real data only) |
| `master_dashboard.py` | SLM Copilot panel in Tab 4 + lazy load of the SLM |