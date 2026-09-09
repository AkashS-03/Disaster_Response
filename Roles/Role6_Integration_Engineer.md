# Role 6: Integration Engineer — What I Actually Do (explained in depth)

> This role can feel mysterious — other roles build models, I build the *glue*. Read this from the top: it explains, in plain English, what integration means, how Streamlit works, how the models get loaded, how every tab talks to a model, and why this role wins credibility at the viva.

---

## 1. What does an Integration Engineer even do?

The other five roles produce **artifacts**: a `.joblib` file (trained ML model), two `.pth` files (trained neural networks), scalers, and CSVs. A pile of files on disk is not a product — nobody can *use* it.

My job: **turn that pile of files into one working, clickable system** — a web app where a non-technical emergency coordinator can:
1. type in sensor readings and instantly see a risk level,
2. upload a drone photo and instantly see FLOODED / CLEAR,
3. paste an SOS message and instantly get a triage verdict,
4. see one combined "overall incident posture" screen.

That web app is `master_dashboard.py`. If the models are the *engine*, I build the *dashboard*.

**Plain English:** a model is like a calculator that only works from the command line.
I'm the person who puts all the calculators into a phone app with buttons and a screen.

---

## 2. The tool: what is Streamlit?

Streamlit is a Python web framework. It sounds magical but the model is simple:

- You write a normal **Python script** (top to bottom).
- Every time the page loads **OR any widget changes** (slider moved, button clicked, text pasted), Streamlit **re-runs the whole script from the top**.
- During the run, every `st.xxx(...)` call *prints* a UI element into the page: `st.slider` draws a slider, `st.button` a button, `st.markdown` text, `st.image` an image.

So the dashboard is literally a script that runs, draws everything, and re-runs on every interaction. That's why it's fast to build and why Python objects (loaded models) are directly usable in the UI.

### The two things that make the rerun-model workable
- **`st.session_state`** — a dictionary that *survives between reruns*. Widget values, loaded images, and preset choices live here. Without it, your data would reset every time you click anything.
- **`@st.cache_resource`** — a decorator that says *"compute this once, store it, and give me the same object back on every rerun"*. We wrap model loading in it so the 86 MB Random Forest is **not** re-loaded from disk on every slider move.

---

## 3. The model inventory — everything I wire together

| Model | File | Size | What it is |
| :--- | :--- | :---: | :--- |
| Risk Classifier | `stage_01_ml/models/risk_model.joblib` | 86.25 MB | GuardRailedPredictor (Random Forest) → LOW/MODERATE/SEVERE |
| Flood Vision | `stage_02_dl/models/vision_classifier.pth` | 8.73 MB | MobileNetV2 CNN → FLOODED/CLEAR |
| River Forecast | `stage_02_dl/models/lstm_forecaster.pth` | 0.21 MB | Residual FloodLSTM → 12h river peak |
| Time-series scaler | `stage_02_dl/data/time_series/ts_scaler.joblib` | — | StandardScaler (normalises features before the LSTM) |
| NLP Triage (winner) | `stage_03_nlp/models/severity_stat.joblib` | 1.4 MB | TF-IDF + LogisticRegression → LOW/MODERATE/SEVERE |
| NLP deep (compare) | `stage_03_nlp/models/bilstm_severity.pth` | 9.06 MB | BiLSTM + attention (dashboard's "Deep" radio option) |
| XGBoost baseline | `stage_02_dl/models/xgb_baseline.joblib` | 0.25 MB | forecast reference model (not called in the UI) |

### The two file formats (know these cold)
- **`.joblib`** = a *serialised Python object* saved via `joblib.dump`. Loading it back with `joblib.load` gives you a ready-to-call object. The Stage 01 file contains the whole pipeline **plus** the `GuardRailedPredictor` wrapper — so its `.predict()` *already* enforces the safety rules. To load it, Python must be able to import the wrapper class; that's why the dashboard inserts `stage_01_ml` into `sys.path` and does `import safety_guard` at the top.
- **`.pth`** = a PyTorch **state_dict** (just the learned weights, not the code). We have to rebuild the *architecture* in code first (the same classes the trainers used), then `load_state_dict(...)` pours the weights in, then call `.eval()`. That's why the dashboard redefines `FloodLSTM` and rebuilds `models.mobilenet_v2()` before loading weights.

---

## 4. How the models get loaded (the heart of my role)

All of it lives in one function, `load_all_models()`, marked `@st.cache_resource`:

```python
MODELS = load_all_models()
```

Steps inside, for each of the 4 model families (ML → LSTM → Vision → NLP), the pattern is identical and defensive:

1. **Check the file exists** (`os.path.exists(...)`). If not trained → give `models_dict[key] = None`.
2. **Try to load it** inside `try/except`.
3. **On any failure** → `None` + a printed error, **don't crash the app**.

Why `None` instead of crashing? So the dashboard **degrades gracefully** — if the Integration Engineer loads the app before a stage finished training, the other tabs still work and the missing one shows a friendly warning instead of a stack trace. That's a deliberate, demoable engineering decision.

The keys it produces (memorise this dict):
| Key | What it holds |
| :--- | :--- |
| `MODELS['ml']` | the risk pipeline with the guard rail baked in |
| `MODELS['lstm']` + `MODELS['scaler']` | FloodLSTM + the StandardScaler it needs |
| `MODELS['vision']` + `MODELS['v_transform']` | MobileNetV2 CNN + the image-normalisation pipeline |
| `MODELS['nlp']` | a `NlpTriage` object from `build_triage(...)` (guard rail + abstention + entity extractor built in) |

Two integration details worth flagging:
- **Path setup:** the dashboard computes `BASE_DIR`, `STAGE_01_DIR`, `STAGE_02_DIR`, `STAGE_03_DIR` from its own `__file__`, and inserts Stage 01 + Stage 03 into `sys.path` **before** loading anything — so `import safety_guard` (needed to unpickle the guard-railed model) and `from integration_engineer.nlp_triage import build_triage` (needed for the NLP wrapper) both resolve.
- **Device:** `DEVICE = "cuda" if torch.cuda.is_available() else "cpu"` — everything runs on the same device the models were trained on.

---

## 5. The dashboard anatomy (tab by tab)

The app is one page with a hero header and **5 tabs**, split by `st.columns` into input/panel columns, with `st.markdown` HTML classes for the mission-control styling.

### Tab 1 — Stage 01 ML Risk Classifier
- **Inputs:** zone dropdown, river/rainfall/calls sliders, road/bridge number inputs, plus an "Advanced Telemetry" expander (72h rain, 24h calls, 72h river average, trend) and 3 quick presets (🟢 Normal Day / 🟡 Heavy Rain / 🔴 Flash Flood) that pre-fill `st.session_state["s1_*"]`.
- **Flow:** the widgets' values are assembled into a **pandas DataFrame** with exactly the 12 feature columns the trained pipeline expects → `MODELS['ml'].predict(df)` (guard safety included) + `predict_proba` for confidence.
- **Output:** colour-coded result card (`res-severe/moderate/low`), guard-rail banner if a deterministic rule fired, and a directive card.
- **Graceful path:** if `MODELS['ml'] is None`, it shows a "model not trained" warning instead of crashing.

### Tab 2 — Stage 02 Flood Vision
- **Inputs:** two buttons load benchmark images (flooded/clear), or the user uploads a drone photo via `st.file_uploader`. Either way the image is stored in `st.session_state["selected_img"]` (so it survives reruns) and displayed.
- **Flow:** the image is run through `MODELS['v_transform']` — resize to 224×224, convert to tensor, normalise with ImageNet mean/std (exactly what training used) — then one forward pass `MODELS['vision'](img)`, `argmax` → FLOODED/CLEAR, `softmax` for confidence.
- **Output:** verdict card + confidence bar.

### Tab 3 — Stage 02 River Forecast
- **Inputs:** current river gauge, rainfall, call-rate sliders + a "Basin Dynamic Trajectory" `select_slider` (Receding / Stable / Rapid / Flash Surge).
- **Flow:** builds a synthetic 48-step input window, **scales it with `MODELS['scaler']`**, pushes it through the LSTM, and converts the scaled prediction back into **metres** (using the scaler's stored scale on the river feature). That numeric "bump" is then blended with the human-set trajectory mode into `net_delta_12h`, and `peak_12h = current + delta`.
- **Output:** 3 live telemetry tiles (Now / Peak +12h / Expected Rise), a 5-point hydrograph line chart, and an alert banner if `peak_12h ≥ 5.0m`.

### Tab 4 — Stage 03 NLP Message Triage
- **Inputs:** 3 preset buttons (🚨 Severe / 🌊 Moderate / ☀️ Low) that write `st.session_state["sos_input"]`, a `st.text_area` bound to that same key, and a radio to pick **Classical (interpretable)** vs **Deep (BiLSTM + attention)** — a live comparison of the two tracks.
- **Flow:** `MODELS['nlp'].triage(msg, use_deep=...)` → a dict with `prediction` (LOW/MODERATE/SEVERE/**REVIEW**), `confidence`, `guard_hits`, `reason`. The guard rail runs inside the wrapper, and confidence < 0.50 produces REVIEW.
- **Output:** verdict card + confidence bar; REVIEW renders purple (`res-review`, `color-review`) with a "🧑💼 HUMAN TRIAGE REQUIRED" directive (`dir-human`). Below the verdict, a **Detected Entities** panel renders chips from the NLP agent's `extract_entities()` — people counts (amber), locations (green), and detail flags (purple) — so the coordinator sees *how many + where + what's needed* without re-reading the message. The panel intentionally relies on the same wrapper object the backend uses; if nothing is found it renders nothing, never a crash.

### Tab 5 — Unified Command Center
- **Fuses** the live outputs of the other tabs into one decision:
  - `ml_pred`, `vis_pred_label`, `peak_12h` → **TIER 3 / TIER 2 / TIER 1** with an animated status ring, a % figure, and a plain-language directive ("Execute full municipal evacuation protocol…").
  - A **readiness table** listing every sub-system (Stage 01 ML, Stage 02 Vision, Stage 02 Time-Series, Stage 03 NLP) with its current signal and green "Operational" status.
- This is the "so what" screen — everything the coordinator needs in one glance.

---

## 6. session_state — the quiet hero (and a real lesson)

`st.session_state` is a **dict that survives reruns**. In this app it holds:
- `s1_*` — Stage 01 preset values (so clicking "⛅ Moderate Scenario" fills the sliders),
- `selected_img` / `img_name` — the current drone image (uploads survive tab switches),
- `sos_input` — the NLP message text (so presets fill the text area).

**The lesson we actually hit:** NLP presets originally wrote to a key named `sos_text` while the text area read `sos_input` — so clicking a preset *silently did nothing*, the box looked broken. Fix: the button must write to the **same key the widget uses**. That's the kind of integration bug only my role discovers, because it's not about the model — it's about state and wiring.

---

## 7. The design language (why it looks like mission control)

All custom looks come from one big CSS block via `st.markdown`:
- A dark gradient app background + animated grid overlay;
- `.panel` / `.panel-title` cards with cyan badges;
- `.result-card` classes (`res-severe/moderate/low/review`) with colour-coded borders and glows — red = critical, amber = warning, green = safe, purple = human review;
- `.directive` classes (`dir-critical/warning/safe/human`) for the action message;
- `.tile` telemetry cards and the status-ring SVG;
- A `.streamlit/config.toml` with `base="dark"` so *native* widgets (text areas, number inputs, radios) match the dark theme instead of rendering as white boxes.

Why this matters: at the viva, the panel sees a *system*, not a notebook. Colour coding makes safety legible: a glance tells you urgent vs nominal.

---

## 8. How it's run and how I prove it works

- Run: `streamlit run master_dashboard.py` (from the repo root) → serves on `http://localhost:8501`.
- I added the dark theme via `config.toml` under `.streamlit/`.
- **Health check:** `http://localhost:8501/_stcore/health` returns 200 when the server is up. I also verify the full app (all models + all tabs) via Streamlit's `AppTest` harness headlessly — no browser needed — clicking presets, driving sliders, pasting SOS text, and asserting the verdict cards render with zero exceptions.

---

## 9. Why Integration wins the viva

- A panel **trusts a working demo** more than any printed metric. "Models exist" is weak; "here's me operating the system live" wins.
- It **proves the stages work together**: the fused TIER ring in Tab 5 only works if the ML, CNN, LSTM, and NLP outputs are all correct and flowing.
- It makes **safety visible**: the guard rail banner, the purple REVIEW state, and the readiness table all *demonstrate* the safety story rather than just claiming it.

---

## 10. Where everything lives (file map)

| Path | Purpose |
| :--- | :--- |
| `master_dashboard.py` | the entire web app (layout, CSS, model loading, all 5 tabs) |
| `.streamlit/config.toml` | dark theme so native widgets match the dashboard |
| `Roles/Role7_NLP_Engineer.md` | NLP role doc (the Stage 03 agent I integrate) |
| `Explainations/NLP_Explained.md` | beginner walkthrough of the NLP agent |

---

## 11. Likely Viva Questions (with deep answers)

1. **How do the stages connect?** — Every stage writes a trained artifact to a known path. My dashboard maps those paths, loads each artifact at startup into a `MODELS` dict, and calls them live in the UI. Stage 01 and Stage 03 paths are added to `sys.path` so their wrapper classes (`safety_guard`, `nlp_triage`) can be imported for loading.
2. **What if a model isn't trained yet?** — Each load block checks the file exists and is wrapped in `try/except`; if it fails, the dict entry is `None` and the tab shows a clear warning. The app never hard-crashes from a missing model.
3. **Why Streamlit?** — It's Python-native (models are used directly, no server/API boilerplate), re-runs the script on interaction, and `st.session_state` + `@st.cache_resource` make state and heavy-loaded models easy. Fast to build, perfect for a live demo.
4. **How is safety enforced in the UI?** — Same wrappers as training: the Stage 01 `GuardRailedPredictor` is loaded as-is (its `predict` enforces the thresholds), and the NLP `build_triage` wrapper runs keyword floors first and sends low-confidence messages to REVIEW. The UI just displays what the safe wrappers decide.
5. **How does the NLP tab work?** — It loads the same `build_triage` object the backend uses, offers a classical-vs-deep radio for a live comparison, writes preset SOS examples into `session_state["sos_input"]` (which also seeds the text area), and renders REVIEW as a distinct purple human-triage state.
6. **Why is the app one script re-run every interaction?** — That's Streamlit's execution model. It's what makes building and demoing fast; caching prevents redundant model loads; session_state preserves user data between reruns.
7. **How do you know it actually works?** — `AppTest` harness drives the app headlessly (clicks presets, sets widget values, runs inference) and asserts verdict cards render with zero exceptions, plus the `/health` endpoint check. The same harness caught the `sos_text`/`sos_input` wiring bug.
8. **What was the hardest integration bug?** — The NLP preset buttons wrote to a session key that didn't match the text area's key, so presets silently did nothing. Fixing it taught the rule: session keys and widget keys must match — a wiring bug, not a model bug.

## SLM: Detailed Explanation & My Role Facts (Role 8 tie-in)

### What the SLM is (integration view)
It is a **language model** (next-word predictor) wrapped in a tiny Python class (`SlmAssistant`) that the dashboard treats like any other optional model. It is **assistive only**: whatever it suggests, the triage verdict, the guard rail, and the human REVIEW state are decided by the shipped pipeline above it.

### How it is loaded (defensive integration)
Inside `load_all_models()`:
- I import `SlmAssistant` from `stage_04_slm`'s `integration_engineer.slm_integration`, register it as **`MODELS['slm']`** — but only if its weights exist (`slm_lstm.pth` + `slm_lm_meta.json` in `stage_04_slm/models`). If missing, `MODELS['slm'] = None`.
- The whole block is wrapped in `try/except`, matching every other model loader: **the app never crashes because the SLM is absent**.
- Weight loading is **lazy**: constructing the wrapper does nothing heavy; the `torch` weights load on the first `.perplexity()` / `.complete()` call. Tab 4 stays snappy and models load once via `@st.cache_resource`.

### The Copilot panel (what I render, all honest widgets)
| Widget | What the user sees | Honesty rule |
| :--- | :--- | :--- |
| **Cleaned draft** | `slm.refine(msg)` — collapses whitespace, ensures terminal punctuation | *deterministic hygiene* — zero generated content |
| **Next-word guesses** | `slm.complete(msg, k=5)` — top-5 words with % (softmax), rendered as chips | caption reads *"SLM guesses for the most probable next word … NOT extracted facts or generated data"* |
| **DomainFit (perplexity)** | `slm.perplexity(msg)` → band text (<300 very typical / 300–900 typical / >900 unusual → keep human in loop) | caption reads *"Gauge only — it never overrides the triage verdict above"* |
| **Missing model** | friendly info: *"run `slm_train.py` once and it appears here"* | no crash, no fake fallback |

### Why it is safe to demo live
- The panel lives **below** the triage card in Tab 4 — it visually and logically cannot change the verdict.
- Performance: one tiny LSTM forward pass on CPU is a few tens of milliseconds; `st.code`, `st.caption`, `st.markdown` chips render instantly.
- **AppTest harness verified headlessly:** I typed an SOS message, clicked the 🚨 Severe preset, switched classical/deep engines → **zero exceptions**, the Copilot code block and its captions rendered. That is the same harness that caught the original `sos_text`/`sos_input` wiring bug, so the Copilot was checked the same strict way.

### Likely SLM questions for the Integration Engineer
1. **"What happens on first run before the SLM is trained?"** — `MODELS['slm']` is `None`; the Copilot shows the "run slm_train.py once" info and nothing else breaks. Verified by the loader structure and AppTest.
2. **"Can the SLM override a SEVERE verdict?"** — No. The verdict/conf guard rail/REVIEW are computed by `MODELS['nlp']`; the SLM panel is a separate widget set below it.
3. **"Is the SLM loaded on every rerun?"** — No. `@st.cache_resource` caches `MODELS`, and the SLM's `torch` load is lazy on top of that — one load for the whole session, then fast inference.