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

The app is one page with a hero header and **6 tabs**, split by `st.columns` into input/panel columns, with `st.markdown` HTML classes for the mission-control styling.

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
- **Output:** verdict card + confidence bar; REVIEW renders purple (`res-review`, `color-review`) with a "🧑💼 HUMAN TRIAGE REQUIRED" directive (`dir-human`). Below the verdict, a **Detected Entities** panel renders chips from the NLP agent's `extract_entities()` — people counts (amber), locations (green), and detail flags (purple).

### Tab 5 — Stage 04: SLM Severity Summarizer (Dedicated Briefing HUD)
- **Inputs:** Field dispatch report text area seeded by 5 curated disaster scenario presets (Flash Flood MCI, Embankment Warning, Power Outage / Debris, Dam Overflow Alert, Routine Ward Log), plus an operational Severity Selector (`Auto-Detect`, `LOW (< 1 sent)`, `MODERATE (1 sent)`, `SEVERE (2 sents)`).
- **Flow:** Calls `MODELS['slm_briefing'].generate_briefing(report, severity=...)` which runs `extract_key_factors()`, performs autoregressive greedy decoding via `TransformerLoRA`, and guarantees sentence constraints via `enforce_severity_length()`.
- **Output:**
  1. **Key Factors Highlight Box**: 3 mission-critical metric cards (**Location**, **People / Impact**, **Risk Level**).
  2. **Severity-Adaptive Summary Card**: Color-coded card (Green/Orange/Crimson) displaying the brevity-constrained summary with word count and sentence count indicators.
  3. **1-Click Offline Voice Briefing Player**: Embedded Web Speech API player allowing instant audio playback with zero cloud TTS network calls.
  4. **Live Telemetry & PEFT Benchmarks**: Real-time CPU latency tile (~85ms), time savings metric (~84.7%), and architecture spec table.

### Tab 6 — Unified Command Center
- **Fuses** the live outputs of all sub-systems into one combined decision:
  - `ml_pred`, `vis_pred_label`, `peak_12h` → **TIER 3 / TIER 2 / TIER 1** with an animated status ring, a % figure, and a plain-language directive.
  - A **readiness table** listing every sub-system (Stage 01 ML, Stage 02 Vision, Stage 02 Time-Series, Stage 03 NLP, Stage 04 SLM) with its current signal and green "Operational" status.

---

## 6. session_state — the quiet hero (and a real lesson)

`st.session_state` is a **dict that survives reruns**. In this app it holds:
- `s1_*` — Stage 01 preset values (so clicking "⛅ Moderate Scenario" fills the sliders),
- `selected_img` / `img_name` — the current drone image (uploads survive tab switches),
- `sos_input` — the NLP message text (so presets fill the text area),
- `slm_input_report` — the Stage 04 incident report text (so SLM presets fill the dispatch input area).

---

## 7. The design language (why it looks like mission control)

All custom looks come from one big CSS block via `st.markdown`:
- A dark gradient app background + animated grid overlay;
- `.panel` / `.panel-title` cards with cyan badges;
- `.result-card` classes (`res-severe/moderate/low/review`) with colour-coded borders and glows;
- `.directive` classes (`dir-critical/warning/safe/human`) for the action message;
- `.tile` telemetry cards and the status-ring SVG;
- A `.streamlit/config.toml` with `base="dark"` so *native* widgets match the dark theme.

---

## 8. How it's run and how I prove it works

- Run: `streamlit run master_dashboard.py` (from the repo root) → serves on `http://localhost:8501`.
- Verified compilation: `python -m py_compile master_dashboard.py` returns exit code 0.
- Verified headless execution: All 6 tabs load cleanly with zero exceptions.

---

## 9. Why Integration wins the viva

- A panel **trusts a working demo** more than any printed metric. "Models exist" is weak; "here's me operating the system live" wins.
- It **proves the stages work together**: the fused TIER ring in Tab 6 only works if all previous stages are operational.
- It makes **safety and brevity visible**: the guard rail banner, the purple REVIEW state, the Key Factor cards, and the 1-click voice briefing all *demonstrate* the operational value live.

---

## 10. Where everything lives (file map)

| Path | Purpose |
| :--- | :--- |
| `master_dashboard.py` | the entire web app (layout, CSS, model loading, all 6 tabs) |
| `.streamlit/config.toml` | dark theme so native widgets match the dashboard |
| `Roles/Role8_SLM_Engineer.md` | SLM role doc (Stage 04 tactical briefing agent) |
| `Explainations/SLM_Explained.md` | beginner walkthrough & architectural specs of the SLM |

---

## 11. Likely Viva Questions (with deep answers)

1. **How do the stages connect?** — Every stage writes a trained artifact to a known path. My dashboard maps those paths, loads each artifact at startup into a `MODELS` dict, and calls them live in the UI.
2. **What if a model isn't trained yet?** — Each load block checks the file exists and is wrapped in `try/except`; if it fails, the dict entry is `None` and the tab shows a clear warning. The app never hard-crashes from a missing model.
3. **Why Streamlit?** — It's Python-native (models are used directly, no server/API boilerplate), re-runs the script on interaction, and `st.session_state` + `@st.cache_resource` make state and heavy-loaded models easy.
4. **How is the Stage 04 SLM integrated into Tab 5?** — I load the `TacticalBriefingAssistant` wrapper into `MODELS['slm_briefing']`. When the user clicks "Generate Briefing" or selects a preset, the wrapper extracts key factors, runs `TransformerLoRA` inference in 85.7 ms, and renders the result card along with an embedded Web Speech API audio player.
5. **How does the voice synthesis work without an internet connection?** — It injects a safe HTML/JavaScript payload that calls the client browser's built-in `window.speechSynthesis` API. This runs 100% offline on the client machine with zero cloud round-trips.

---

## SLM: Detailed Explanation & My Role Facts (Role 8 tie-in)

### What the SLM is (integration view)
It is a **Tactical Briefing Assistant** wrapped in a clean, defensive Python class (`TacticalBriefingAssistant` in `stage_04_slm/integration_engineer/slm_integration.py`). The dashboard loads it into **`MODELS['slm_briefing']`** and gives it a dedicated control HUD in **Tab 5 ("🎙️ Stage 04: SLM Severity Summarizer")**.

### How it is loaded (defensive integration)
Inside `load_all_models()`:
- I import `TacticalBriefingAssistant`, checking if `stage_04_slm/models/slm_briefing.pth` and `slm_briefing_meta.json` exist.
- If present, it initializes the `TransformerLoRA` model and vocabularies once under `@st.cache_resource`.
- If missing, `MODELS['slm_briefing'] = None`, and Tab 5 displays a friendly setup notice (`st.info`) guiding the user to run `slm_train.py` without breaking any other tab in the app.

### The Tab 5 HUD Widgets (what I render)
| Widget | Implementation | Operational Role |
| :--- | :--- | :--- |
| **Scenario Presets** | 5 disaster buttons (MCI, Embankment, Outage, Dam, Routine) | Seeds input report instantly for live viva demo |
| **Severity Selector** | Radio (`Auto-Detect`, `LOW`, `MODERATE`, `SEVERE`) | Controls briefing length constraint (<1 sent, 1 sent, 2 sents) |
| **Key Factors Box** | 3 HTML metric cards (`Location`, `Impact`, `Risk Level`) | Visual situation awareness before briefing |
| **Adaptive Summary Card** | Color-coded card (Green/Orange/Crimson) | Actionable brevity-enforced tactical text |
| **Voice Briefing Player** | Web Speech API audio synthesis | 1-click spoken dispatch directly into operator headset |
| **Telemetry HUD** | Latency tile (85.7ms), time savings (84.7%) | Live verification of edge engineering performance |

### Likely SLM questions for the Integration Engineer
1. **"Why separate SLM into Tab 5 instead of bundling it into Tab 4 NLP?"** — Tab 4 is dedicated to single-message citizen SOS triage (urgency classification and entity detection). Tab 5 is dedicated to multi-unit command briefing (summarizing multi-page field logs into spoken dispatches for the incident commander). Keeping them in separate tabs maintains clear architectural boundaries.
2. **"Does the Web Speech API require internet?"** — No. Modern operating systems (Windows, macOS, Linux/Chrome) bundle local text-to-speech voices natively. The browser calls these offline local voices with zero network requests.
3. **"How does the UI ensure fast interaction?"** — The model weights (~11.6 MB) are cached in RAM via `@st.cache_resource`. Inference takes only 85.7 ms, so the summary and audio player appear virtually instantaneously.