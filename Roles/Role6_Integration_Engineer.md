# Role 6: Integration Engineer

## What I Own
I wire all the models into one working, demonstrable system: the **master dashboard**. I make it so a non-technical user (or a viva panel) can click and see the whole disaster-response pipeline live.

## Key Deliverable: `master_dashboard.py`
A single **Streamlit** web dashboard that exposes all stages:

### What it loads (the full model inventory it wires together)
| Model | File | Size | Stage |
| :--- | :--- | :---: | :--- |
| Risk Classifier | `stage_01_ml/models/risk_model.joblib` | 86.25 MB | GuardRailedPredictor(RandomForest) |
| Flood Vision | `stage_02_dl/models/vision_classifier.pth` | 8.73 MB | MobileNetV2 CNN |
| River Forecast | `stage_02_dl/models/lstm_forecaster.pth` | 0.21 MB | Residual FloodLSTM |
| XGBoost Baseline | `stage_02_dl/models/xgb_baseline.joblib` | 0.25 MB | forecast reference |
| Time-series scaler | `stage_02_dl/data/time_series/ts_scaler.joblib` | — | StandardScaler |

### What the dashboard does
1. **Stage 01 Risk Classification** — user enters sensor values (river level, rainfall, calls, closures, zone); the `GuardRailedPredictor` returns LOW / MODERATE / SEVERE.
2. **Stage 02 Flood Vision** — user uploads a drone image; the MobileNetV2 CNN says FLOODED / CLEAR.
3. **Stage 02 River Forecast** — shows the 12h-ahead LSTM forecast vs current level.
4. **Results cards** — colour-coded (critical = red, warning = amber, safe = green) for instant readability.

### The guard rail in action
The dashboard uses the **`GuardRailedPredictor`**, so even if a user enters a borderline input, the safety thresholds are enforced — it can never under-warn below the expert rules. This is the "safety-critical" story exposed to the end user.

## Why Integration Matters for the Debate
A panel trusts a *system* more than a *single metric*. My dashboard:
- Proves all the stage models actually **work together** (not just on paper).
- Shows a **real UI** — instant credibility in a live demo.
- Makes the safety guard rail visible and explainable.

## Likely Viva Questions
1. **How do the stages connect?** — All stages read/write shared CSVs and checkpoints; the dashboard loads the trained artifacts (`.joblib` + `.pth` + scaler) and calls them live.
2. **What if the model isn't trained?** — Clear error message via a checkpoint-existence check before running.
3. **Why Streamlit?** — Fast, python-native, produces a clean interactive UI with minimal boilerplate — perfect for a demo.
4. **How is safety enforced in the UI?** — The `GuardRailedPredictor` wrapper is used at inference, so deterministic thresholds hold wherever the UI reads them.
