# Role 3: ML Engineer

## What I Own
I build the **Stage 01 risk classifier** — a Random Forest model wrapped in a **`GuardRailedPredictor`**. My model turns sensor inputs into a risk label: LOW / MODERATE / SEVERE.

## Model Architecture
- **Base model:** `RandomForestClassifier` (ensemble of decision trees).
- **Full pipeline:** `Pipeline(imputer → one-hot encoder for zones → StandardScaler → RandomForest)`.
- **Wrapper:** `GuardRailedPredictor(pipeline)` — a deterministic safety layer ON TOP of the ML output.

### Architectural Specs (key numbers)
| Parameter | Value |
| :--- | :---: |
| Algorithm | RandomForestClassifier |
| **Number of trees (n_estimators)** | **100** |
| Max depth | default (None → fully grown) |
| **class_weight** | **'balanced'** (counteracts minority class bias) |
| random_state | 42 |
| Numeric features | median-imputed + StandardScaler |
| Categorical features | 1 (zone_id) → most_frequent impute + OneHotEncoder (handle_unknown='ignore') |
| **Model file size** | **86.25 MB** (`risk_model.joblib`) |
| Augmented training rows | 35,136 real + ~4,560 boundary = ~39,696 |
| Decision rule thresholds | SEVERE: river≥4.5 / (rain≥150 & river≥3.5); MODERATE: river≥3.0 / rain≥80 / calls≥100 |

### Why Random Forest
Robust to noise/outliers, handles mixed data, gives feature importances, no heavy tuning, and is interpretable — appropriate for safety-critical use. `class_weight='balanced'` plus boundary augmentation addresses the minority-class problem identified in evaluation.

## Why a Guard Rail? (The "safety-critical ML" story)
A pure ML model is statistical — it might miss. In disaster response, **missing a flood is worse than over-warning**. So I wrap the model so that:
- If a row should be **SEVERE** by the expert rules (river ≥ 4.5 OR rain ≥ 150 with river ≥ 3.5) → the guard rail **forces** SEVERE, whatever the model says.
- If it should be **MODERATE** → forces MODERATE (unless the model says SEVERE, which is kept).
- LOW predictions are left to the model.

This is called a **deterministic guard rail** on a **probabilistic model**. We get the best of both: the ML generalises within regions, but the safety thresholds are **guaranteed**.

## Implementation Detail
`GuardRailedPredictor` is a small class in `stage_01_ml/safety_guard.py`. It delegates all sklearn methods (`predict_proba`, `classes_`, `feature_names_in_`) to the wrapped pipeline via `__getattr__`, and overrides `predict()` with the safety logic. It is **pickleable** so joblib can save/load it alongside the pipeline.

## Key Numbers (Before → After)
| Metric | Plain RandomForest | GuardRailedPredictor |
| :--- | :--- | :--- |
| Global Accuracy | 0.8408 | **0.8422** |
| Edge-case accuracy | 9/17 (52.94%) | **17/17 (100%)** |
| Safety guarantee | none (could under-warn) | **guaranteed** |

## The MODERATE recall story (important for the debate)
Our MODERATE recall looks terrible (≈1%). But analysis showed every MODERATE row in the temporal test set has `river_level = 20.0m` — which by the decision rules should be **SEVERE**. So the test labels themselves are inconsistent with the rules; **the low MODERATE recall is a label artifact, not a model failure**. We surface this honestly rather than hiding it.

## Likely Viva Questions
1. **Why Random Forest?** — Robust, handles mixed data, gives feature importances, doesn't need deep tuning; baseline interpretability is valuable in safety.
2. **Explain the guard rail.** — Deterministic rules on top of a probabilistic model; guarantees safety thresholds, ML handles within-region generalisation.
3. **Does the guard rail over-warn?** — Intentionally. In disaster response false alarms are cheaper than missed floods; and the rules come from the data-engineering expert decision thresholds.
4. **Why keep MODERATE recall low?** — It's a label/rule inconsistency in the test data (river=20m should be SEVERE), not a model deficiency. We disclose it.
