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

## SLM: Detailed Explanation & My Role Facts (Role 8 tie-in)

### What the SLM is (ML view)
The SLM is a **Severity-Conditioned Sequence-to-Sequence Summarizer**. From an ML Engineer's perspective, the SLM represents **task-aligned model selection, objective evaluation gating, and edge parameter efficiency**.

### Model Selection & Baseline Gating
My project rule: **a model architecture ships only if it clears quantitative gating criteria on fidelity, latency, and operational constraints.**
- **Why not use a Cloud LLM API?** — Fails the offline gating requirement. Disasters destroy connectivity. A model requiring cloud servers scores 0% in disaster resilience.
- **Why an Encoder-Decoder Transformer with PEFT/LoRA?** —
  - Multi-head cross-attention directly maps diffuse casualties and geographic markers to structured directives.
  - Parameter-Efficient Fine-Tuning (LoRA $r=8, \alpha=16$) freezes base representations, preventing catastrophic forgetting of grammar while fine-tuning only ~118,000 parameters.
  - CPU inference latency benchmarks at **85.7 ms** (well below our sub-300ms SLA).
- **Gating Scorecard for Shipping Stage 04:**
  - ROUGE-1 F1: **0.4701** (Target: >0.4500) $\rightarrow$ **PASS**
  - ROUGE-2 F1: **0.2675** (Target: >0.2500) $\rightarrow$ **PASS**
  - ROUGE-L F1: **0.4453** (Target: >0.4000) $\rightarrow$ **PASS**
  - Length Compliance: **100.0%** (Target: >95.0%) $\rightarrow$ **PASS**
  - CPU Latency: **85.7 ms** (Target: <300 ms) $\rightarrow$ **PASS**
  - Reading Time Reduction: **84.7%** (Target: >80.0%) $\rightarrow$ **PASS**

### Likely SLM questions for the ML Engineer
1. **"Why use PEFT/LoRA instead of full fine-tuning?"** — Full fine-tuning of multi-million parameter transformers on edge CPUs causes gradient instability, high memory consumption, and potential overfitting on domain sets. LoRA trains low-rank adapter matrices $B \cdot A$ with rank $r=8$, reducing trainable parameters by ~98% while achieving a validation loss of 0.2584.
2. **"How do you evaluate summarization quality objectively?"** — We use standard n-gram overlap metrics (ROUGE-1, ROUGE-2, ROUGE-L, BLEU-2) against held-out ground-truth tactical summaries, combined with deterministic audits for sentence count compliance and key factor extraction accuracy.
3. **"What is the operational trade-off between model size and accuracy?"** — Massive 70B models offer marginal improvements in conversational prose, but introduce 1.5s+ latency and fail in blackouts. Our 11.6 MB model trades open-domain general knowledge for sub-90ms deterministic, mission-critical execution.

