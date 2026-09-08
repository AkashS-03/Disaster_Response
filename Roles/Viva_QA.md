# Disaster Response Project — Viva Q&A Prep

A plain-English cheat sheet covering every model, term, and number you'll need. Read this the night before — you'll be ready for the panel.

---

## 1. The Big Picture (why does this project exist?)

**Q: What is your project?**
A: An autonomous multi-agent disaster-response coordination system. It uses machine learning to assess flood risk in real time from sensor data, deep learning (CNN) to detect flooding from drone images, deep learning (LSTM) to forecast river levels 12 hours ahead, and NLP to triage emergency text messages by severity. Everything is surfaced in one dashboard so emergency coordinators can act fast.

**Q: What is a "multi-agent" system?**
A: Multiple specialised AI "agents" (roles) work together like a team — Data Engineer prepares data, ML Engineer classifies risk, DL Engineer detects floods and forecasts, NLP Engineer triages emergency messages, Evaluation Engineer independently audits, Integration Engineer wires it into a dashboard. Each has a clear job; together they form the full pipeline.

---

## 2. Stage 01 — Machine Learning (Risk Classification)

**Q: What does Stage 01 do?**
A: It takes hourly sensor readings — river level, rainfall, emergency call volume, road/bridge closures, zone — and classifies risk as **LOW / MODERATE / SEVERE**, so coordinators know where to send resources.

**Q: What model?**
A: A **Random Forest Classifier** wrapped in a **GuardRailedPredictor**.

**Q: What is a Random Forest?**
A: An **ensemble** of many decision trees. Each tree casts a vote on the risk level; the majority wins. It's robust, handles mixed data types, and is interpretable — great for safety-critical use.

**Q: What is the GuardRailedPredictor? (YOUR key differentiator)**
A: A **deterministic safety layer on top of the probabilistic ML model**. The expert decision rules are enforced as a hard guarantee:
- If `river_level ≥ 4.5` OR (`rainfall_72h ≥ 150` AND `river_level ≥ 3.5`) → **forced SEVERE**.
- If `river_level ≥ 3.0` OR `rainfall_72h ≥ 80` OR `calls ≥ 100` → forced **MODERATE** (or SEVERE if the model said so).
- LOW is left to the model.
The ML generalises within regions; the guard rail guarantees the safety thresholds are never violated. **Plain English: the model can never "miss" a flood below the expert threshold.**

**Q: What did the guard rail actually change?**
A: Edge-case accuracy went from **52.94% (9/17) → 100% (17/17)**, and overall accuracy from 0.8408 → **0.8422**. The guarantee is the real win, not the tiny accuracy bump.

**Q: What are edge cases?**
A: Test inputs sitting **exactly at the decision boundaries** — e.g. river level 2.99m (LOW edge) vs 3.01m (MODERATE), or rainfall 79 vs 81. A small error here is the most dangerous failure mode. We test just above and just below every threshold.

**Q: Why is the MODERATE recall only ~1%?**
A: This is a **data-label artifact, not a model failure**. Every MODERATE row in the temporal test set has `river_level = 20.0m` — which the decision rules classify as **SEVERE**. So the test labels themselves contradict the rules. We disclose this honestly rather than hiding it — dishonest metrics get caught.

**Q: How did you avoid data leakage?**
A: **Temporal split** — train on data before a cutoff date, test on data after. The model never sees the future. We also fit the scaler on training data only (no future statistics leaking in).

**Q: What is your accuracy?**
A: **0.8422** overall. But accuracy isn't the whole story — per-class metrics and edge-case behaviour matter more for safety.

**Q: What are the most important features?**
A: `river_level` (0.218), `river_level_rolling_72h_avg` (0.193), `rainfall_rolling_72h_sum` (0.097), `river_level_trend` (0.097), `emergency_call_volume` (0.093). Makes sense — how high the river is, its trend, recent rain and calls drive risk.

---

## 3. Stage 02 — Deep Learning

### Part A: Flood Vision (CNN)

**Q: What does the CNN do?**
A: Detects whether a **drone aerial image** shows flooding. Output: **FLOODED / CLEAR**.

**Q: What architecture, and why MobileNetV2?**
A: **MobileNetV2**, a lightweight and efficient CNN, fine-tuned on our drone data. Chosen because it's **fast and small** — suitable for deployment on edge devices/drones, not just a big GPU.

**Q: What is transfer learning?**
A: We start with a network already pre-trained on ImageNet (millions of general images), then **fine-tune the last layers** on our 7,000 flood images. We reuse what the network already learned about shapes/edges/textures instead of training from scratch.

**Q: What data?**
A: **AIDERv2** aerial drone benchmark — **3,500 flooded + 3,500 clear** images (perfectly balanced). The original dataset (2 flooded / 23 clear) was a catastrophic failure we fixed by sourcing proper drone imagery.

**Q: Results on 1,400 held-out images?**
A: Overall accuracy **96.00%**, **flooded recall 95.45%**, clear recall **96.52%**. Confusion matrix: **TP=650, FN=31, TN=694, FP=25**.

**Q: Why is "flooded recall" the key metric?**
A: Recall = how many *actual* floods we catch. Missing a flood (false negative) is dangerous. **95.45% recall** means we catch ~95 of every 100 real floods. That's what earns the **PASS** verdict.

### Part B: River-Level Forecast (LSTM)

**Q: What does the LSTM do?**
A: Given the past **48 hours** of sensor data, predicts the river level **12 hours ahead** — an early warning for evacuations.

**Q: What is an LSTM?**
A: **Long Short-Term Memory** — a recurrent neural network with gated "memory cells" that remember patterns over time. It's the right tool for time-series data where order matters.

**Q: What makes yours a "Residual" FloodLSTM?**
A: Instead of predicting the absolute future level from scratch, it predicts the **delta (change)** from the current level, then adds it back:
`prediction = current_level + delta`
The model only has to learn the small movement — easier and more stable than predicting the whole absolute value.

**Q: Results?**
A: On 5,616 independent held-out windows:
| Model | MAE |
| :--- | :---: |
| Naive (stay the same) | 0.6609 |
| XGBoost | 16.62 |
| **Residual FloodLSTM** | **0.4462** |
- LSTM beats naive by ~32.5% and beats it on **72.56%** of samples.
- But there are **157 catastrophic underestimates** (>2.5 under-prediction in high stage) → **CONDITIONAL PASS**, disclosed honestly.

**Q: Why CONDITIONAL PASS and not PASS?**
A: It wins on MAE and beats-naive, but those 157 catastrophic underestimates in high-water conditions mean we can't fully trust it for evacuation decisions yet. We also only have the 12h horizon — 3h/6h models don't exist (real gap, not faked).

**Q: How is the forecasting data clean of leakage?**
A: Chronological split with a **gap** (train ends Aug 15, test starts Sep 1) + strict assertion that no test window overlaps training timestamps + scaler fit on training only.

---

## 3c. Stage 03 — NLP (Text Triage)

**Q: What does Stage 03 do?**
A: Reads real emergency messages (tweets, direct reports, news) and triages each as **LOW / MODERATE / SEVERE** — so coordinators see urgent messages instantly among thousands.

**Q: What data?**
A: **33,791 real messages** — Figure Eight disaster responses (26,180) + Kaggle "Disaster Tweets" (7,611). Stratified 80/20 split → train 27,032 / test 6,759. Labels are **derived** via disclosed rules (SEVERE = rescue/medical/death/missing; MODERATE = floods/storms/etc; LOW = rest) — disclosed honestly, not hidden.

**Q: What models did you build?**
A: Two tracks on identical data — let evidence decide, not hype:
| Track | Model | Macro-F1 | SEVERE recall |
| :--- | :--- | :---: | :---: |
| A (shipped) | TF-IDF + Logistic Regression | **0.4242** | 0.3211 |
| B | BiLSTM + attention | 0.4012 | **0.5915** |

**Q: Why ship the Logistic Regression (baseline gating)?**
A: The deep model must **clearly beat** the simple, interpretable one to justify its complexity. It wins SEVERE recall but loses macro-F1, so interpretability + safety layers win. The trade-off is disclosed, not hidden.

**Q: What is the text guard rail?**
A: Deterministic keyword floors evaluated **before** the statistical model — can only **escalate**, never downgrade: trapped/rescue/stranded/injured/SOS/baby → forced **SEVERE**; flood/storm/earthquake/evacuat/shelter → forced **MODERATE**. Fired on **2,251 / 6,759** test messages.

**Q: What is abstention (human-in-the-loop)?**
A: If confidence **< 0.50** the system returns **REVIEW** and a human operator decides — never guess on an emergency message. The dashboard shows it as a distinct purple "HUMAN TRIAGE REQUIRED" card.

**Q: What did the guard rail + abstention achieve?**
A: At threshold 0.50: auto coverage **67.4%**, macro-F1(auto) 0.4197, auto-accuracy 0.458, and **SEVERE handled (auto + human) 0.85**.

**Q: Why CONDITIONAL PASS?**
A: The raw statistical model is weak (auto-accuracy 0.458 < 0.55 gate, macro-F1 0.4197 < 0.45 gate) — but with the guard rail + abstention, SEVERE handling (0.85 ≥ 0.70) and coverage (≥ 55%) pass, making it usable as a triage aid. Label noise (derived severity) is disclosed as a limitation.

**Q: What is attention in your BiLSTM?**
A: Learned weights over the words — it tells us **which words drove the decision** (an explainability handle), and gives the deep track its SEVERE-recall edge.

---

## 4. Evaluation & Integrity (YOUR strongest selling point)

**Q: Why did you have an Evaluation Engineer?**
A: To be **adversarial and independent**. We never trust a training report — we re-run our own tests, compute our own numbers, and issue a real verdict (PASS / CONDITIONAL PASS / FAIL). This honesty is what separates us from other teams.

**Q: What integrity issues did you find and fix?**
A:
1. Scorecard **hardcoded false "PASS"** with fabricated numbers — fixed to read actual verdicts from reports (forecast now correctly shows CONDITIONAL PASS).
2. **Fake edge-case text** in vision eval (claimed wet-asphalt/puddle tests that were never run on real images) — replaced with an honest "pending" note.
3. **Units error** — master dataset river_level 0.82–221.63 (not meters); clean dataset caps at 20m for classification; we use master for forecasting (needs variance) and disclose the distinction.
4. **MODERATE recall artifact** — exposed as a label inconsistency, not hidden.

**Q: Why not just report the best numbers?**
A: In a live demo or debate, fabricated numbers get caught — and in disaster response, dishonesty is dangerous. Real, defensible, adversarially-audited numbers win.

**Q: What is baseline gating?**
A: A model is only useful if it **beats a trivial baseline**. For forecasting, the trivial baseline is "the level stays the same" (naive persistence). We prove the LSTM beats it (72.56% of samples), so the model adds real value.

---

## 4b. Model Tech Specs (memorise these numbers)

### Stage 01 — Random Forest + GuardRailedPredictor
| Parameter | Value |
| :--- | :---: |
| Algorithm | RandomForestClassifier |
| **Trees (n_estimators)** | **100** |
| **class_weight** | **balanced** |
| random_state | 42 |
| Numeric preprocess | median impute + StandardScaler |
| Categorical | zone_id → OneHotEncoder |
| **Model file size** | **86.25 MB** (`risk_model.joblib`) |
| Augmented training rows | 35,136 real + ~4,560 boundary |

### Stage 02 — Flood Vision (MobileNetV2 CNN)
| Parameter | Value |
| :--- | :---: |
| Backbone | MobileNetV2 (ImageNet pre-trained) |
| Input | 224×224×3 |
| Classifier head | Linear(1280 → 2) |
| **Total params** | **2,226,434 (≈2.23M)** |
| **Trainable params** | **2,562** (head only; backbone frozen) |
| **Model file size** | **8.73 MB** (`vision_classifier.pth`) |
| **Epochs** | **2** |
| **Batch size** | **32** |
| Optimizer | Adam, lr=0.001 |
| Loss | CrossEntropyLoss |
| Training accuracy | ~96%+ |

### Stage 02 — River Forecast (Residual FloodLSTM)
| Parameter | Value |
| :--- | :---: |
| **Input features** | **4** (river, rain, calls, closures) |
| **Lookback** | **48 hours** |
| **Horizon** | **12 hours** |
| **Hidden size** | **64** |
| **LSTM layers** | **2** |
| FC head | 64→32→1 (predicts delta) |
| **Total params** | **53,313 (≈53K)** |
| **Model file size** | **0.21 MB** (`lstm_forecaster.pth`) |
| **Epochs** | **20** |
| **Batch size** | **64** |
| Learning rate | 0.001 |
| Optimizer | **AdamW**, weight_decay 1e-4 |
| Loss | **SmoothL1 (Huber)** |
| Scheduler | ReduceLROnPlateau (factor 0.5, patience 2) |
| Gradient clip | max_norm 1.0 |
| Training windows | 21,560 |
| Test windows | 5,616 |

### Stage 03 — NLP: TF-IDF + LogisticRegression (shipped)
| Parameter | Value |
| :--- | :---: |
| Vectorizer | TF-IDF, ngrams (1,2), min_df=5, sublinear_tf, ≤200K features |
| Classifier | LogisticRegression (lbfgs) |
| class_weight | balanced |
| Abstention threshold | 0.50 (below → REVIEW / human) |
| **Model file size** | **1.4 MB** (`severity_stat.joblib`) |
| Test macro-F1 / SEVERE recall | 0.4242 / 0.3211 |

### Stage 03 — BiLSTM + attention (deep track, not shipped)
| Parameter | Value |
| :--- | :---: |
| Input | word indices, max length 64 |
| Embedding | 128-dim (learned) |
| BiLSTM | hidden 128, bidirectional |
| Attention | self-attention pooling (word-level explainability) |
| Loss | CrossEntropyLoss with inverse-frequency class weights |
| Optimizer / LR | Adam / 1e-3 |
| Epochs / Batch | 8 / 128, early stop patience 3 |
| **Model file size** | **9.06 MB** (`bilstm_severity.pth`) |
| Test macro-F1 / SEVERE recall | 0.4012 / 0.5915 |
| Guard-rail floors fired | 2,251 / 6,759 |

### XGBoost Baseline (forecast)
| Parameter | Value |
| :--- | :---: |
| **n_estimators** | **100** |
| **max_depth** | **5** |
| learning_rate | **0.08** |
| random_state | 42 |
| Model file size | 0.25 MB (`xgb_baseline.joblib`) |

---

## 5. The Dashboard (Integration)

**Q: What does the dashboard do?**
A: A **Streamlit** web app combining everything:
- **Risk classification** — user enters sensor values → LOW/MODERATE/SEVERE (with guard-rail safety).
- **Flood vision** — user uploads a drone image → FLOODED/CLEAR.
- **River forecast** — shows the 12h-ahead LSTM forecast.
- **Message triage** — paste an SOS message → LOW / MODERATE / SEVERE / **REVIEW** (classical vs deep live comparison; purple REVIEW = human triage required).
Results are colour-coded (red = critical, amber = warning, green = safe) for instant readability.

**Q: Why does integration matter?**
A: It proves the models work **together** as a live system — instant credibility in a demo. A panel trusts a working system over paper metrics.

---

## 6. Quick-Reference Numbers Card

| What | Number |
| :--- | :--- |
| Stage 01 accuracy | 0.8422 (84.22%) |
| Stage 01 edge-case accuracy | 17/17 = 100% (was 52.94% unguarded) |
| Master dataset rows | 35,136 |
| Class split | SEVERE 16,713 / LOW 13,898 / MODERATE 4,525 |
| Vision overall accuracy | 96.00% |
| Vision flooded recall | 95.45% |
| Vision confusion | TP=650, FN=31, TN=694, FP=25 |
| Vision test images | 1,400 |
| LSTM MAE | 0.4462 |
| Naive MAE | 0.6609 |
| LSTM beats-naive | 72.56% |
| Catastrophic underestimates | 157 (disclosed) |
| LSTM verdict | CONDITIONAL PASS |
| Vision verdict | PASS |
| NLP test messages | 6,759 (train 27,032 / total 33,791) |
| NLP class split | LOW 47.1% / MODERATE 42.4% / SEVERE 10.5% |
| NLP shipped macro-F1 / SEVERE recall | 0.4242 / 0.3211 |
| NLP deep macro-F1 / SEVERE recall | 0.4012 / 0.5915 |
| NLP guard-rail floors fired | 2,251 / 6,759 |
| NLP auto coverage @ 0.50 | 67.4% (SEVERE handled 0.85) |
| NLP verdict | CONDITIONAL PASS |

---

## 7. Decision Rules (memorise)
- **SEVERE**: `river_level ≥ 4.5` OR (`rainfall_72h ≥ 150` AND `river_level ≥ 3.5`)
- **MODERATE**: `river_level ≥ 3.0` OR `rainfall_72h ≥ 80` OR `emergency_calls ≥ 100`
- **LOW**: everything else

---

## 8. Safety story in one sentence
> "We combine a probabilistic ML model with deterministic expert-rule guard rails so that in a disaster-response system, the model can never under-warn below a legally/physically grounded safety threshold — and we back every claim with independent, honest, adversarial evaluation."

---

## 9. Technical Terms & Definitions (know these cold)

### ML / General
- **Machine Learning (ML):** Algorithms that learn patterns from data without being explicitly programmed. Our Random Forest "learns" which sensor readings → which risk label.
- **Deep Learning (DL):** ML using multi-layer neural networks. Our CNN (vision) and LSTM (forecast) are deep networks.
- **Supervised learning:** Training on labelled data (input → known correct output). Our classifier learns from rows that already have a risk_label.
- **Regression vs Classification:** Regression predicts a continuous number (river level). Classification predicts a category (LOW/MODERATE/SEVERE).
- **Ensemble:** Combining many weak models into one strong one. Random Forest = many decision trees voting.
- **Ensemble learning:** The technique of combining models; Random Forest is one example; bagging is the underlying mechanism.
- **Bagging (Bootstrap Aggregating):** Each tree trains on a random sample (with replacement) of the data, reducing variance/overfitting.
- **Bootstrap:** Random sampling with replacement — each tree sees a slightly different subset.
- **Overfitting:** Model memorises training data and fails on new data. Guarded against by bagging, dropout, early stopping.
- **Underfitting:** Model too simple to learn the pattern — high error on both train and test.
- **Hyperparameters:** Settings chosen before training (n_estimators=100, depth, lr). vs **parameters** (weights the model learns).
- **Feature engineering:** Creating/selecting informative input features (e.g., rolling 72h rainfall sum).
- **Feature importance:** How much each feature contributes to predictions (river_level most important).
- **One-hot encoding:** Turning categories (zone A/B/C/D) into binary 0/1 columns so ML can use them.
- **StandardScaler / Normalization:** Rescaling features to mean 0, std 1 so no feature dominates.
- **Imputation:** Filling missing values (median for numbers, most_frequent for categories).
- **Class imbalance:** Unequal class sizes (SEVERE dominant); handled via class_weight='balanced' + guard rail.
- **Data augmentation:** Creating extra synthetic boundary samples to improve coverage near thresholds.
- **Temporal leakage:** Letting future/full-dataset info leak into training. We use a time-based split with a gap to prevent it.
- **Baseline / Baseline gating:** A trivial predictor (e.g., "predict no change") a real model must beat to be useful.

### Evaluation
- **Accuracy:** (TP+TN)/total — overall correctness. Not always the best metric (see recall).
- **Precision:** Of all predicted X, how many were actually X. High precision = few false alarms.
- **Recall / Sensitivity:** Of all actual X, how many we caught. High recall = few missed floods. **Most important for safety.**
- **F1-score:** Harmonic mean of precision & recall — balanced single number.
- **Confusion matrix:** TP/FP/FN/TN table showing exactly where the model errs.
- (TP = correctly caught flood, FP = false alarm, FN = missed flood, TN = correctly clear)
- **Macro vs weighted average:** Macro = average of each class equally; weighted = weighted by class size.
- **Cross-validation:** Splitting data multiple ways, training & testing repeatedly, averaging results for robustness.
- **Edge-case test:** Testing exactly at decision thresholds (river 2.99 vs 3.01) — the most dangerous failure zone.
- **Calibration:** Whether predicted confidence matches reality (e.g., 80% confidence → correct 80% of the time).
- **Overconfidence:** Model says high confidence but is wrong; we detect and flag it.
- **Independent evaluation / Adversarial audit:** A separate evaluator re-derives metrics rather than trusting the training report.

### CNN / Vision
- **CNN (Convolutional Neural Network):** A neural net using convolution filters to detect patterns (edges, textures) in images. Great for vision.
- **Convolution / Filter / Kernel:** A small window (e.g., 3×3) sliding over the image extracting features.
- **MobileNetV2:** A lightweight CNN for efficient/edge deployment. Uses depthwise separable convolutions (fewer params, faster).
- **Depthwise separable convolution:** Splits filtering into a depthwise + pointwise step — far fewer parameters than standard conv.
- **Transfer learning:** Start from a network pre-trained on ImageNet, fine-tune the last layers on our task. Saves time/data.
- **Fine-tuning:** Continuing training of a pre-trained model on new data, maybe with a lower learning rate.
- **Freezing weights:** Keeping the backbone's weights fixed (requires_grad=False) so only the head learns.
- **ImageNet:** A huge benchmark image dataset (1M+ images, 1000 classes) used to pre-train the backbone.
- **Pretrained weights:** Weights already learned on ImageNet; our starting point.
- **DataLoader / Batch:** Loading images in groups (batch size 32) for efficient training.
- **Epoch:** One full pass over the entire training dataset.
- **Normalization (image):** Scaling pixel channels using ImageNet mean/std so inputs are well-conditioned.
- **data augmentation (vision):** Transforming images (flip, rotate, crop) to add variety — we used a balanced real dataset instead.

### RNN / LSTM / Forecasting
- **RNN (Recurrent Neural Network):** A network with internal "memory" that processes sequences step by step.
- **LSTM (Long Short-Term Memory):** An RNN with gated memory cells (forget/input/output gates) that remembers long-range patterns. Built for time-series.
- **Hidden state / hidden size:** The LSTM's internal memory vector; ours is 64-dimensional.
- **Gates (forget/input/output):** Mechanisms in the LSTM cell that decide what to remember/forget/output.
- **Sequence / sliding window:** We chop history into windows of 48 steps to predict the next 12.
- **Lookback window:** Past steps used as input (48 hours).
- **Forecast horizon:** How far ahead we predict (12 hours).
- **Time step:** One slice of the sequence (one hour of data).
- **Residual learning / Skip connection:** Model predicts the change (delta) from the last value and adds it back — easier than predicting the absolute value.
- **Delta prediction:** Outputting the *change* rather than the level; the heart of our Residual FloodLSTM.
- **SmoothL1 / Huber loss:** Loss that is L1 for large errors (robust to outliers) and L2 for small errors (smooth). Resists extreme river-peak gradients.
- **Gradient:** The direction/amount to adjust weights from the loss.
- **Gradient clipping:** Capping gradients at a max norm (1.0) to prevent exploding gradients in recurrent nets.
- **Optimizer (Adam/AdamW):** The algorithm that updates weights. AdamW adds weight decay for better generalisation.
- **Learning rate (lr):** Step size for weight updates (0.001). Too big = unstable, too small = slow.
- **Learning rate scheduler:** Automatically lowers lr when validation plateaus (ReduceLROnPlateau).
- **Early stopping:** Stop training when validation stops improving to avoid overfitting.
- **Weight decay:** Adds a penalty to large weights to reduce overfitting (regularisation).
- **Catastrophic underestimation:** Predicting far below the true dangerous level — exactly what causes evacuation failures. 157 flagged.
- **Naive persistence baseline:** The "prediction" that the value stays the same; the minimum bar for a forecast to be useful.

### NLP / Text
- **NLP (Natural Language Processing):** Making computers understand/classify text. We triage emergency messages by severity.
- **TF-IDF (Term Frequency–Inverse Document Frequency):** Converts words to numbers so rare-but-distinctive words (e.g. "trapped") weigh more than common words ("the").
- **N-gram:** Groups of n consecutive words — we use 1–2, so both "help" and "please help" are captured.
- **Bag-of-words:** Representing text by word frequencies, ignoring order.
- **Tokenisation:** Splitting text into word tokens.
- **Text cleaning / Normalisation:** Lowercasing, fixing spacing so "Flood!" and "flood" are the same token; kept light because crisis text is noisy.
- **Vocabulary:** The set of tokens the model knows (~50K for NLP).
- **Embedding:** A learned numeric vector per word capturing meaning ("flood" sits near "water" in vector space).
- **Logistic Regression (text):** A linear classifier over TF-IDF features — fast, tiny, interpretable (you can see which words push toward SEVERE).
- **BiLSTM (Bidirectional LSTM):** Two LSTMs reading a message left→right and right→left, so the model sees full context.
- **Attention (self-attention pooling):** Learned weights over words — tells us WHICH words drove a decision (explainability).
- **Class imbalance (NLP):** SEVERE is rare (10.5%); handled by balanced class weights, inverse-frequency loss weights, and guard-rail escalation.
- **Derived labels / label engineering:** Labels derived from keywords/genres with disclosed rules rather than human-labelled per message — disclosed as a limitation.
- **Abstention / Human-in-the-loop:** The model says "REVIEW" when unsure (confidence < threshold) instead of guessing; a human decides.
- **Coverage:** % of messages the system answers vs sends to humans (67.4% at threshold 0.50).
- **Baseline gating (NLP):** We ship whichever of two tracks has better held-out evidence; the deep model must clearly beat the simple one.
- **Guard rail (text):** Deterministic keyword floors evaluated before the statistical model; can only escalate, never downgrade.

### Data
- **Temporal / time-based split:** Splitting train/test by time (not randomly) so the model never sees the future.
- **Sliding window overlap:** When consecutive windows share timestamps; must not cross the train/test boundary (we enforce a gap).
- **Units error:** master dataset river_level 0.82–221.63 (raw index, NOT meters); clean dataset caps at 0.82–20.0 (meters).
- **Label artifact / Inconsistency:** Test labels that contradict the decision rules (MODERATE rows at river=20m should be SEVERE).
- **Data dictionary:** Document describing each column and its meaning/units.

### Model file terminology
- **.joblib / pickle:** Python format for serialising/saving a trained model (our risk_model + xgb_baseline).
- **.pth:** PyTorch checkpoint file holding the model's learned weights (vision_classifier, lstm_forecaster).
- **Checkpoint:** The saved weights of a trained model, loaded later for inference.
- **state_dict:** A PyTorch dict mapping layer names to their weight tensors, saved in a .pth file.
- **Serialisation / Pickling:** Saving an object (like a model) to disk so it can be reloaded later.
- **Pickle safety:** Code can be executed on unpickle — we keep classes in importable modules (`safety_guard.py`), not in `__main__`.

### Safety / Deployment
- **Guard rail / Deterministic guard:** Hard-coded expert rules layered on a probabilistic model so safety thresholds are guaranteed.
- **Safety-critical system:** A system whose failure can cause harm (disaster response) — hence the guard rail + honest evaluation.
- **Dashboard / Streamlit:** A Python web framework that turns scripts into interactive apps we demo.
- **API / Endpoint:** An interface other programs call to get predictions from a model.
- **Latency:** Time to get a prediction; MobileNetV2 is chosen to be fast enough for edge/drone use.
- **Edge deployment / Edge device:** Running the model on the device (drone, local sensor) rather than a cloud GPU.

---

## 10. More Viva Questions (depth)

**Q: What is the difference between Stage 1 ML and Stage 2 DL?**
A: Stage 1 uses a Random Forest (classical ML) on tabular sensor data for risk classification. Stage 2 uses deep neural networks — a CNN for image flood detection and an LSTM for time-series forecasting. Different data types need different models.

**Q: Why do you need both a CNN and an LSTM?**
A: They solve different problems. The CNN reads 2D images (is a drone photo flooded?). The LSTM reads a 1D time sequence (what will the river level be in 12h?). Different data structures → different architectures.

**Q: What does it mean to "freeze" the CNN backbone?**
A: We set `requires_grad=False` on all backbone layers so they keep their ImageNet weights and don't update. Only the new classifier head (1280→2) learns. This transfers general image knowledge while training only ~2,562 params — fast and strong.

**Q: Why only 2,562 trainable parameters in the CNN?**
A: Because we freeze the 2.22M-param backbone and reuse its pre-learned features. Only the final classification head is retrained — this is transfer learning, which needs far less data and compute.

**Q: Why is the forecast residual and why does it matter?**
A: Instead of predicting the absolute level, the LSTM predicts the *change* (delta) from the current level, then adds the current level back. The model only learns a small movement — easier, more stable, and robust. That's why it beats naive persistence by 32.5%.

**Q: Why SmoothL1Loss instead of MSE for the LSTM?**
A: MSE over-penalises large errors, so extreme river peaks can dominate training with huge gradients. SmoothL1 (Huber) is linear for big errors — it's robust to flood outliers, keeping training stable.

**Q: What is gradient clipping and why use it?**
A: RNNs can suffer "exploding gradients" where updates become huge and destabilise training. Clipping caps the gradient norm (here at 1.0) so updates stay controlled.

**Q: Why an LSTM and not a simple feedforward net for forecasting?**
A: Because river level depends on history/order. An LSTM has memory (gates) that remembers patterns across the 48-hour sequence. A feedforward net can't capture that temporal dependency.

**Q: How do you know the forecast isn't just memorising?**
A: We test on a held-out time period (after the train gap) the model never saw, and we compare against a naive baseline it must beat.

**Q: What is the difference between accuracy and recall, and which matters here?**
A: Accuracy is overall correctness. Recall (for flooded) is how many real floods we catch. A missed flood = danger, so **flooded recall (95.45%) is our headline safety metric**, not accuracy.

**Q: How do you decide PASS vs FAIL?**
A: Against pre-defined gating criteria — e.g., vision: flooded recall ≥ 85%; forecast: MAE < naive AND beats-naive > 50%; NLP: SEVERE handling ≥ 0.70 AND coverage ≥ 55% AND auto-macro-F1 ≥ 0.45. We enforce them consistently.

**Q: What is a label artifact and why isn't it a model bug?**
A: The test MODERATE rows all have river_level = 20.0m, which the decision rules call SEVERE. So the *labels* contradict the *rules* — the low MODERATE recall is the data's inconsistency, not the model failing to learn.

**Q: What is the most important feature and why?**
A: `river_level` (importance 0.218) plus its 72h rolling average (0.193). It makes sense — the river's height and trend are the strongest physical predictors of flooding.

**Q: How would you improve the models given more time?**
A: For vision, real edge-case imagery + more epochs/data augmentation. For the forecast, add 3h/6h horizons, more features, and address the 157 catastrophic underestimates (e.g., asymmetric loss penalising undershoot more). For ML, seek more boundary real data.

**Q: Could you use the dashboard on a real drone/sensor?**
A: Yes for the CNN (MobileNetV2 is edge-friendly), but the full pipeline currently runs on a laptop; edge deployment would need model conversion/quantisation.

**Q: Why did you build two NLP tracks instead of one?**
A: Evidence over assumptions. We let held-out macro-F1 decide rather than assuming deep learning is better — and the simple model won, so we ship interpretability.

**Q: Why ship the Logistic Regression when the BiLSTM catches more real SEVERE messages?**
A: The deep model wins SEVERE recall (0.59 vs 0.32) but hurts overall balance (macro-F1 0.40 vs 0.42). Baseline gating says the complex model must clearly win — it doesn't — and the keyword guard rail recovers the SEVERE gap in the shipped system.

**Q: How is the NLP guard rail different from Stage 01's?**
A: Same safety philosophy, different input. Stage 01 enforces numeric sensor thresholds; NLP enforces keyword floors on text. Both are deterministic and can only escalate severity, never downgrade.

**Q: What happens when the model is unsure?**
A: Abstention. Confidence below 0.50 → the message is marked REVIEW (purple "HUMAN TRIAGE REQUIRED") and a human operator decides. An honest "review" beats a wrong automated response in an emergency.

**Q: How did you handle class imbalance in NLP?**
A: `class_weight="balanced"` for the classical model, inverse-frequency loss weights for the BiLSTM, and the guard rail escalates keyword-heavy SEVERE cases so the rare class is never ignored.

**Q: How do you prevent leakage in NLP?**
A: The stratified split is made once; the shake test messages (6,759) are untouched — TF-IDF fits on train only and the BiLSTM never trains on test.

**Q: What would you improve in NLP?**
A: Human-labelled severity (instead of derived rules), domain-tuned embeddings, more SEVERE examples, and cost-sensitive training to close the SEVERE-recall gap.

---

## 11. One-Liners for Common Jargon Fumbles
- "A **threshold** is the river height at which risk flips to MODERATE/SEVERE."
- "**Hyperparameters** are settings I choose before training; **parameters** are what the model learns."
- "**Recall** = did we catch the floods that actually happened. That's the safety metric."
- "**Transfer learning** = reuse a pre-trained brain, retrain just the new top."
- "**Residual** = predict the step, not the whole staircase."
- "**Pickle** = save the model so we can load it later."
- "**Guard rail** = a hard safety rule the model can't break."
- "**TF-IDF** = rare-but-distinctive words matter — 'trapped' beats 'the'."
- "**N-grams** = catch 'please help' as well as 'help'."
- "**Attention** = the model tells me which words it looked at."
- "**Abstention** = says 'REVIEW' instead of guessing when unsure."
- "**Text guard rail** = keyword floors that can only escalate, never downgrade."

