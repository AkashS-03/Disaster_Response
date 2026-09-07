# Role 4: Deep Learning (DL) Engineer

## What I Own
I build the **Stage 02 Deep Learning models** — two completely different DL systems:
1. **Flood Vision CNN** (MobileNetV2) — detects flooding from drone images.
2. **River-Level Forecast LSTM** — predicts the river level 12 hours ahead.

## Part A: Flood Vision — MobileNetV2 CNN

### Problem
Given a drone aerial image, decide: **FLOODED** or **CLEAR**.

### Architecture
- **MobileNetV2** — a lightweight, efficient CNN ideal for edge deployment (fast, small). We fine-tune it: replace the final classifier layer with a 2-class output (`Flooded` / `Clear`).
- Transfer learning: we start from ImageNet pre-trained weights and fine-tune on our 7,000 drone images.

### Architectural Specs (key numbers)
| Parameter | Value |
| :--- | :---: |
| Backbone | MobileNetV2 (ImageNet pre-trained) |
| Input size | 224 × 224 × 3 (RGB) |
| Feature extractor input dim | `last_channel` = **1280** |
| Classifier head | Linear(1280 → 2) |
| Total parameters | **2,226,434** (≈2.23M) |
| Trainable parameters | **2,562** (only the 1280→2 classifier head) |
| Frozen parameters | ~2.22M (backbone frozen) |
| **Model file size** | **8.73 MB** (`vision_classifier.pth`) |
| Optimizer | Adam, lr = **0.001** |
| Loss | CrossEntropyLoss |
| **Epochs** | **2** |
| **Batch size** | **32** |
| Total batches/epoch | 7000/32 ≈ 219 |
| Device | CPU (torch.set_num_threads(4)) |
| Normalization | mean [0.485,0.456,0.406], std [0.229,0.224,0.225] |

### Data
- **AIDERv2** Aerial Drone Disaster benchmark.
- **3,500 flooded + 3,500 clear = 7,000 images**, perfectly balanced.
- The initial dataset failed (2 flooded / 23 clear) — we fixed it by sourcing proper drone imagery.

### Results (independent test, 1,400 held-out images)
| Metric | Value |
| :--- | :---: |
| Overall Accuracy | **96.00%** |
| Flooded Recall (catches actual floods) | **95.45%** |
| Clear Recall | **96.52%** |
| Flooded Precision | 96.30% |
| Confusion | TP=650, FN=31, TN=694, FP=25 |

- **Flooded Recall 95.45%** means we catch ~95 of every 100 real floods — critical for safety.
- **Verdict: PASS** (flooded recall ≥ 85%).

## Part B: River-Level Forecast — Residual FloodLSTM

### Problem
Given the past **48 hours** of sensor data, predict the river level **12 hours ahead**.

### Why "Residual" LSTM (the key idea)
Instead of predicting the absolute level from scratch, the model predicts the **change (delta)** from the current level, then adds it back:
```
prediction = current_level + delta
```
Why? Absolute levels are in a narrow range; the *change* is a small, learnable quantity. This is a **residual learning** trick — the model only has to get the small movement right, which is easier and far more stable.

### Architecture
`FloodLSTM(input_size=4, hidden=64, layers=2)` → the LSTM reads the 48-step sequence, a fully-connected head outputs the delta, and we add back the last observed level.

### Architectural Specs (key numbers)
| Parameter | Value |
| :--- | :---: |
| **Input size (features/step)** | **4** (river_level, rainfall, calls, closures) |
| **Sequence length (lookback)** | **48 hours** |
| **Forecast horizon** | **12 hours** |
| **Hidden size** | **64** units |
| **Number of LSTM layers** | **2** |
| LSTM dropout | 0 (none) |
| LSTM output | `out[:, -1, :]` (last timestep) |
| FC1 | Linear(64 → 32) + ReLU |
| FC2 (output head) | Linear(32 → 1) → predicts delta |
| **Total parameters** | **53,313** (≈53K) |
| **Model file size** | **0.21 MB** (`lstm_forecaster.pth`) |
| **Epochs** | **20** |
| **Batch size** | **64** |
| Learning rate | **0.001** |
| Optimizer | **AdamW**, weight_decay = 1e-4 |
| Loss | **SmoothL1Loss (Huber)** — resists extreme-peak gradients |
| Scheduler | ReduceLROnPlateau (factor 0.5, patience 2) |
| Gradient clipping | `max_norm = 1.0` |
| Training windows | 21,560 |
| Test windows | 5,616 |
| Device | CPU (torch.set_num_threads(4)) |

### Data Preparation (no leakage)
- Chronological split with a **gap**: train ends `2024-08-15`, test starts `2024-09-01`.
- Strict assertion enforces **no window overlap** between train and test.
- Scaler fits on **train only** to prevent future-data leakage.

### Results (independent, 5,616 held-out windows)
| Model | MAE | Beats-Naive |
| :--- | :---: | :---: |
| Naive Persistence (stay the same) | 0.6609 | — |
| **Residual FloodLSTM** | **0.4462** | **72.56%** |
| XGBoost baseline | 16.62 | — |

- LSTM improves on naive by **~32.5%** and beats naive on **72.56%** of samples.
- **Verdict: CONDITIONAL PASS** — it passes on MAE and beats-naive, but there are **157 catastrophic underestimates** in high-water stages (>2.5 under when stage > 6.0) that we disclose.
- **Known gap (flagged, not faked):** we only have a 12h model; 3h/6h intermediate horizons don't exist yet.

## Likely Viva Questions
1. **Why MobileNetV2?** — Lightweight CNN, fast and accurate enough for edge/drone deployment; transfer learning from ImageNet.
2. **What is transfer learning?** — Reusing a network pre-trained on a large dataset, then fine-tuning the last layers on our smaller task.
3. **Why "residual" LSTM?** — Predict the delta (change), not the absolute level; much easier, more stable, restores the current level afterwards.
4. **What is an LSTM?** — Long Short-Term Memory: a recurrent network that remembers patterns over time via gated memory cells — ideal for time-series.
5. **How do you prevent temporal leakage?** — Chronological split with a gap + train-only scaler + strict no-overlap assertion.
6. **Why CONDITIONAL PASS on the forecast?** — Good MAE but 157 catastrophic underestimates in high-water stages = can't fully trust for evacuation; disclosed honestly.
