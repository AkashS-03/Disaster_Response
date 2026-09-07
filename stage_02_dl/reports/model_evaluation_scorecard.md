# Deep Learning Stage 2: Unified Model Evaluation Scorecard

| Model Track | Model Architecture | Primary Benchmark Metric | Gating Criterion | Status / Verdict |
| :--- | :--- | :--- | :--- | :---: |
| **Flood Vision** | MobileNetV2 (CNN) | Flooded Recall: **95.45%** (Acc: 96.00%) | Recall >= 85.0%, Size >= 20/class | **PASS** |
| **River Forecasting** | Residual FloodLSTM | MAE: **0.4462** (Beats Naive: **72.56%**) | MAE < Naive, Beats Naive > 50% | **UNKNOWN** |

### Executive Recommendation for Deployment
Vision model: **PASS**. Forecasting model: **UNKNOWN**.
Checkpoints are active in `master_dashboard.py`.
