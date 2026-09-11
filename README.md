# AquaShield Command — Disaster Response Coordination

🚨 **AquaShield Command** is an autonomous multi-agent AI system for urban crisis management, combining 🤖 Classical ML, 👁️ Deep Learning, 💬 NLP, ⚡ Small Language Models (SLMs), and 🧠 Generative AI to analyze disasters, assess risk, simulate compound scenarios, and coordinate intelligent emergency response.

---

## System Architecture & Pipeline Stages

| Stage | Domain / Role | Core Architecture | Key Artifacts | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Stage 01** | 🤖 **Classical ML Risk Classifier** | Random Forest Ensemble with deterministic `GuardRailedPredictor` over 12 hydrological sensor telemetry features. | `stage_01_ml/models/risk_model.joblib`, `stage_01_ml/safety_guard.py` | ✅ Shipped |
| **Stage 02** | 📸 **UAV Drone Aerial Vision** | MobileNetV2 transfer learning classifying inundated vs clear sectors from aerial reconnaissance feeds. | `stage_02_dl/models/vision_classifier.pth` | ✅ Shipped |
| **Stage 02** | 📈 **Hydrological Time-Series Forecaster** | 2-Layer Residual PyTorch `FloodLSTM` predicting 12-hour river gauge trajectories and levee breach crests. | `stage_02_dl/models/lstm_forecaster.pth` | ✅ Shipped |
| **Stage 03** | 💬 **Emergency Message NLP Triage** | TF-IDF + Logistic Regression with keyword safety guardrails and confidence-based human-in-the-loop review. | `stage_03_nlp/models/severity_stat.joblib` | ✅ Shipped |
| **Stage 04** | 🎙️ **Tactical Voice Briefing SLM** | Compact `TransformerLoRA` Seq2Seq (~5.31M params, ~85 ms CPU latency, 100% offline) delivering severity-conditioned voice briefings. | `stage_04_slm/models/slm_briefing.pth` | ✅ Shipped |
| **Stage 05** | 🧠 **Generative AI Scenario Studio** | Dual-engine physical telemetry and crisis text generator synthesizing 20 stress scenarios + the Wildcard Capstone *"Operation Blackout Deluge"*. | `stage_05_genai/` | ✅ Shipped |

---

## Mission Control Dashboard

Run the unified mission-control dashboard with all 6 stages:
```bash
streamlit run master_dashboard.py
```
