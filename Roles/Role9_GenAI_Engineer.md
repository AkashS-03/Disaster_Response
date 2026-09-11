# Role 9: GenAI Engineer (Synthetic Scenario Synthesis & Pipeline Battle-Testing)

## Mission & Problem Statement
Major natural disasters are fortunately rare, resulting in sparse, highly imbalanced historical training data. If our multi-agent crisis response system is only evaluated against historical records, it retains dangerous blind spots—especially for **compound, cascading, and adversarial disaster phenomena** (e.g. cloudbursts combined with power grid failures, blocked sea outfalls during spring high tides, or physical sensors dying in deep water).

My mission as the **GenAI Engineer** is to:
1. **Teach the system to imagine what hasn't happened yet**: Build a dual-engine Generative AI scenario pipeline that synthesizes physically consistent, multi-zone compound catastrophes.
2. **Battle-Test the Entire Pipeline**: Expose all 4 upstream stages (Stage 01 ML, Stage 02 DL LSTM/Vision, Stage 03 NLP, Stage 04 SLM) to extreme synthetic stress before real-world operational deployment.
3. **Pioneer the Team Wildcard Challenge**: Present and defend **"Operation Blackout Deluge"**—a custom compound catastrophe combining a 180 mm/hr cloudburst, a 4.8m astronomical spring tide lock, a total municipal electrical substation collapse, and submerged hospital ICU generators.

---

## What Success Looks Like
1. **Multi-Modal Scenario Generator**: Synthesizes coupled data vectors adhering to strict physical invariants:
   - **Tabular Hydrology & Telemetry**: 12 exact features required by Stage 01 ML (`risk_model.joblib`), including rolling 72-hour rain sums, infrastructure closures, and trend metrics.
   - **48-Hour Hydrograph Trajectories**: Multivariate sequences (`[river_level, rainfall, calls, closures]`) scaled for Stage 02 PyTorch FloodLSTM forecasting.
   - **Civilian SOS Messages**: Realistic distress texts for Stage 03 NLP message triage.
   - **Tactical Multi-Unit Dispatch Logs**: Incident logs embedding tactical domain codes (`PRI-1`, `MEDEVAC`, `LZ-CLEAR`, `SITREP`) for Stage 04 PEFT/LoRA SLM voice briefings.
2. **Standardized 20-Event Stress Benchmark**: Evaluated across 4 distinct operational stress regimes:
   - *Regime 1: Extreme Tail Events* (500-Year return period cloudbursts up to 210 mm/hr, 6.2m river crests).
   - *Regime 2: Multi-Zone Compound Cascades* (Flood + Electrical Substation Grid Collapse, Spring Tide Lockouts).
   - *Regime 3: Adversarial Telemetry Degraded* (Submerged 0.0m dead gauges, frozen sensors, call swarm saturation).
   - *Regime 4: Rapid Dynamic Escalation* (Flash river surge +2.5m in 45 min, transit culvert drownings).
3. **The "Wildcard" Capstone Live Defense**:
   - Invented, executed, and defended **"Operation Blackout Deluge"** live in mission control.
   - Proved that when a physical river gauge drops to 0.0m during an extreme storm, our multi-modal fusion and hard safety guardrails immediately force a **SEVERE** tactical escalation, preventing catastrophic false-safe classifications.
4. **Offline Edge & Cloud Dual-Engine Architecture**: Runs 100% offline on standard CPU in **<60 ms** without external cloud GPU dependencies, while supporting optional cloud LLMs (Gemini / OpenAI) via an interchangeable provider pattern.
5. **Dashboard Integration**: Deployed in dedicated **Tab 6 ("🧠 Stage 05: GenAI Scenario Studio")** inside `master_dashboard.py` with 1-click full-pipeline battle-testing, multi-stage scorecard visualizers, and Web Speech tactical voice audio synthesis.

---

## Squad Roles & Responsibilities (Stage 05 GenAI)

| Squad Role | Core Responsibilities | Key Artifacts |
| :--- | :--- | :--- |
| **Data Engineer** | Assembled historical disaster baseline seeds (Mumbai 2005, Kerala 2018, Hurricane Sandy, Chennai 2015) and empirical covariance structures; fitted parametric extreme-value tails. | `data_engineer/historical_seeds.json`, `data_engineer/baseline_distributions.py` |
| **EDA / Prompt Eng.** | Executed the Historical Data Blind Spot Audit (uncovering tail-risk deficit, compound concurrence omission, and sensor dropout); formulated physics-constrained prompt engineering taxonomies. | `eda_engineer/blind_spot_audit.py`, `eda_engineer/prompt_templates.py` |
| **GenAI Engineer** | Constructed the dual-engine scenario generation pipeline: physical telemetry synthesizer (`generate_tabular_12_features`, `generate_timeseries_48h`) and contextual text generator (`generate_sos_message`, `generate_tactical_dispatch`). | `genai_engineer/telemetry_synthesizer.py`, `genai_engineer/text_synthesizer.py`, `genai_engineer/scenario_generator.py` |
| **Evaluation Engineer** | Benchmarked full multi-agent pipeline performance across all 20 synthetic events + Wildcard; audited safety guardrail trips, model alignment, and hydrograph breach projections; generated figures and audit reports. | `evaluation_engineer/stress_tester.py`, `evaluation_engineer/eval_pipeline_stress.py`, `reports/genai_stress_report.md`, `reports/figures/genai_stress_evaluation.png` |
| **Integration Engineer** | Built the dashboard integration adapter and packaged the Scenario Studio into Tab 6 of `master_dashboard.py` featuring 1-click pipeline battle-testing, live response cards, and Wildcard presentation mode. | `integration_engineer/genai_integration.py`, `master_dashboard.py` (Tab 6 HUD) |

---

## Golden Numbers (Memorise These for Viva)

- **Total Scenarios Evaluated**: **21 events** (20 Standardized Benchmark Events across 4 regimes + 1 Wildcard Capstone).
- **Safety Guardrail Activation Rate**: **61.9%** across the battery (13/21 events triggered hard safety guardrails).
- **End-to-End Pipeline Alignment**: **95.2%** overall consistency between generated severity ground truth and deployed pipeline verdicts.
- **Adversarial Sensor Resilience**: **100.0%** of corrupted 0.0m dead gauges successfully detected and prevented from false-safe downgrade by the `GuardRailedPredictor`.
- **Stage 02 LSTM Levee Breaches Projected**: **11 events** correctly forecasted to breach the critical 5.0m danger barrier within 12 hours.
- **Mean Pipeline Latency under Full Load**: **53.8 ms** end-to-end across all 4 models on a standard laptop CPU (zero cloud GPU overhead).
- **Extreme Value Tail Reach**: Generated cloudburst intensities up to **220 mm/hr** (exceeding historical maximum of 145 mm/hr in training data).
- **Tactical Domain Code Preservation**: **100%** retention of critical emergency shorthand (`PRI-1`, `MEDEVAC`, `LZ-CLEAR`, `SITREP`) in synthesized dispatches.
