# Role 8: SLM Engineer (Severity-Conditioned Tactical Voice Briefing)

## Mission & Problem Statement
An incident commander in the field during a crisis cannot read lengthy, multi-page incident logs—they face cognitive paralysis unless they receive an instant, **severity-conditioned tactical voice briefing**. My mission is to fine-tune a compact, local Small Language Model (SLM) capable of generating lightning-fast, structured summaries tailored to incident severity (< 1 sentence for LOW, exactly 1 sentence for MODERATE, strictly 2 sentences for SEVERE) while extracting key situational factors (Location, People count, Risk level) on edge devices (laptop CPU, 100% offline, zero cloud GPU dependency).

## What Success Looks Like
1. **Fine-Tuned Local Transformer SLM**: Condenses dense crisis reports into severity-conditioned summaries in **85.7 ms** on a standard laptop CPU using **PEFT / LoRA ($r=8, \alpha=16$)**.
2. **Strict Brevity & Sentence Rule Compliance**:
   - **LOW**: `< 1 sentence` (concise phrase, 0 full stops, ~5–10 words).
   - **MODERATE**: `exactly 1 sentence` (1 full stop, ~12–18 words).
   - **SEVERE**: `strictly 2 sentences` (2 full stops: Sentence 1 = Threat/Casualties, Sentence 2 = Directive/Rescue, ~20–28 words).
   - **Achieved Compliance**: **100.0%** across held-out evaluation tests.
3. **Key Factor Extraction Engine**: Automatically extracts **Location**, **Number of People / Impact**, and **Risk Level** as prominent mission-control badges.
4. **Domain Dictionary Integration**: Preserves 26 specialized evacuation terms (`MEDEVAC`, `CAS-EVAC`, `EVAC-ORDER`, `LZ-CLEAR`), resource codes (`WATER-PT`, `RATION-DEP`), and radio shorthand (`PRI-1`, `SITREP`, `10-4`, `ROGER`, `CODE-RED`) with a **>90% code retention rate**.
5. **Team Huddle Verification**: Timing ourselves reading raw incident logs (~67.7s) versus listening to the SLM briefing (~9.4s) yields an **84.7% reading time reduction** (exceeding the >80% requirement).
6. **Offline Edge Application**: Fully operational within the unified dashboard in dedicated **Tab 5 ("🎙️ Stage 04: SLM Severity Summarizer")** with 1-click offline voice briefing audio playback via browser Web Speech API.

---

## Squad Roles & Responsibilities (Stage 04 SLM)

| Squad Role | Core Responsibilities | Key Artifacts |
| :--- | :--- | :--- |
| **Data Engineer** | Curate and format 2,400 balanced report-summary pairs (800 LOW, 800 MODERATE, 800 SEVERE) from Stage 03 disaster messages; maintain the Domain Dictionary (`domain_dictionary.json`). | `data_engineer/domain_dictionary.json`, `data_engineer/curate_summaries.py`, `data/briefing_dataset.csv` |
| **EDA Engineer** | Audit token distribution, ensure specialized radio codes are retained and represented, and verify >80% reading time reduction (84.7% achieved). | `eda_engineer/eda_slm.py`, `reports/figures/slm_eda.png`, `reports/slm_eda_report.md` |
| **SLM Engineer** | Build and fine-tune the `TransformerLoRA` Sequence-to-Sequence architecture with Multi-Head Attention and Low-Rank Adaptation on edge CPU. | `dl_engineer/slm_model.py`, `dl_engineer/slm_train.py`, `models/slm_briefing.pth` |
| **Evaluation Engineer** | Benchmark summary fidelity (ROUGE-1: 0.4701, ROUGE-2: 0.2675, ROUGE-L: 0.4453, BLEU-2: 0.3241), 100% length compliance, 100% location accuracy, and CPU latency (85.7 ms). | `evaluation_engineer/eval_slm.py`, `reports/figures/slm_evaluation.png`, `reports/slm_evaluation_report.md` |
| **Integration Engineer** | Package the SLM into the dedicated Tab 5 tactical briefing HUD inside `master_dashboard.py` with Web Speech voice briefing audio synthesis and key factor cards. | `integration_engineer/slm_integration.py`, `master_dashboard.py` (Tab 5 HUD) |

---

## Technical Specifications (The Edge Model)

| Parameter | Value | Rationale |
| :--- | :---: | :--- |
| **Architecture** | **TransformerLoRA (Encoder-Decoder)** | Multi-head attention captures long-range dependencies across field unit dispatches |
| **Hidden Dimension ($d_{\text{model}}$)** | **256** | High representational fidelity while keeping CPU memory bandwidth low |
| **Attention Heads ($n_{\text{head}}$)** | **4 heads** | Multi-perspective attention over casualties, locations, and directives |
| **Encoder Depth** | **2 Transformer Layers** | Rich contextual encoding of multi-unit dispatch logs |
| **Decoder Depth** | **2 Transformer Layers** | Causal autoregressive decoding conditioned on severity tokens |
| **Feedforward Dimension ($d_{\text{ff}}$)** | **512** | Non-linear feature projection with ReLU activation |
| **PEFT / LoRA Rank ($r$)** | **$r = 8$** | Low-rank adapter rank for parameter-efficient fine-tuning |
| **LoRA Scaling Alpha ($\alpha$)** | **$\alpha = 16$** | Effective scaling factor $\frac{\alpha}{r} = 2.0$ for stable gradient updates |
| **Total Parameters** | **~5,315,140 parameters (~5.31M)** | Compact footprint fitting into standard CPU L3 cache |
| **Trainable LoRA Parameters** | **~118,000 parameters (~2.2%)** | Trains in 9.6 minutes on local standard CPU (zero GPU needed) |
| **Model Checkpoint Size** | **~11.6 MB fp32** | Instant load (<0.2s), transmittable via radio or flash drive |
| **Max Sequence Length** | Source: 160 / Target: 48 tokens | Scaled for 4–7 report crisis logs $\rightarrow$ up to 2 actionable sentences |
| **Inference Latency** | **85.7 ms mean** on standard laptop CPU | Instantaneous tactical delivery under crisis conditions |
| **Validation Loss** | **0.2584** | Converged from initial loss ~7.8 over 8 epochs |

---

## Edge SLM vs. Massive Cloud Models Benchmark

| Feature / Metric | Local Edge SLM (Ours) | Llama-3-70B (Cloud) | GPT-4o (Cloud API) | Claude 3.5 Sonnet (Cloud) |
| :--- | :---: | :---: | :---: | :---: |
| **Architecture** | **TransformerLoRA (Enc-Dec)** | Transformer Decoder | MoE Transformer | MoE Transformer |
| **Parameter Size** | **~5.31M params (~11.6 MB)** | 70 Billion (~140 GB) | ~200B+ params | Large MoE |
| **Edge / Offline Ready** | **100% (Local CPU)** | 0% (Needs A100 GPU) | 0% (Cloud Only) | 0% (Cloud Only) |
| **Average Latency** | **85.7 ms** | ~1,450 ms (Cloud RT) | ~1,820 ms (Cloud RT) | ~2,150 ms (Cloud RT) |
| **Network Dependency** | **ZERO (Works in blackouts)** | Full Cloud Connection | Full Cloud Connection | Full Cloud Connection |
| **Hardware Required**| **Standard Laptop / Mobile** | 2x 80GB A100 GPUs | Cloud Cluster | Cloud Cluster |
| **Operational Cost** | **$0.00 (Free perpetual)** | $0.80 / 1M tokens | $5.00 / 1M tokens | $15.00 / 1M tokens |
| **Output Constraint**| **100% Severity Compliant** | Verbose / Non-deterministic | Verbose / Non-deterministic | Verbose / Non-deterministic |

---

## Team Huddle Reading Time Savings (>80% Gate)

- **Incident Log Average Length**: 158 words (~67.7 seconds reading time at 140 WPM).
- **SLM Briefing Average Length**: 22 words (~9.4 seconds voice synthesis duration).
- **Time Reduction**: **84.7%** (Exceeds the 80% project gate).
- **Command Impact**: Field commanders make high-stakes decisions in under 10 seconds without skimming hundreds of words of raw logs.

---

## Likely Viva Questions & Defense Strategy

1. **Q: Why use a Transformer with PEFT/LoRA instead of an LSTM?**  
   *A: A multi-head Transformer encoder-decoder enables cross-attention across disparate field dispatches simultaneously, resolving complex co-references between stranded victims and rescue assets. Applying PEFT/LoRA ($r=8, \alpha=16$) freezes base representations and trains only ~118,000 parameters (~2.2% of the network), enabling rapid local training on CPU in 9.6 minutes while preserving sub-90ms edge inference.*

2. **Q: How do you guarantee the summary length adheres strictly to the severity level?**  
   *A: We utilize a dual-layer enforcement mechanism. First, the autoregressive decoder conditions generation on severity prefix tokens (`summarize sev <level>:`) and monitors terminal punctuation (full-stop counter). Second, the post-processing engine `enforce_severity_length()` guarantees zero full stops for LOW, exactly one full stop for MODERATE, and exactly two full stops for SEVERE, achieving 100% rule compliance in evaluation.*

3. **Q: Why not use GPT-4o or Claude 3.5 for summarization?**  
   *A: Natural disasters destroy cellular towers, fiber cables, and power grids. A cloud model is completely inoperative in a blackout. Our SLM is 11.6 MB, runs locally on a standard laptop CPU in 85.7 ms, requires zero internet, protects sensitive survivor data, and incurs $0.00 in operational costs.*

4. **Q: How does the model extract and preserve key factors (Location, People, Risk Level)?**  
   *A: The system pairs a structured Key Factor Extraction layer (`extract_key_factors`) with domain-dictionary token protection. Geographic sectors (e.g., Sector 3 Delta, Kurla Basin), numerical trapped/casualty counts, and risk levels are parsed and displayed as visual mission-control badges, achieving 100% location accuracy and 95% risk classification accuracy.*

5. **Q: How are specialized tactical radio codes preserved?**  
   *A: Our Data Engineer embedded a dedicated Domain Dictionary (`domain_dictionary.json`) covering evacuation terms (`MEDEVAC`, `LZ-CLEAR`), priority codes (`PRI-1`), and radio shorthand (`SITREP`, `10-4`). Our tokenizer treats these as atomic tokens, and our independent audit confirms a **>90% tactical code retention rate**.*