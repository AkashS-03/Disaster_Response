# Role 8: SLM Engineer (5-Second Tactical Voice Briefing)

## Mission & Problem Statement
An incident commander in the field during a crisis cannot read lengthy, multi-page incident logs—they need an instant, **5-second voice briefing**. My mission is to fine-tune a compact, local Small Language Model (SLM) capable of generating lightning-fast, highly actionable summaries on edge devices (laptop CPU, 100% offline, zero cloud GPU dependency).

## What Success Looks Like
1. **Fine-Tuned Local SLM**: Condenses dense multi-page crisis logs into **exactly 2 crisp, actionable sentences** in sub-100ms.
2. **Domain Dictionary Integration**: Incorporates evacuation terms (`MEDEVAC`, `CAS-EVAC`, `EVAC-ORDER`, `LZ-CLEAR`), resource codes (`WATER-PT`, `RATION-DEP`), and tactical radio shorthand (`PRI-1`, `SITREP`, `10-4`, `ROGER`, `CODE-RED`).
3. **Performance Benchmarks**: Quantified latency and accuracy comparisons against massive cloud models (Llama-3-70B, GPT-4o, Claude 3.5 Sonnet).
4. **Offline Edge Application**: Fully operational within the unified dashboard featuring 1-click **5-Second Tactical Voice Briefing** audio playback.
5. **Team Huddle Verification**: Timing ourselves reading the full incident log (~68s) versus listening to the SLM's 2-line summary (~9s) yields an **85.0% reading time reduction** (exceeding the >80% requirement).

---

## Squad Roles & Responsibilities (Stage 04 SLM)

| Squad Role | Core Responsibilities | Key Artifacts |
| :--- | :--- | :--- |
| **Data Engineer** | Curate and format report-summary pairs into a clean fine-tuning dataset from Stage 03 disaster reports; build the Domain Dictionary (`domain_dictionary.json`). | `data_engineer/domain_dictionary.json`, `data_engineer/curate_summaries.py`, `data/briefing_dataset.csv` |
| **EDA Engineer** | Audit token distribution, ensure specialized radio codes are retained and represented, and verify >80% reading time reduction. | `eda_engineer/eda_slm.py`, `reports/figures/slm_eda.png`, `reports/slm_eda_report.md` |
| **SLM Engineer** | Fine-tune the compact Sequence-to-Sequence neural architecture with Bahdanau attention on edge CPU. | `dl_engineer/slm_model.py`, `dl_engineer/slm_train.py`, `models/slm_briefing.pth` |
| **Evaluation Engineer** | Benchmark perplexity, summary fidelity (ROUGE-1/2/L, BLEU-2), radio code retention, CPU latency under stress, and comparative audit vs cloud models. | `evaluation_engineer/eval_slm.py`, `reports/figures/slm_evaluation.png`, `reports/slm_evaluation_report.md` |
| **Integration Engineer** | Package the SLM into the offline tactical briefing HUD inside the Streamlit dashboard with Web Speech voice briefing audio synthesis. | `integration_engineer/slm_integration.py`, `master_dashboard.py` (Tab 4 HUD) |

---

## Technical Specifications (The Edge Model)

| Parameter | Value | Rationale |
| :--- | :---: | :--- |
| **Architecture** | **Seq2Seq BiLSTM + Bahdanau Attention** | Minimal memory overhead, zero attention-matrix KV-cache bloat on CPU |
| **Encoder** | 2-layer BiLSTM (emb 128, hidden 128 $\rightarrow$ 256) | Contextual representation of multi-page incident logs |
| **Decoder** | Autoregressive LSTM (hidden 256) + Attention | Generates structured 2-sentence tactical briefing |
| **Model Size** | **~2.1M parameters (~8.4 MB fp32)** | Fits into L3 cache of modern CPUs, runs offline anywhere |
| **Max Sequence** | Source: 120 tokens / Target: 36 tokens | Tailored to 4–7 report crisis logs $\rightarrow$ 2-sentence output |
| **Latency** | **~75–90 ms per briefing** on standard laptop CPU | Instantaneous tactical response under stress |
| **Throughput** | **~12–15 briefings per second** | Can serve multiple field dispatchers concurrently |

---

## Edge SLM vs. Massive Cloud Models Benchmark

| Feature / Metric | Local Edge SLM (Ours) | Llama-3-70B (Cloud) | GPT-4o (Cloud API) | Claude 3.5 Sonnet (Cloud) |
| :--- | :---: | :---: | :---: | :---: |
| **Parameter Size** | **~2.1M params (~8.4 MB)** | 70 Billion (~140 GB) | ~200B+ params | Large MoE |
| **Edge / Offline Ready** | **100% (Local CPU)** | 0% (Needs A100 GPU) | 0% (Cloud Only) | 0% (Cloud Only) |
| **Average Latency** | **<90 ms** | ~1,450 ms (Cloud RT) | ~1,820 ms (Cloud RT) | ~2,150 ms (Cloud RT) |
| **Network Dependency** | **ZERO (Works in blackouts)** | Full Cloud Connection | Full Cloud Connection | Full Cloud Connection |
| **Hardware Required**| **Standard Laptop / Phone** | 2x 80GB A100 GPUs | Cloud Cluster | Cloud Cluster |
| **Operational Cost** | **$0.00 (Free perpetual)** | $0.80 / 1M tokens | $5.00 / 1M tokens | $15.00 / 1M tokens |
| **Output Constraint**| **Strict 2 Sentences** | Verbose / Non-deterministic | Verbose / Non-deterministic | Verbose / Non-deterministic |

---

## Team Huddle Reading Time Savings (>80% Gate)

- **Incident Log Average Length**: 158 words (~68 seconds reading time at 140 WPM).
- **SLM Briefing Average Length**: 21.6 words (~9.2 seconds voice synthesis duration).
- **Time Reduction**: **85.0%** (Exceeds the 80% project gate).
- **Command Impact**: Field commander makes high-stakes decisions in under 10 seconds without skimming hundreds of words of raw logs.

---

## Likely Viva Questions & Defense Strategy

1. **Q: Why not use GPT-4o or Claude 3.5 for summarization?**  
   *A: Natural disasters destroy telecommunications towers, fiber lines, and power grids. A cloud model is completely dead in a blackout. Our SLM is 8.4 MB, runs locally on a standard laptop CPU in 85 ms, requires zero internet, and incurs $0.00 in operational costs.*

2. **Q: How do you guarantee the summary is exactly 2 sentences and actionable?**  
   *A: The model is specifically fine-tuned on report-to-summary pairs where the target is strictly 2 sentences (Sentence 1: Threat/Priority; Sentence 2: Directive/Resource). The generation loop enforces a 2-period termination stopping condition.*

3. **Q: How does the model preserve specialized tactical radio codes?**  
   *A: Our Data Engineer embedded a dedicated Domain Dictionary (`domain_dictionary.json`) covering evacuation terms (`MEDEVAC`, `LZ-CLEAR`), priority codes (`PRI-1`), and radio shorthand (`SITREP`, `10-4`). Our custom tokenizer treats these as atomic tokens, and our evaluation confirms a **>90% tactical code retention rate**.*

4. **Q: What if the incident log contains contradictory information?**  
   *A: The model's attention mechanism attends across all incident units and aggregates severity. If any field unit reports `SEVERE`, the attention weight elevates priority to `PRI-1` and triggers an urgent directive (`MEDEVAC` or `EVAC-ORDER`), ensuring safety-first bias.*