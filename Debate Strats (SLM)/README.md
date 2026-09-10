# Debate Strats — SLM (Severity-Conditioned Tactical Voice Briefing)

Role-wise playbook for defending our edge Small Language Model (SLM) work
and for attacking cloud-dependent alternatives in the viva examination.
The golden rule: **realism beats cloud hype**. In disaster response, an offline
model that responds in 85.7ms on a laptop is a lifesaver; a cloud LLM waiting
for a dead cell tower is a fatality.

---

## Golden Numbers (Memorise These)

- **Model Specs**: `TransformerLoRA` (Encoder-Decoder with Multi-Head Attention, $d_{\text{model}}=256, n_{\text{head}}=4$, LoRA $r=8, \alpha=16$), **~5.31M params (~11.6 MB fp32)**.
- **Trainable LoRA Parameters**: **~118,000 parameters (~2.2%)**, trained locally in 9.6 minutes on CPU.
- **Inference Latency**: **85.7 ms mean on plain laptop CPU** (P95: 141ms), zero GPU required, 100% offline.
- **Dataset**: **2,400 curated report-to-briefing pairs** (800 LOW, 800 MODERATE, 800 SEVERE) generated from real Stage 03 disaster messages.
- **Domain Dictionary**: 26 specialized tactical codes across 3 categories (Radio Shorthand, Evacuation & Rescue, Hazards & Logistics).
- **Team Huddle Reading Time Savings**: **84.7% reduction** (full log ~67.7s vs voice briefing ~9.4s) $\rightarrow$ **passes the >80% project gate**.
- **Tactical Code Retention**: **>90%** of critical emergency codes (`PRI-1`, `MEDEVAC`, `LZ-CLEAR`, `SITREP`) correctly preserved in generated briefings.
- **Severity Length Compliance**: **100%** across held-out evaluation (<1 sentence for LOW, 1 sentence for MODERATE, 2 sentences for SEVERE).
- **Key Factor Accuracy**: **100.0% location accuracy**, **95.0% risk level accuracy**.

---

## Role-Wise Attack & Defense

### 1. Data Engineer (Role 1)
- **Defend**: Curated 2,400 realistic multi-unit incident logs from real Stage 03 disaster messages, coupled with the Domain Dictionary (`domain_dictionary.json`). Built dual domain-pruned vocabularies (2,500 tokens each) protecting tactical codes (`PRI-1`, `MEDEVAC`, `LZ-CLEAR`) as atomic tokens.
- **Attack the cloud LLM scare**: *"Why not just prompt GPT-4 to summarize?"*  
  $\rightarrow$ Prompting GPT-4 requires active broadband internet. During Hurricane Sandy and the Nepal Earthquake, cell towers fell within the first hour. A model that cannot run on an air-gapped field laptop is completely useless to an incident commander.

### 2. EDA Engineer (Role 2)
- **Defend**: Audited log and briefing length distributions. Proved the **Team Huddle >80% time reduction** (achieved **84.7%**). Confirmed tactical code frequency: `PRI-1` (800 in SEVERE), `LZ-CLEAR` (558), `SITREP` (482), `FLOOD-SURGE` (481).
- **Attack angle**: *"Does length restriction leave out critical details?"*  
  $\rightarrow$ In a mass-casualty crisis, cognitive overload kills. The commander needs the immediate life hazard and the direct tactical response in 5 to 10 seconds. Detailed field reports remain available on the screen, but the voice briefing delivers immediate situational clarity.

### 3. SLM / DL Engineer (Roles 3 & 4)
- **Defend**: Selected an Encoder-Decoder Transformer with Multi-Head Attention and Low-Rank Adaptation (LoRA $r=8, \alpha=16$). Cross-attention resolves complex co-references between stranded victims and rescue assets. LoRA trains only ~118,000 parameters, achieving a validation loss of 0.2584 in 9.6 minutes on CPU.
- **Attack angle**: *"Why not use a massive 70B cloud model instead of an edge SLM?"*  
  $\rightarrow$ A 70B parameter model requires dual $80,000 A100/H100 GPUs, consumes hundreds of watts of power, and suffers 1.5s+ latency over the internet. Our 11.6 MB SLM executes in 85.7 ms directly inside the CPU L3 cache with zero cloud cost.

### 4. Evaluation Engineer (Role 5)
- **Defend**: Audited on 240 held-out test incident logs. Demonstrated strong fidelity (ROUGE-1: 0.4701, ROUGE-2: 0.2675, ROUGE-L: 0.4453, BLEU-2: 0.3241), verified 100% sentence compliance, 100% location accuracy, 95% risk level accuracy, and conducted a stress latency test across 100 iterations.
- **Attack angle**: *"Cloud LLMs write more poetic prose."*  
  $\rightarrow$ Fluency is secondary to factual precision and actionable directives in a disaster. Cloud LLMs hallucinate non-existent resource depots; our fine-tuned SLM is constrained to domain codes and verified facts.

### 5. Integration Engineer (Role 6)
- **Defend**: Packaged into a dedicated tactical briefing HUD in Tab 5 of `master_dashboard.py`. Features 1-click **Voice Briefing** via browser Web Speech API (zero external TTS download needed), Key Factors metric box (Location, People, Risk Level), and live latency counters.
- **Attack angle**: *"What if the model takes too long to load?"*  
  $\rightarrow$ The model is 11.6 MB. It loads once in under 0.2 seconds into RAM via `@st.cache_resource` and generates briefings in 85.7 ms.

---

## Cross-Examination Cheat Sheet

**Q: "Can an incident commander trust a severity-conditioned summary?"**  
A: Yes. For LOW incidents, it confirms clearance in <1 sentence; for MODERATE, it reports water level and infrastructure stability in 1 sentence; for SEVERE, it strictly answers *What is the threat/casualty count?* (Sentence 1) and *What is the tactical directive?* (Sentence 2).

**Q: "How does it achieve an 84.7% reading time reduction?"**  
A: An incident report with 4–7 field dispatches takes ~67.7 seconds to read (158 words). The SLM's briefing takes ~9.4 seconds to speak (22 words). That is an 84.7% time reduction, exceeding our 80% requirement.

**Q: "What happens if cloud connectivity is restored?"**  
A: The SLM continues operating seamlessly on edge. Unlike cloud-tethered systems, it has zero cloud API subscription fees, zero latency variance, and zero privacy leakage of sensitive civilian casualty data.