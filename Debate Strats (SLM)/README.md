# Debate Strats — SLM (5-Second Tactical Voice Briefing)

Role-wise playbook for defending our edge Small Language Model (SLM) work
and for attacking cloud-dependent alternatives in the viva examination.
The golden rule: **realism beats cloud hype**. In disaster response, an offline
model that responds in 85ms on a laptop is a lifesaver; a cloud LLM waiting
for a dead cell tower is a fatality.

---

## Golden Numbers (Memorise These)

- **Model Specs**: Compact Seq2Seq BiLSTM + Bahdanau Attention, **~2.1M params (~8.4 MB fp32)**.
- **Inference Latency**: **<90 ms on plain laptop CPU**, zero GPU required, 100% offline.
- **Dataset**: **2,400 curated incident log-to-tactical summary pairs** generated from Stage 03 real disaster messages.
- **Domain Dictionary**: 26 specialized tactical codes across 3 categories (Radio Shorthand, Evacuation & Rescue, Hazards & Logistics).
- **Team Huddle Reading Time Savings**: **85.0% reduction** (full log ~68s vs 5-second voice briefing ~9s) → **passes the >80% project gate**.
- **Tactical Code Retention**: **>90%** of critical emergency codes (`PRI-1`, `MEDEVAC`, `LZ-CLEAR`, `SITREP`) correctly preserved in generated briefings.
- **Brevity Compliance**: **100%** of outputs strictly adhere to **2 crisp, actionable sentences**.

---

## Role-Wise Attack & Defense

### 1. Data Engineer (Role 1)
- **Defend**: Curated 2,400 realistic multi-page incident logs from real Stage 03 disaster messages, coupled with the Domain Dictionary (`domain_dictionary.json`). Built a custom tokenizer that protects tactical codes (`PRI-1`, `MEDEVAC`, `LZ-CLEAR`) as atomic tokens.
- **Attack the cloud LLM scare**: *"Why not just prompt GPT-4 to summarize?"*  
  → Prompting GPT-4 requires active internet. During Hurricane Sandy and the Nepal Earthquake, cell towers fell within the first hour. A model that cannot run on an air-gapped field laptop is useless to an incident commander.

### 2. EDA Engineer (Role 2)
- **Defend**: Audited log and briefing length distributions. Proved the **Team Huddle >80% time reduction** (achieved **85.0%**). Confirmed tactical code frequency: `SITREP` (66.5%), `PRI-1/2` (72.4%), `LZ-CLEAR/HOT` (100% actioned).
- **Attack angle**: *"Does 2 sentences leave out critical details?"*  
  → In a mass-casualty crisis, cognitive overload kills. The commander needs the immediate life hazard and the direct tactical response in 5 seconds. Detailed unit logs remain available on the screen, but the voice briefing delivers immediate clarity.

### 3. SLM / DL Engineer (Roles 3 & 4)
- **Defend**: Selected an Attention-based Seq2Seq architecture over massive transformer decoders because it provides sub-100ms deterministic inference on CPU without massive KV-cache RAM consumption. Trains in ~2.5 minutes on CPU.
- **Attack angle**: *"Isn't an LSTM outdated compared to Transformers?"*  
  → Transformers are designed for open-domain generation where parameter scale matters. For structured, domain-constrained incident summarization on an edge device with strict latency (<100ms) and zero GPU, our attention-equipped SLM executes in 85ms with an 8.4 MB footprint. A 70B transformer requires an $80,000 server.

### 4. Evaluation Engineer (Role 5)
- **Defend**: Audited on 240 held-out test incident logs. Demonstrated strong fidelity (ROUGE-1: 0.52+, ROUGE-L: 0.48+, BLEU-2: 0.41+), verified >90% domain code retention, and conducted a stress latency test across 100 iterations.
- **Attack angle**: *"Cloud LLMs write more fluent prose."*  
  → Fluency is secondary to factual precision and actionable directives in a disaster. Cloud LLMs hallucinate non-existent resource depots; our fine-tuned SLM is constrained to domain codes and verified facts.

### 5. Integration Engineer (Role 6)
- **Defend**: Packaged into an offline tactical briefing HUD within the unified dashboard. Features 1-click **5-Second Voice Briefing** via browser Web Speech API (zero external TTS download needed), interactive tactical code badges, and a live latency counter.
- **Attack angle**: *"What if the model takes too long to load?"*  
  → The model is 8.4 MB. It loads once in under 0.2 seconds into RAM and generates briefings in 85ms.

---

## Cross-Examination Cheat Sheet

**Q: "Can an incident commander trust a 2-line summary?"**  
A: Yes, because the 2 lines are strictly structured: Sentence 1 gives the highest-priority threat and location (`PRI-1 SITREP`); Sentence 2 gives the tactical resource directive (`MEDEVAC / LZ-CLEAR`). It answers *What is the danger?* and *What do we do right now?*

**Q: "How does it achieve an 85% reading time reduction?"**  
A: An incident log with 4–7 field reports takes ~68 seconds to read (158 words). The SLM's 2-sentence briefing takes ~9 seconds to speak (21 words). That is an 85.0% time reduction, exceeding our 80% requirement.

**Q: "What happens if cloud connectivity is restored?"**  
A: The SLM continues operating seamlessly on edge. Unlike cloud-tethered systems, it has zero cloud API subscription fees, zero latency variance, and zero privacy leakage of sensitive civilian casualty data.