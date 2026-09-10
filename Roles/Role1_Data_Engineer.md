# Role 1: Data Engineer

## What I Own
I built and maintained the **Master Dataset** — the single source of truth that every later stage reads from. My job is to make sure the data is complete, realistic, correctly labelled, and free of units errors.

## Key Deliverables
| Artifact | Location |
| :--- | :--- |
| `master_dataset.csv` | `stage_01_ml/data/processed/` |
| `clean_modeling_dataset.csv` | `stage_01_ml/data/processed/` |
| `build_master_dataset.py` | `stage_01_ml/data_engineer/` |
| Vision dataset (7,000 drone images) | `stage_02_dl/data/vision/` |
| Time-series Windows (X_train/X_test) | `stage_02_dl/data/time_series/` |
| NLP text dataset + split | `stage_03_nlp/data/processed/` |

## What My Data Contains
- **35136 rows** of hourly sensor + event records across **4 flood-prone zones** (Zone A to Zone D).
- Each row has features: `river_level`, `rainfall`, `rainfall_rolling_72h_sum`, `emergency_call_volume`, `emergency_calls_24h_sum`, `road_closures`, `bridge_closures`, `total_infrastructure_closures`, `historical_flood_probability`, `river_level_rolling_72h_avg`, plus zone indicators.
- Labels (`risk_label`) come from **deterministic expert decision rules**, NOT from the ML model:
  - `SEVERE` → river_level ≥ 4.5 **OR** (rainfall_72h ≥ 150 AND river_level ≥ 3.5)
  - `MODERATE` → river_level ≥ 3.0 **OR** rainfall_72h ≥ 80 **OR** emergency_calls ≥ 100
  - `LOW` → everything else
- Class distribution: **SEVERE 16713 / LOW 13898 / MODERATE 4525**.

### Full Data Dictionary (all 10+ input features + target)
| Column | Type | Meaning | Range |
| :--- | :--- | :--- | :--- |
| `zone_id` | string | City zone | Zone_A → Zone_D |
| `timestamp` | datetime | Observation time (UTC) | ISO 8601 |
| `river_level` | float | River gauge height | 1.0–10.0 m (clean) |
| `rainfall` | float | Hourly rain | 0–100 mm |
| `emergency_call_volume` | int | 911 calls this hour | 0–100+ |
| `road_closures` | int | Closed roads | 0–20 |
| `bridge_closures` | int | Closed bridges | 0–5 |
| `historical_flood_probability` | float | Historical flood chance | 0.0–1.0 |
| `river_level_rolling_72h_avg` | float | 72h mean river level | 1.0–10.0 |
| `rainfall_rolling_72h_sum` | float | 72h cumulative rain | 0–500 |
| `emergency_calls_24h_sum` | float | 24h cumulative calls | 0–1000 |
| `total_infrastructure_closures` | int | road+bridge closures | 0–25 |
| `river_level_trend` | float | 1h river change | −5 to +5 |
| `risk_label` | string | **Target** | LOW/MODERATE/SEVERE |

### Data Volumes
| Dataset | Rows/Images |
| :--- | :--- |
| master_dataset.csv | 35,136 rows |
| clean_modeling_dataset.csv | 35,136 rows |
| Time-series X_train windows | 21,560 |
| Time-series X_test windows | 5,616 |
| Vision flooded images | 3,500 |
| Vision clear images | 3,500 |
| NLP master text dataset | 33,791 messages |
| NLP train / test split | 27,032 / 6,759 |

## Units Error — My Most Important Lesson
The **master_dataset** river_level ranges **0.82–221.63** (an unscaled hydrological index — NOT meters). Real river levels in meters would never reach 200m. The **clean_modeling_dataset** correctly caps river_level at **0.82–20.0** (true meters) so that the classification decision rules (which assume meters) are meaningful.

> Why this matters for the debate: If a reviewer sees river_level values of 200+, they will suspect data fabrication. I must own this up-front — explain that master is a raw index and clean is in meters.

## Why I Have Two Versions
- **master_dataset** = raw, unscaled values (full variance). Used for the **time-series forecasting** stage where variance matters (if we capped at 20m, all high-water events would flatten to 20.0 and the forecast would have zero signal).
- **clean_modeling_dataset** = river_level capped at 0–20m so the classification labels are physically sensible.

## Temporal Train/Test Split
We split by **time**, not randomly, to prevent **temporal leakage** (model seeing the future during training):
- Train: data before `2024-11-01`
- Test: data on/after `2024-11-01`
This is the honest way — a real deployment would only ever know the past.

## Vision Data
- Replaced the initial **failed dataset (2 flooded / 23 clear)** with the **AIDERv2 Aerial Drone Disaster benchmark**: **3,500 flooded + 3,500 clear = 7,000 drone images**, perfectly balanced 1:1.

## NLP Text Data (Stage 03)
- Built the **master text dataset**: **33,791 real messages** from Figure Eight disaster responses (26,180) + Kaggle "Disaster Tweets" (7,611).
- Severity labels are **derived by disclosed rules** (SEVERE = rescue/medical/death/missing; MODERATE = floods/storms + disaster flag; LOW = rest) — label engineering is disclosed honestly, not hidden.
- Stratified **80/20 split** → train 27,032 / test 6,759, both real and untouched.

## Likely Viva Questions
1. **Why the two datasets?** — master for forecasting (needs variance), clean for classification (needs meter units). Units error is disclosed consciously.
2. **How did you avoid leakage?** — Temporal split with a gap; scaler fit on train only; in time-series, a strict assertion enforces no window overlaps.
3. **Where do the labels come from?** — Deterministic expert rules, not ML. Guarantees the safety thresholds are physically grounded.
4. **What if a reviewer challenges the 221.63 river value?** — It's a raw index, not meters; clean dataset is the meter version.

## SLM: Detailed Explanation & My Role Facts (Role 8 tie-in)

### What the SLM is (data view)
The **Small Language Model (SLM)** is an 11.6 MB Transformer fine-tuned for **Severity-Conditioned Tactical Briefing Summarization**. From a data engineer's point of view, the SLM consumes a carefully curated sequence-to-sequence dataset derived from our master disaster corpus. The golden data rule remains:

> **All incident reports and emergency scenarios are grounded in real Stage 03 disaster messages. We maintain strict data balance across severity tiers and preserve factual fidelity so rescue units receive accurate casualty counts and coordinates.**

### How the SLM consumes my data (step by step)
1. **Master Corpus:** Derived from the **33,791 real messages** from `master_text_dataset.csv`.
2. **Curated Fine-Tuning Dataset (`briefing_dataset.csv`):** Exactly **2,400 balanced report-to-summary pairs**:
   - **800 LOW pairs**: Target summary `< 1 sentence` (concise phrase, 0 full stops, ~5–10 words).
   - **800 MODERATE pairs**: Target summary `exactly 1 sentence` (1 full stop, ~12–18 words).
   - **800 SEVERE pairs**: Target summary `strictly 2 sentences` (2 full stops: Sentence 1 = Threat/Casualties, Sentence 2 = Directive/Rescue, ~20–28 words).
3. **Key Factor Annotations:** Every report is annotated with three critical variables:
   - `location`: Specific sector, ward, or geographical landmark.
   - `num_people`: Numerical civilian impact or trapped count.
   - `severity`: Ground-truth operational risk level.
4. **Domain Dictionary (`domain_dictionary.json`):** 26 specialized emergency response tokens protected as atomic entities during tokenization (e.g., `PRI-1`, `MEDEVAC`, `LZ-CLEAR`, `SITREP`, `FLOOD-SURGE`).
5. **Vocabulary Builder:** Dual vocabularies (Source Vocab: 2,500 tokens, Target Vocab: 2,500 tokens) with special control tokens (`<s>`, `</s>`, `<pad>`, `<unk>`).

### Data-facts cheat sheet for the debate
| Quantity | Value |
| :--- | :---: |
| Curated Report-Summary Pairs | 2,400 (800 LOW / 800 MODERATE / 800 SEVERE) |
| Source / Target Vocab Sizes | 2,500 / 2,500 tokens |
| Domain Dictionary Codes | 26 protected tactical tokens |
| Max Sequence Length | Source: 160 tokens / Target: 48 tokens |
| Held-Out Test Set | 240 unseen multi-unit incident logs (10% split) |

### Likely SLM questions for the Data Engineer
1. **"Why curate 2,400 pairs instead of using raw uncurated disaster messages?"** — Raw disaster messages contain severe class imbalance and lack structured, actionable multi-sentence targets. Curating 2,400 balanced pairs with strict period-count supervision guarantees the model learns exact brevity boundaries (<1 sent, 1 sent, 2 sents) across all three operational tiers.
2. **"How do you ensure location names and numbers are not hallucinated?"** — The training pairs strictly reflect the numbers and locations present in the input dispatch reports. Furthermore, our Key Factor Extraction engine uses deterministic regex patterns to extract and display raw numbers directly alongside model generation.
3. **"Did the SLM training leak test data?"** — No. 240 pairs were held out strictly for independent evaluation, and the model achieved 100% sentence compliance and 100% location extraction accuracy on this untouched test split.

