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
The **Small Language Model (SLM)** is a 17 MB neural network we build and train ourselves. Its entire job is **next-word prediction**: given the words of a real SOS message so far, it tries to guess the next word. From a data engineer's point of view, the SLM is a consumer of the **master text dataset** — and the single most important rule about it is:

> **It is trained on 100% REAL data. No synthetic text is generated, injected, or used for augmentation anywhere in the SLM.** This rule exists because a fabricated rescue message could send teams to the wrong place.

### How the SLM consumes my data (step by step)
1. **Corpus:** all **33,791 real messages** from `master_text_dataset.csv` — 26,180 Figure Eight + 7,611 Kaggle. The same split discipline as triage applies: the **test split is never touched** by SLM training.
2. **Tokenising:** each message is lowercased and split with the regex `[a-z0-9']+` — so "Flood, water rising!" → `["flood", "water", "rising"]`. Anything outside that (Hindi text, emoji, punctuation) is dropped — a disclosed limit.
3. **Vocabulary:** I count word frequencies across all messages and keep words seen **at least 2 times**, ranked by frequency, capped at **16,000 words**; plus 4 special tokens:
   | Token | ID | Meaning |
   | :--- | :---: | :--- |
   | `<s>` | 0 | start of message |
   | `</s>` | 1 | end of message |
   | `<pad>` | 2 | padding filler |
   | `<unk>` | 3 | any word outside the vocab |
   Total **16,004** tokens. `min_freq=2` is my style: rare junk ("a1b2c3", typos) is dropped instead of kept as noise.
4. **Windows (the training view):** every real message is wrapped `<s> … words … </s>` and cut into chunks of up to **24** tokens. Each chunk is a `(input, target)` pair that says *"given these words, predict the next word"*. 33,791 messages → **48,244 windows** built purely from real text.
5. **No augmentation:** unlike Stage 01/02 boundary augmentation, the SLM has **zero synthetic rows**. If someone asks "why not add paraphrases to help it learn?" the answer is: paraphrases would be *fabricated text* — that is exactly the synthetic data our project rules prohibit.

### Data-facts cheat sheet for the debate
| Quantity | Value |
| :--- | :---: |
| Real messages used | 33,791 |
| Windows (real) | 48,244 |
| Vocabulary | 16,004 (16,000 + 4 specials) |
| Min frequency to enter vocab | 2 |
| Max sequence length | 24 tokens |
| English coverage | lowercase `[a-z0-9']` only (disclosed limit) |

### Likely SLM questions for the Data Engineer
1. **"Why so little data for a language model?"** — It is deliberately a *small* domain tool (17 MB, offline, CPU-only). Data volume is a fair criticism for big general LLMs; for our scope the honest test is whether it learned real patterns (training loss 7.62 → 5.72, well under the ~9.68 random level).
2. **"Did the SLM see the test set?"** — No. Same 80/20 discipline, test untouched, and Track C (the classifier attempt) was evaluated only on that untouched test split.
3. **"What is `<unk>` and why does it matter?"** — It catches rare words the vocab didn't keep, so the model never crashes on new text; it just marks them "unknown". This is expected and disclosed, not a bug.
