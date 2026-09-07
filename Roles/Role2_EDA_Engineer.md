# Role 2: EDA Engineer (Exploratory Data Analysis)

## What I Own
I explore the data **before** modelling to understand distributions, correlations, class balance, and boundary behaviour. My insights drive feature choice and guard against silent failures like class imbalance and leakage.

## Key Activities
1. **Distribution & range analysis** — flagged the river_level range as implausible for meters (0.82–221.63) → drove the clean vs master distinction.
2. **Class balance audit** — SEVERE (16713), LOW (13898), MODERATE (4525). Imbalance is real; we do NOT blind-rebalance, we handle it at evaluation time per-class.
3. **Feature correlation** — rainfall rolling sums, historical flood probability correlate with risk; guard against data leakage from future-leaking features.
4. **Boundary behaviour exploration** — mapped the exact decision-rule thresholds (river ≥ 4.5, rain ≥ 150, calls ≥ 100) so edge-case tests target the right regions.

## Key Insight: The "decision boundary" is where safety lives
The most dangerous failure mode for a disaster classifier is **at the boundary** — when river_level is 4.49m vs 4.51m. A tiny measurement error, or a model that's 1% off, flips a safe day into a catastrophic day (or hides one). My EDA identifies which regions the classifier must get exactly right.

## Data Augmentation for Boundary Coverage
To give the classifier enough examples at the decision boundaries, we **augment** the training set with synthetic boundary rows:
- 4 zones × (10 river values + 6 rainfall values + 3 call-volume values) = many combinations
- Repeat factor 60 → ~4,560 augmented rows layered on top of 35,136 real training rows
- Purpose: **over-sample the exact boundary regions** so the Random Forest learns them, rather than treating them as rare.

## Surface-level Result
Augmentation alone did NOT fix the severe-boundary recall problem — the underlying boundary examples are physically few. That is exactly why we layered **GuardRailedPredictor** (see Evaluation Engineer) on top: deterministic rules guarantee the boundary is always respected.

## Likely Viva Questions
1. **What is class imbalance and how did you handle it?** — We kept labels real, evaluated per-class, and augmented boundaries; determinism (guard rails) is the guarantee, not rebalancing.
2. **How do you know you have leakage?** — EDA checks each feature's meaning; rolling sums/72h averages from the past, never the future.
3. **Why augment if it didn't fix the boundary recall?** — Augmentation improved boundary *coverage*; the deterministic guard rail is what guarantees correctness.
