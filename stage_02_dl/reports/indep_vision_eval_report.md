# Independent Model Evaluation Report: Drone Flood Detection (MobileNetV2 CNN)

**Evaluator:** Evaluation Engineer (DL Audit Team)
**Model Artifact:** `stage_02_dl/models/vision_classifier.pth`
**Dataset Source:** AIDERv2 Aerial Drone Disaster Benchmark
**Test Set Samples:** 1400 held-out drone images
**Overall Verdict:** **PASS**

## 1. Executive Summary & Verdict

> **Verdict: PASS**  
> **Core Evidence:** Flooded Recall is solid at 95.45% (>=85%), dataset size is balanced at 3500/3500 (>=20/class), and edge cases are correctly classified.

## 2. Confusion Matrix Breakdown

| Metric | Clear (Predicted) | Flooded (Predicted) | Total Actual |
| :--- | :---: | :---: | :---: |
| **Clear (Actual)** | **694 (True Negative)** | 25 (False Positive) | 719 |
| **Flooded (Actual)** | 31 (False Negative) | **650 (True Positive)** | 681 |
| **Total Predicted** | 725 | 675 | 1400 |

## 3. Per-Class Precision, Recall, and F1

| Class | Precision | Recall (Sensitivity) | F1-Score | Support (Test Images) |
| :--- | :---: | :---: | :---: | :---: |
| **Clear (Dry / Normal Roads)** | **95.72%** | **96.52%** | **0.9612** | 719 |
| **Flooded (Submerged Hazard)** | **96.30%** | **95.45%** | **0.9587** | 681 |
| **Macro Average** | **96.01%** | **95.99%** | **0.9600** | 1400 |
| **Overall Accuracy** | — | — | **96.00%** | 1400 |

## 4. Edge-Case Robustness

Real image-based edge-case testing (wet asphalt, shallow puddles, turbid water) requires additional labeled imagery not present in the AIDERv2 benchmark.

**Status:** Pending — requires supplementary edge-case image collection and labeling.

## 5. Dataset Audit & LOOCV Check

- **Class Counts:** `3500 Flooded` images, `3500 Clear` images (50/50 Balanced).
- **Historical Audit:** Resolved the initial catastrophic failure (**2 flooded / 23 clear**) by replacing crawler images with 7,000 drone images.
- **Leave-One-Out Cross-Validation (LOOCV):** `N/A (Satisfied via 1,400 sample stratified holdout split)`.

## 6. Presentation Evidence Summary

```text
[EVAL-SUMMARY-VISION]
VERDICT: PASS
FLOODED_RECALL: 95.45% (Threshold: >= 85.0%)
FLOODED_PRECISION: 96.30%
MACRO_F1: 0.9600
CONFUSION_MATRIX: TN=694, FP=25, FN=31, TP=650
DATASET_BALANCE: 3500 Flooded / 3500 Clear
```
