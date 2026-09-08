# Stage 03 NLP - EDA Report (real data)

- **Dataset size:** 33791 real labelled texts
- **Sources:** Figure Eight (crowdsourced humanitarian msgs) + Kaggle disaster tweets
- **Severity split:** LOW 15908 (47.1%), MODERATE 14335 (42.4%), SEVERE 3548 (10.5%)
- **Words/message:** mean 21.8, median 19
- **Vocabulary:** 50401 unique tokens
- **Imbalance takeaway:** SEVERE is the minority class (10.5%) - exactly like reality (disasters are rare). Downstream we use class_weight + weighted loss and a deterministic guard rail, mirroring Stage 01.
