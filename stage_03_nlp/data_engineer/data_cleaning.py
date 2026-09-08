"""
Stage 03 - NLP | Data Engineer
===============================
Clean the real corpora and build a unified MASTER text dataset with a
documented, rule-based severity label (LOW / MODERATE / SEVERE).

Severity LABEL RULES (documented, transparent, defensible):
  SEVERE   : message indicates an active, life-threatening situation
             -> search_and_rescue | medical_help | death | missing_people
  MODERATE : message reports a real hazard, infrastructure problem or an
             aid need (floods, storm, earthquake, fire, infra, water/food/
             shelter, request, weather_related, aid_related, ...)
             Kaggle real-disaster tweets (target==1) also map to MODERATE.
  LOW      : everything else (news, general info, not related).

Rationale: no public dataset provides a ready "severity" column, so we
derive severity from expert humanitarian categories using explicit rules.
This is honest label engineering, disclosed for the viva.
"""

import os
import pandas as pd

SEVERE_COLS = ["search_and_rescue", "medical_help", "death", "missing_people"]
MODERATE_COLS = [
    "floods", "storm", "earthquake", "fire", "infrastructure_related",
    "water", "food", "shelter", "request", "weather_related", "aid_related",
    "aid_centers", "other_infrastructure", "hospitals", "direct_report",
]


def _parse_categories(cat_series):
    """Expand the ';'-separated category string into one-hot columns."""
    rows = cat_series.str.split(";", expand=True)
    labels = [c.rsplit("-", 1)[0] for c in str(cat_series.iloc[0]).split(";")]
    out = pd.DataFrame(index=cat_series.index, columns=labels, dtype="int64")
    for i in range(rows.shape[1]):
        out[labels[i]] = rows[i].str.rsplit("-", n=1).str[1].astype(int)
    # known data-quality quirk in the raw file: 'related-2' -> treat as 1
    out["related"] = out["related"].map(lambda x: 1 if x == 2 else x)
    return out


def _assign_severity(df):
    # restore index-agnostic logic: work on a copy of the label block
    labels = df.copy()
    is_severe = labels[SEVERE_COLS].sum(axis=1) > 0
    is_moderate = labels[MODERATE_COLS].sum(axis=1) > 0
    is_related = (labels["related"] == 1)
    sev = pd.Series("LOW", index=df.index)
    sev[is_moderate & is_related] = "MODERATE"
    sev[is_severe] = "SEVERE"
    return sev


def build_master():
    print("=== Stage 03 NLP - Data Engineer: Build Master Text Dataset ===")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw = os.path.join(base_dir, "data", "raw")
    proc = os.path.join(base_dir, "data", "processed")
    os.makedirs(proc, exist_ok=True)

    # ---------- 1. Figure Eight messages + categories ----------
    msg = pd.read_csv(os.path.join(raw, "disaster_messages.csv"))
    cat = pd.read_csv(os.path.join(raw, "disaster_categories.csv"))
    print(f"Figure Eight: {len(msg)} messages, {len(cat)} category rows")

    msg = msg.drop_duplicates(subset=["id"]).reset_index(drop=True)
    cats = _parse_categories(cat["categories"])
    merged = pd.concat([msg[["id", "message", "genre"]], cats], axis=1)
    merged = merged.dropna(subset=["message"])
    print(f"  after dedup/nan: {len(merged)} rows")

    fe = pd.DataFrame({
        "text": merged["message"],
        "source": "figure_eight",
        "genre": merged["genre"].fillna("unknown"),
    })
    fe["is_crisis"] = (merged["related"] == 1).astype(int)
    fe["severity"] = _assign_severity(merged)

    # ---------- 2. Kaggle disaster tweets ----------
    tw = pd.read_csv(os.path.join(raw, "disaster_tweets_train.csv"))
    tw = tw.dropna(subset=["text"]).reset_index(drop=True)
    kag = pd.DataFrame({
        "text": tw["text"],
        "source": "kaggle_disaster_tweets",
        "genre": "social",  # tweets are social-media genre
    })
    kag["is_crisis"] = tw["target"]
    kag["severity"] = tw["target"].map({1: "MODERATE", 0: "LOW"})

    # ---------- 3. unified master ----------
    master = pd.concat([fe, kag], ignore_index=True)
    master["text"] = master["text"].astype(str).str.strip()
    master = master[master["text"].str.len() > 0].reset_index(drop=True)
    master["id"] = range(1, len(master) + 1)

    master_path = os.path.join(proc, "master_text_dataset.csv")
    master.to_csv(master_path, index=False)
    print(f"\nMaster dataset: {master.shape}")
    print(master["severity"].value_counts().to_string())
    print("\nBy source:")
    print(master.groupby(["source", "severity"]).size().to_string())

    # ---------- 4. stratified 80/20 train/test split (on severity) ----------
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(
        master, test_size=0.20, random_state=42, stratify=master["severity"])
    train.to_csv(os.path.join(proc, "nlp_train.csv"), index=False)
    test.to_csv(os.path.join(proc, "nlp_test.csv"), index=False)
    print(f"\nTrain: {len(train)}  Test: {len(test)}  (stratified on severity)")

    master_meta = {
        "rows": int(len(master)),
        "train": int(len(train)),
        "test": int(len(test)),
        "sources": master["source"].value_counts().to_dict(),
        "severity": master["severity"].value_counts().to_dict(),
        "is_crisis": master["is_crisis"].value_counts().to_dict(),
    }
    import json
    with open(os.path.join(proc, "master_meta.json"), "w") as f:
        json.dump(master_meta, f, indent=2)


if __name__ == "__main__":
    build_master()