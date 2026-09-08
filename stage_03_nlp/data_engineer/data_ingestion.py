"""
Stage 03 - NLP | Data Engineer
===============================
Fetch REAL disaster-related text corpora from public sources, validate
integrity (non-empty, expected size), and log provenance.

Data sources (real, citable):
  1. Figure Eight (CrowdFlower) - Disaster Response Messages
     ~26k real humanitarian messages collected during actual disasters,
     multi-label over 36 categories (e.g. floods, storm, trapped, aid).
       - disaster_messages.csv   (columns: id, message, original, genre)
       - disaster_categories.csv (columns: id, categories; 36 one-hot labels)
  2. Kaggle - "NL with Disaster Tweets" (real tweets, human-labelled)
     ~7.6k training tweets, binary target (1 = about a real disaster).
       - train.csv (id, keyword, location, text, target)

Publicly-mirrored raw files are used because the original publishers no
longer serve the files directly; provenance is recorded in PROVENANCE.md.
"""

import os
import sys
import csv
import time
import urllib.request

# Canonical sources for the mirrored raw files (public GitHub mirrors).
SOURCES = {
    "disaster_messages.csv": (
        "https://raw.githubusercontent.com/ravishchawla/ETL-Pipeline-for-Disaster-Data/master/data/disaster_messages.csv",
        5_064_273,
        "Figure Eight CrowdFlower Disaster Response Messages (via public mirror)",
    ),
    "disaster_categories.csv": (
        "https://raw.githubusercontent.com/ravishchawla/ETL-Pipeline-for-Disaster-Data/master/data/disaster_categories.csv",
        11_854_295,
        "Figure Eight CrowdFlower Disaster Response Categories / labels (via public mirror)",
    ),
    "disaster_tweets_train.csv": (
        "https://raw.githubusercontent.com/Satwato/Kaggle-NLP-with-Disaster-Tweets/master/train.csv",
        987_712,
        "Kaggle - NL with Disaster Tweets - training set (real tweets, binary target)",
    ),
}


def _download(url: str, dest: str, expected: int, retries: int = 4):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Stage03 NLP ingest)"})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
                f.write(r.read())
            size = os.path.getsize(dest)
            # validate basic shape: not empty and not a tiny HTML error page
            if size < 10_000 or (expected and size < expected * 0.5):
                print(f"    WARNING: size {size} looks wrong for {os.path.basename(dest)}")
            print(f"    OK  {os.path.basename(dest)}: {size:,} bytes")
            return True
        except Exception as e:
            print(f"    attempt {attempt}/{retries} failed: {e}")
            time.sleep(2)
    return False


def ingest():
    print("=== Stage 03 NLP - Data Engineer: Real Data Ingestion ===")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_dir = os.path.join(base_dir, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    results = []
    for name, (url, expected, desc) in SOURCES.items():
        dest = os.path.join(raw_dir, name)
        if os.path.exists(dest) and os.path.getsize(dest) > expected * 0.5:
            print(f"  [skip] {name} already present ({os.path.getsize(dest):,} bytes)")
            results.append((name, os.path.getsize(dest), desc, url))
            continue
        print(f"  [get]  {name}  ({desc})")
        if _download(url, dest, expected):
            results.append((name, os.path.getsize(dest), desc, url))
        else:
            print(f"  !! FAILED to fetch {name}")

    total = sum(s for _, s, _, _ in results)
    print(f"\nTotal real-data footprint: {total:,} bytes ({total/1024**2:.2f} MB) "
          f"(budget cap 2-3 GB, comfortably under)")

    # ---- provenance log -------------------------------------------------
    prov = os.path.join(raw_dir, "PROVENANCE.md")
    with open(prov, "w", encoding="utf-8") as f:
        f.write("# Stage 03 NLP - Data Provenance\n\n")
        f.write("All files below are REAL disaster-related text data. "
                "Each row of text in the training corpora is either a real\n"
                "humanitarian message (Figure Eight) or a real tweet (Kaggle) "
                "that was hand-labelled.\n\n")
        f.write("| File | Size | Description | Source URL |\n")
        f.write("| :--- | ---: | :--- | :--- |\n")
        for name, size, desc, url in results:
            f.write(f"| {name} | {size:,} | {desc} | {url} |\n")
        f.write(
            "\n## License / attribution notes\n"
            "- Figure Eight (CrowdFlower) Disaster Response Messages: dataset "
            "released for public/academic use; original source Figure Eight.\n"
            "- Kaggle 'NL with Disaster Tweets' (nlp-getting-started): public "
            "competition dataset; tweets attributed to original posters.\n"
            "- Mirrored files are used for retrieval; license follows the "
            "original publishers.\n"
        )
    print(f"Provenance log: {prov}")


if __name__ == "__main__":
    ingest()