"""
Stage 03 - NLP | EDA Engineer
==============================
Descriptive analysis of the real master text dataset: class balance,
length distributions, vocabulary coverage, source mix and the imbalance
story that shapes the modelling decisions downstream.
"""

import os
import json
import re
from collections import Counter

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _tok(text):
    return [t for t in re.findall(r"[a-z0-9']+", text.lower())]


def eda():
    print("=== Stage 03 NLP - EDA Engineer ===")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    proc = os.path.join(base_dir, "data", "processed")
    rep_dir = os.path.join(base_dir, "reports")
    fig_dir = os.path.join(rep_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    df = pd.read_csv(os.path.join(proc, "master_text_dataset.csv"))
    print(f"Master rows: {len(df)}")

    # ---------- 1. class distribution / imbalance ----------
    dist = df["severity"].value_counts()
    n = len(df)
    imbalance = dist / n
    print("\nSeverity distribution:")
    for cls, c in dist.items():
        print(f"  {cls:<9} {c:>6}  ({c/n*100:.1f}%)")

    sev_frac = dist.get("SEVERE", 0) / n
    mod_frac = dist.get("MODERATE", 0) / n
    low_frac = dist.get("LOW", 0) / n

    # ---------- 2. text length ----------
    lens = df["text"].str.split().str.len()
    print(f"\nWord-count: mean {lens.mean():.1f}  median {lens.median():.0f} "
          f"p95 {lens.quantile(0.95):.0f}  max {lens.max()}")

    # ---------- 3. vocabulary ----------
    vocab = Counter()
    for text in df["text"]:
        vocab.update(_tok(text))
    print(f"Vocabulary: {len(vocab)} unique tokens "
          f"(doc freq>=1); {sum(1 for v in vocab.values() if v >= 5)} with df>=5")

    # ---------- 4. source mix ----------
    print("\nBy source:")
    print(df.groupby(["source", "severity"]).size().to_string())
    print("\nGenre mix (Figure Eight):")
    print(df[df["source"] == "figure_eight"]["genre"].value_counts().to_string())

    # ---------- 5. top terms per severity ----------
    print("\nTop 15 tokens per severity class:")
    per_class = {}
    for cls in ["LOW", "MODERATE", "SEVERE"]:
        sub = df[df["severity"] == cls]
        c = Counter()
        for text in sub["text"]:
            c.update(_tok(text))
        per_class[cls] = c
        top = c.most_common(15)
        print(f"  {cls}: " + ", ".join(f"{w}×{f}" for w, f in top))

    # ---------- imbalance ratio ----------
    top_ratio = dist.max() / dist.min()
    print(f"\nImbalance: majority (LOW) is {top_ratio:.2f}x "
          f"the minority (SEVERE) class size")

    # ---------- figures ----------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = {"LOW": "#2e9e5b", "MODERATE": "#f0a324", "SEVERE": "#d23d3d"}
    axes[0].bar(dist.index, dist.values,
                color=[colors[c] for c in dist.index])
    axes[0].set_title("Severity class balance\n(real-world, naturally imbalanced)")
    axes[0].set_ylabel("messages")

    axes[1].hist(lens, bins=40, color="#1f6feb", alpha=0.85)
    axes[1].set_title("Message length (words)")
    axes[1].set_xlabel("words")

    for cls, c in per_class.items():
        axes[2].bar(list(dict(c.most_common(8)).keys()),
                    list(dict(c.most_common(8)).values()),
                    color=colors[cls], alpha=0.65, label=cls)
    axes[2].set_title("Top tokens by severity")
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "nlp_eda.png"), dpi=110)
    plt.close()
    print(f"\nFigure saved: {fig_dir}/nlp_eda.png")

    # ---------- report ----------
    with open(os.path.join(rep_dir, "eda_report_nlp.md"), "w", encoding="utf-8") as f:
        f.write("# Stage 03 NLP - EDA Report (real data)\n\n")
        f.write(f"- **Dataset size:** {n} real labelled texts\n")
        f.write(f"- **Sources:** Figure Eight (crowdsourced humanitarian msgs) + "
                f"Kaggle disaster tweets\n")
        f.write("- **Severity split:** "
                f"LOW {dist.get('LOW',0)} ({low_frac*100:.1f}%), "
                f"MODERATE {dist.get('MODERATE',0)} ({mod_frac*100:.1f}%), "
                f"SEVERE {dist.get('SEVERE',0)} ({sev_frac*100:.1f}%)\n")
        f.write(f"- **Words/message:** mean {lens.mean():.1f}, "
                f"median {lens.median():.0f}\n")
        f.write(f"- **Vocabulary:** {len(vocab)} unique tokens\n")
        f.write("- **Imbalance takeaway:** SEVERE is the minority class "
                f"({sev_frac*100:.1f}%) - exactly like reality (disasters are "
                "rare). Downstream we use class_weight + weighted loss and a "
                "deterministic guard rail, mirroring Stage 01.\n")


if __name__ == "__main__":
    eda()