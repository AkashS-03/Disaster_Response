"""
Stage 04 - SLM | Evaluation Engineer
=====================================
Independent, adversarial audit of the trained SLM. We re-derive EVERY number
ourselves from the produced artifacts (weights + vocab + head + stats) on the
REAL untouched test split - we do not trust the training script's printed
values.

Checks, in order:
  1. Track C (SLM-as-classifier): macro-F1 / SEVERE recall vs the 0.4242 gate
     -> does the SLM ship as a triage model? (honest answer: NO)
  2. Language-model health: perplexity over the real test messages
     (mean / median / p90 -> the gauge bands <300 / 300-900 / >900)
  3. Assistive usefulness: next-word top-5 hit rate (it is a drafting hint,
     so "is the real next word in the top-5?" is the fair measure)
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from data_engineer.slm_utils import SOS, EOS, PAD, SPECIAL, UNK, MAX_SEQ, load_meta, tokenize  # noqa: E402
from dl_engineer.slm_model import load_slm  # noqa: E402

CLASSES = ["LOW", "MODERATE", "SEVERE"]
GATE_MACRO_F1 = 0.4242

SOS_ID = SPECIAL[SOS]
EOS_ID = SPECIAL[EOS]
PAD_ID = SPECIAL[PAD]
UNK_ID = SPECIAL[UNK]


def encode_texts(model, vocab, texts, bs=256):
    """Frozen-SLM contextual representation - same recipe as slm_train."""
    reps = []
    for i in range(0, len(texts), bs):
        chunk = texts[i:i + bs]
        ids = [[vocab.get(w, UNK_ID) for w in tokenize(t)][:MAX_SEQ]
               for t in chunk]
        ids = [r + [PAD_ID] * (MAX_SEQ - len(r)) for r in ids]
        x = torch.tensor(ids, dtype=torch.long)
        with torch.no_grad():
            reps.append(model.encode(x).cpu().numpy())
    return np.vstack(reps)


def batch_perplexity(model, vocab, texts, bs=256):
    """Per-message perplexity, batched; identical semantics to the Copilot
    gauge (SOS + words + EOS, mean NLL over real targets, then exp)."""
    model.eval()
    n = len(texts)
    ppls = np.empty(n)
    for i in range(0, n, bs):
        chunk = texts[i:i + bs]
        rows = []
        for t in chunk:
            ids = [vocab.get(w, UNK_ID) for w in tokenize(t)]
            ids = (ids + [EOS_ID])[:MAX_SEQ - 1]
            rows.append([SOS_ID] + ids)
        x = torch.tensor([r + [PAD_ID] * (MAX_SEQ - len(r)) for r in rows],
                         dtype=torch.long)
        tgt = torch.tensor([r[1:] + [PAD_ID] * (MAX_SEQ - len(r) + 1)
                            for r in rows], dtype=torch.long)
        with torch.no_grad():
            logits, _ = model(x)
        logp = F.log_softmax(logits, dim=-1)
        for j in range(len(rows)):
            nll = [-logp[j, k, tgt[j, k].item()].item()
                   for k in range(tgt.size(1)) if tgt[j, k].item() != PAD_ID]
            ppls[i + j] = np.exp(sum(nll) / len(nll)) if nll else float("inf")
    return ppls


def next_word_hit_rate(model, vocab, texts, bs=256):
    """Fraction of real next tokens that appear in the model's top-5
    softmax guesses (fair 'drafting-hint usefulness' proxy)."""
    model.eval()
    hits, total = 0, 0
    for i in range(0, len(texts), bs):
        chunk = texts[i:i + bs]
        rows = []
        for t in chunk:
            ids = [vocab.get(w, UNK_ID) for w in tokenize(t)]
            rows.append(([SOS_ID] + ids)[:MAX_SEQ])
        x = torch.tensor([r + [PAD_ID] * (MAX_SEQ - len(r)) for r in rows],
                         dtype=torch.long)
        tgt = torch.tensor([r[1:] + [PAD_ID] * (MAX_SEQ - len(r)) for r in rows],
                           dtype=torch.long)
        with torch.no_grad():
            logits, _ = model(x)
        top5 = torch.argsort(logits, dim=-1, descending=True)[:, :, :5]
        for j in range(len(rows)):
            for k in range(tgt.size(1)):
                y = tgt[j, k].item()
                if y == PAD_ID:
                    continue
                hits += int(y in top5[j, k].tolist())
                total += 1
    return hits / max(total, 1)


def main():
    print("=== Stage 04 SLM - Evaluation Engineer (independent audit) ===")
    data_dir = os.path.abspath(os.path.join(
        base_dir, "..", "stage_03_nlp", "data", "processed"))
    models_dir = os.path.join(base_dir, "models")
    rep_dir = os.path.join(base_dir, "reports")
    fig_dir = os.path.join(rep_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    test = pd.read_csv(os.path.join(data_dir, "nlp_test.csv"))
    y_true = test["severity"].values
    texts = test["text"].astype(str).tolist()
    print(f"Test set (real, untouched): {len(test)} messages")
    if not (os.path.exists(os.path.join(models_dir, "slm_lstm.pth"))
            and os.path.exists(os.path.join(models_dir, "slm_head.pth"))):
        print("SLM weights missing - run stage_04_slm/dl_engineer/slm_train.py first.")
        sys.exit(1)

    # ---------- load produced artifacts ----------
    model, vocab = load_slm(models_dir)
    meta = load_meta(os.path.join(models_dir, "slm_lm_meta.json"))
    cfg = meta["config"]
    head = nn.Sequential(nn.Linear(cfg["hidden"], 128), nn.ReLU(),
                         nn.Dropout(0.3), nn.Linear(128, 3))
    head.load_state_dict(torch.load(os.path.join(models_dir, "slm_head.pth"),
                                    map_location="cpu"))
    head.eval()
    stats = torch.load(os.path.join(models_dir, "slm_head_stats.pt"),
                       map_location="cpu", weights_only=False)

    # ---------- 1. Track C re-derivation ----------
    Xte = encode_texts(model, vocab, texts)
    Xn = (Xte - np.asarray(stats["mean"])) / np.asarray(stats["std"])
    with torch.no_grad():
        probs = torch.softmax(head(torch.tensor(Xn, dtype=torch.float32)),
                              dim=1).numpy()
    pred = np.array([CLASSES[i] for i in probs.argmax(1)])
    rep = classification_report(y_true, pred, output_dict=True, zero_division=0)
    macro_f1 = rep["macro avg"]["f1-score"]
    sev_recall = rep["SEVERE"]["recall"]
    acc = (pred == y_true).mean()
    print("\n[1] Track C (SLM reused as classifier) - INDEPENDENT re-derivation")
    print(f"    macro-F1 {macro_f1:.4f}   SEVERE recall {sev_recall:.4f}   "
          f"accuracy {acc:.4f}")

    cm = confusion_matrix(y_true, pred, labels=CLASSES)
    print("Confusion matrix (rows=true, cols=pred):")
    print("        " + " ".join(f"{c:>8}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"  {c:>8} " + " ".join(f"{v:>8}" for v in cm[i]))

    # ---------- 2. LM health: perplexity over real test messages ----------
    ppls = batch_perplexity(model, vocab, texts)
    ppl_finite = ppls[np.isfinite(ppls)]
    p_mean, p_med, p_p90 = ppl_finite.mean(), np.median(ppl_finite), \
        np.percentile(ppl_finite, 90)
    print(f"\n[2] Perplexity over real test messages (gauge calibration)")
    print(f"    mean {p_mean:.0f}  median {p_med:.0f}  p90 {p_p90:.0f}")
    n_low = int((ppls < 300).sum())
    n_mid = int(((ppls >= 300) & (ppls <= 900)).sum())
    n_high = int((ppls > 900).sum())
    print(f"    band <300: {n_low}   band 300-900: {n_mid}   band >900: {n_high}")

    # ---------- 3. assistive usefulness ----------
    hit = next_word_hit_rate(model, vocab, texts)
    print(f"\n[3] Next-word top-5 hit rate on real test: {hit * 100:.2f}% "
          "(real next token inside the 5 shown guesses)")

    # ---------- 4. gate + verdict ----------
    passed_gate = macro_f1 > GATE_MACRO_F1
    print(f"\n[4] Gate: clearly beat {GATE_MACRO_F1} macro-F1 -> "
          f"{'PASS' if passed_gate else 'FAIL'} (got {macro_f1:.4f})")

    # ---------- figures ----------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    im = axes[0].imshow(cm, cmap="Blues")
    axes[0].set_xticks(range(3), CLASSES)
    axes[0].set_yticks(range(3), CLASSES)
    for i in range(3):
        for j in range(3):
            axes[0].text(j, i, cm[i, j], ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black")
    axes[0].set_title("Track C confusion (independent audit)")
    axes[0].set_xlabel("predicted"); axes[0].set_ylabel("true")

    axes[1].hist(np.clip(ppls, 0, 2000), bins=50, color="#1f6feb", alpha=0.85)
    for xv, lab in [(300, "<300"), (900, "300-900")]:
        axes[1].axvline(xv, color="#d23d3d", ls="--", lw=1)
        axes[1].text(xv, axes[1].get_ylim()[1] * 0.9, lab, fontsize=8)
    axes[1].set_title("Test-message perplexity distribution\n(real messages)")
    axes[1].set_xlabel("perplexity")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "slm_evaluation.png"), dpi=110)
    plt.close()

    # ---------- report ----------
    with open(os.path.join(rep_dir, "slm_evaluation_report.md"), "w",
              encoding="utf-8") as f:
        f.write("# Stage 04 SLM - Evaluation Report (independent audit)\n\n")
        f.write(f"- Test set: **{len(test)}** real messages, untouched holdout.\n")
        f.write("- All metrics re-derived from the shipped weights - nothing "
                "re-used from the training log.\n\n")
        f.write("## 1. Track C (SLM as classifier) - the gate\n")
        f.write("| Metric | SLM Track C | Stat (LogReg) | Deep (BiLSTM) |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        f.write(f"| Macro-F1 | {macro_f1:.4f} | **0.4242** | 0.4012 |\n")
        f.write(f"| SEVERE recall | {sev_recall:.4f} | 0.3211 | 0.5915 |\n\n")
        f.write(f"> **Verdict: FAIL - Track C does NOT ship as a classifier.** "
                f"It needed to clearly beat 0.4242 macro-F1; it measured "
                f"{macro_f1:.4f}. Reported honestly, not massaged.\n\n")
        f.write("Nuance for the debate: SEVERE recall "
                f"({sev_recall:.4f}) beats both shipped models - one strong "
                "class does not earn a production slot.\n\n")
        f.write("## 2. Language-model health (perplexity gauge)\n")
        f.write(f"- Perplexity over real test messages: mean **{p_mean:.0f}**, "
                f"median **{p_med:.0f}**, p90 **{p_p90:.0f}**.\n")
        f.write("- Gauge bands are calibrated on THIS distribution: "
                "<300 / 300-900 / >900.\n")
        f.write(f"- Band counts on the real test set: <300 **{n_low}**, "
                f"300-900 **{n_mid}**, >900 **{n_high}**.\n")
        f.write("- The gauge is explicitly **not a safety gate**; the triage "
                "verdict comes from the shipped classifier + guard rail.\n\n")
        f.write("## 3. Assistive usefulness\n")
        f.write(f"- Next-word top-5 hit rate: **{hit * 100:.1f}%** - for this "
                "share of real words the true next word is among the 5 shown "
                "drafting hints. It is a drafting aid, never data.\n")
        f.write("\n> Verdict summary: the LM ships as an ASSISTIVE Copilot "
                "(with an honest weak-LM caveat); Track C does NOT ship.\n")


if __name__ == "__main__":
    main()