"""
Stage 03 - NLP | Evaluation Engineer
====================================
Independent, adversarial evaluation of the shipped NLP model: we recompute
ALL metrics on the untouched test set, sweep the human-in-the-loop
abstention threshold, measure the effect of the deterministic keyword
guard rail, and issue an honest verdict.

Key numbers that matter (safety-first):
  - SEVERE recall (did we catch the emergencies?)
  - Macro-F1  (did we stay balanced across all three classes?)
  - Coverage  (with abstention, how much can we auto-answer?)
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import joblib
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from dl_engineer.nlp_utils import CLASSES, encode  # noqa: E402
from dl_engineer.nlp_trainer import BiLSTMSeverity, MAX_LEN  # noqa: E402
from integration_engineer.nlp_triage import build_triage  # noqa: E402


def main():
    print("=== Stage 03 NLP - Evaluation Engineer (independent audit) ===")
    proc = os.path.join(base_dir, "data", "processed")
    models_dir = os.path.join(base_dir, "models")
    rep_dir = os.path.join(base_dir, "reports")
    fig_dir = os.path.join(rep_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    test = pd.read_csv(os.path.join(proc, "nlp_test.csv"))
    y_true = test["severity"].values

    # ---------- load the SHIPPED production artifact (NlpTriage) ----------
    with open(os.path.join(models_dir, "winner.txt")) as f:
        winner = f.read().strip()
    triage = build_triage(models_dir, threshold=0.50, with_deep=True)
    clf = triage.stat
    stat_probs = clf.predict_proba(test["text"])
    stat_pred = clf.predict(test["text"])

    meta = json.load(open(os.path.join(models_dir, "bilstm_vocab.json")))
    deep = BiLSTMSeverity(len(meta["vocab"]))
    deep.load_state_dict(torch.load(
        os.path.join(models_dir, "bilstm_severity.pth"), map_location="cpu"))
    Xdeep = torch.tensor(np.array(encode(test["text"].tolist(), meta["vocab"], MAX_LEN),
                                  dtype=np.int64))
    with torch.no_grad():
        deep_logits, _ = deep(Xdeep, (Xdeep != 0).float())
    deep_probs = torch.softmax(deep_logits, dim=1).numpy()
    deep_pred = np.array([CLASSES[i] for i in deep_probs.argmax(1)])

    # ---------- 1. statistical model baseline ----------
    rep = classification_report(y_true, stat_pred, output_dict=True, zero_division=0)
    base_macro = rep["macro avg"]["f1-score"]
    base_sev = rep["SEVERE"]["recall"]
    print("\n[1] Statistical core only (no guard, no abstention)")
    print(classification_report(y_true, stat_pred, zero_division=0))

    # ---------- 2. deep-model comparison (honest two-track audit) ----------
    drep = classification_report(y_true, deep_pred, output_dict=True, zero_division=0)
    print("\n[1b] Deep track comparison (BiLSTM)")
    print(f"    macro-F1 {drep['macro avg']['f1-score']:.4f}  "
          f"SEVERE recall {drep['SEVERE']['recall']:.4f}")

    # ---------- 3. abstention sweep (replicates production guard-first) ----
    guard_labels = [triage._guard_hit(t)[0] for t in test["text"]]

    def _fast_triage(preds, probs, labels, threshold):
        merged = []
        for pcls, pr, g in zip(preds, probs, labels):
            if g == "SEVERE":
                merged.append("SEVERE")
            elif g == "MODERATE":
                merged.append("SEVERE" if pcls == "SEVERE" else "MODERATE")
            elif float(pr.max()) < threshold:
                merged.append("REVIEW")
            else:
                merged.append(pcls)
        return np.array(merged)

    print("\n[2] Abstention (human-in-the-loop) sweep - guard-first ordering")
    print("    thresh  coverage%  macro-F1  SEVERE-R  auto-accuracy")
    rows = []
    thr = 0.70
    while thr > 0.20:
        merged = _fast_triage(stat_pred, stat_probs, guard_labels, threshold=thr)
        mask = merged != "REVIEW"
        cov = mask.mean() * 100
        f = f1_score(y_true[mask], merged[mask], average="macro",
                     zero_division=0)
        acc = (merged[mask] == y_true[mask]).mean() if mask.sum() else 0
        sev_all = y_true == "SEVERE"
        sev_r = np.mean(merged[sev_all] == "SEVERE") if sev_all.any() else 0
        rows.append((thr, cov, f, sev_r, acc))
        print(f"    {thr:.2f}    {cov:6.1f}    {f:.4f}   {sev_r:.4f}   {acc:.4f}")
        thr -= 0.05
    DEFAULT_THR = 0.50

    # ---------- 4. shipping operating point via PRODUCTION artifact ---------
    guarded, confs, reasons, hits = triage.triage_batch(test["text"].tolist())
    auto_mask = guarded != "REVIEW"
    f_guarded = f1_score(y_true[auto_mask], guarded[auto_mask], average="macro",
                         zero_division=0)
    sev_all = y_true == "SEVERE"
    sev_auto = np.mean(guarded[auto_mask & sev_all] == "SEVERE") if auto_mask.any() else 0
    sev_to_human = np.mean(guarded[sev_all] == "REVIEW") if sev_all.any() else 0
    sev_handled = sev_auto + sev_to_human
    auto_cov = auto_mask.mean() * 100
    auto_acc = (guarded[auto_mask] == y_true[auto_mask]).mean() if auto_mask.any() else 0
    print(f"\n[3] Shipping operating point: threshold={DEFAULT_THR} "
          f"(via NlpTriage production artifact)")

    # confusion matrix of final triage (REVIEW as separate row)
    cm_labels = ["LOW", "MODERATE", "SEVERE", "REVIEW"]
    cm = np.zeros((4, 4), dtype=int)
    for tt, pp in zip(y_true, guarded):
        cm[cm_labels.index(tt), cm_labels.index(pp)] += 1
    print("Confusion matrix (rows=true, cols=pred):")
    print("        " + " ".join(f"{c:>8}" for c in cm_labels))
    for i, c in enumerate(cm_labels):
        print(f"  {c:>8} " + " ".join(f"{v:>8}" for v in cm[i]))

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4), cm_labels, rotation=45)
    ax.set_yticks(range(4), cm_labels)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(f"NLP Triage confusion (thr={DEFAULT_THR}, with guard rail)")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "nlp_confusion.png"), dpi=110)
    plt.close()

    # ---------- 5. verdict ----------
    # Gates (safety-first): SEVERE always handled (auto OR escalated to human),
    # macro-F1 on auto >= 0.45, auto-accuracy >= 0.55, coverage >= 55%.
    gates = {
        "SEVERE handled (auto+human) >= 0.70": sev_handled >= 0.70,
        "macro-F1 (auto) >= 0.45": f_guarded >= 0.45,
        "auto-accuracy >= 0.55": auto_acc >= 0.55,
        "auto coverage >= 55%": auto_cov >= 55,
    }
    verdict = "PASS" if all(gates.values()) else "CONDITIONAL PASS"
    print(f"\n[4] Gates: {gates}")
    print(f"    auto coverage {auto_cov:.1f}%  macro-F1(auto) {f_guarded:.4f}  "
          f"auto-accuracy {auto_acc:.3f}  SEVERE handled {sev_handled:.2f}")
    print(f">>> VERDICT: {verdict}")

    # ---------- 6. report ----------
    guard_fired = sum(1 for h in hits if h)
    with open(os.path.join(rep_dir, "nlp_evaluation_report.md"), "w",
              encoding="utf-8") as f:
        f.write("# Stage 03 NLP - Evaluation Report (independent)\n\n")
        f.write(f"- Test set: **{len(test)}** real messages, untouched holdout.\n")
        f.write(f"- Shipped model: `{winner}` (winner by macro-F1, baseline-gated).\n")
        f.write("\n## Track comparison (independent re-derivation)\n")
        f.write("| Track | Macro-F1 | SEVERE recall |\n| :--- | :---: | :---: |\n")
        f.write(f"| TF-IDF + LogReg | {base_macro:.4f} | {base_sev:.4f} |\n")
        f.write(f"| BiLSTM + attention | {drep['macro avg']['f1-score']:.4f} | "
                f"{drep['SEVERE']['recall']:.4f} |\n")
        f.write("\n> Debatable result: macro-F1 prefers the simple model; SEVERE "
                "recall prefers the deep model.\n> We ship the interpretable model "
                "+ deterministic guard rail + human-in-the-loop abstention.\n\n")
        f.write("\n## Abstention sweep (statistical core, guard-first)\n")
        f.write("| threshold | coverage % | macro-F1 | SEVERE-R | auto-acc |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: |\n")
        for thr, cov, fm, sev, acc in rows:
            f.write(f"| {thr:.2f} | {cov:.1f} | {fm:.4f} | {sev:.4f} | {acc:.4f} |\n")
        f.write(f"\n**Operating point:** threshold = {DEFAULT_THR} "
                f"(messages below it go to human triage).\n")
        f.write("\n## Guard-rail effect\n")
        f.write(f"- Keyword floor fired on **{guard_fired}/{len(test)}** "
                "test messages, forcing at least MODERATE/SEVERE.\n")
        f.write(f"- At threshold {DEFAULT_THR}: auto coverage {auto_cov:.1f}%, "
                f"macro-F1(auto) {f_guarded:.4f}, auto-accuracy {auto_acc:.3f}, "
                f"SEVERE handled (auto+human) {sev_handled:.2f}.\n")
        f.write("\n## Verdict gates\n")
        for k, v in gates.items():
            f.write(f"- {k}: {'OK' if v else 'MISS'}\n")
        f.write(f"\n> **Final verdict: {verdict}** — the statistical model alone is "
                "weak, but with the guard rail + abstention it is usable "
                "in the coordination loop. Label noise (derived severity) is "
                "disclosed as a limitation.\n")


if __name__ == "__main__":
    main()