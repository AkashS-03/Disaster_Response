"""
Stage 03 - NLP | DL Engineer
============================
Train TWO model tracks on the real master dataset and let the held-out
evidence decide which one ships (baseline gating philosophy):

  Track A - TF-IDF + Linear SVM    (classical, interpretable, tiny)
  Track B - BiLSTM + attention     (deep learning, self-attention pooling)

The winner is chosen by macro-F1 (SEVERE-safe) on the untouched test set.
Artifacts are saved to stage_03_nlp/models/ and a training report is
written to stage_03_nlp/reports/model_training_nlp.md.
"""

import os
import json
import sys

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from dl_engineer.nlp_utils import CLASSES, tokenize, build_vocab, encode  # noqa: E402

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

EMBED_DIM = 128
HIDDEN = 128
MAX_LEN = 64
BATCH = 128
EPOCHS = 8
LR = 1e-3


class BiLSTMSeverity(nn.Module):
    """BiLSTM with attention pooling over non-pad timesteps.

    The attention weights give the model an explainability handle: we can
    inspect WHICH words drove a decision (great for a safety-critical viva).
    """
    def __init__(self, vocab_size, num_classes=3, pad_idx=0, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=pad_idx)
        self.lstm = nn.LSTM(EMBED_DIM, HIDDEN, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(HIDDEN * 2, 1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(HIDDEN * 2, num_classes)

    def forward(self, x, mask):
        e = self.emb(x)                       # (B,T,E)
        out, _ = self.lstm(e)                 # (B,T,2H)
        scores = self.attn(out).squeeze(-1)   # (B,T)
        scores = scores.masked_fill(mask == 0, -1e9)  # safe finite fill
        w = torch.softmax(scores, dim=1)      # (B,T)
        ctx = (out * w.unsqueeze(-1)).sum(dim=1)
        logits = self.fc(self.drop(ctx))
        return logits, w


def _load():
    proc = os.path.join(base_dir, "data", "processed")
    train = pd.read_csv(os.path.join(proc, "nlp_train.csv"))
    test = pd.read_csv(os.path.join(proc, "nlp_test.csv"))
    return train, test


def train_svm(train, test):
    """Classical track with an internal validation model pick."""
    print("\n--- Track A: TF-IDF + classical classifiers ---")
    tr, val = train_test_split(train, test_size=0.10, random_state=SEED,
                               stratify=train["severity"])

    # build one shared vectoriser, fit on tr, transform val/test
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True,
                            max_features=200_000, tokenizer=tokenize)
    Xtr = tfidf.fit_transform(tr["text"])
    Xval = tfidf.transform(val["text"])
    ytr, yval = tr["severity"], val["severity"]

    from sklearn.linear_model import LogisticRegression
    lm = LogisticRegression(class_weight="balanced", solver="lbfgs",
                            random_state=SEED, max_iter=2000)
    lm.fit(Xtr, ytr)
    lv = f1_score(yval, lm.predict(Xval), average="macro", zero_division=0)
    print(f"  [val] LogReg macro-F1: {lv:.4f}")

    svm = LinearSVC(class_weight="balanced", random_state=SEED, max_iter=5000)
    svm.fit(Xtr, ytr)
    sv = f1_score(yval, svm.predict(Xval), average="macro", zero_division=0)
    print(f"  [val] LinearSVC macro-F1: {sv:.4f}")

    best = lm if lv >= sv else svm
    name = "LogisticRegression" if lv >= sv else "LinearSVC"
    best.fit(Xtr, ytr)  # refit on internal train

    clf = Pipeline([("tfidf", tfidf), ("clf", best)])
    clf.fit(train["text"], train["severity"])   # final fit on full train
    pred = clf.predict(test["text"])
    rep = classification_report(test["severity"], pred, output_dict=True, zero_division=0)
    macro = rep["macro avg"]["f1-score"]
    sev_recall = rep["SEVERE"]["recall"]
    print(f"\n[test] best classical ({name}) macro-F1: {macro:.4f}  "
          f"SEVERE recall: {sev_recall:.4f}")
    print(classification_report(test["severity"], pred, zero_division=0))
    joblib.dump(clf, os.path.join(base_dir, "models", "severity_stat.joblib"))
    return macro, sev_recall, name


def _make_tensors(df, vocab):
    X = np.array(encode(df["text"], vocab, MAX_LEN), dtype=np.int64)
    y = np.array([CLASSES.index(c) for c in df["severity"]], dtype=np.int64)
    return torch.from_numpy(X), torch.from_numpy(y)


def train_bilstm(train, test):
    print("\n--- Track B: BiLSTM + attention ---")
    tr, val = train_test_split(train, test_size=0.10, random_state=SEED,
                               stratify=train["severity"])
    vocab = build_vocab(tr["text"])
    X, y = _make_tensors(tr, vocab)
    Xv, yv = _make_tensors(val, vocab)
    Xt, yt = _make_tensors(test, vocab)
    mask = (X != 0).float()

    counts = tr["severity"].value_counts().to_dict()
    total = len(tr)
    weights = torch.tensor([total / (len(CLASSES) * counts[c]) for c in CLASSES],
                           dtype=torch.float32)
    print("Class weights (inverse frequency):", {c: round(float(w), 3)
                                                 for c, w in zip(CLASSES, weights)})

    model = BiLSTMSeverity(len(vocab))
    lossf = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH, shuffle=True)

    best_val = -1.0
    best_state = None
    patience, staleness = 3, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tot_loss, nbatch = 0.0, 0
        for xb, yb in loader:
            mb = (xb != 0).float()
            opt.zero_grad()
            logits, _ = model(xb, mb)
            loss = lossf(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_loss += loss.item()
            nbatch += 1
        model.eval()
        with torch.no_grad():
            lv, _ = model(Xv, (Xv != 0).float())
        vf1 = f1_score(yv.numpy(), lv.argmax(1).numpy(), average="macro", zero_division=0)
        print(f"  epoch {epoch:02d}  loss {tot_loss/nbatch:.4f}  val macro-F1 {vf1:.4f}")
        if vf1 > best_val:
            best_val, staleness = vf1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            staleness += 1
            if staleness >= patience:
                print("  early stop")
                break

    model.load_state_dict(best_state) if best_state else None
    model.eval()
    with torch.no_grad():
        logits, w = model(Xt, (Xt != 0).float())
    pred = logits.argmax(1).numpy()
    rep = classification_report(test["severity"].values,
                                np.array([CLASSES[i] for i in pred]),
                                output_dict=True, zero_division=0)
    macro = rep["macro avg"]["f1-score"]
    sev_recall = rep["SEVERE"]["recall"]
    print(classification_report(test["severity"].values,
                                np.array([CLASSES[i] for i in pred]),
                                zero_division=0))
    print(f"BiLSTM macro-F1: {macro:.4f}  SEVERE recall: {sev_recall:.4f}")

    torch.save(model.state_dict(), os.path.join(base_dir, "models", "bilstm_severity.pth"))
    with open(os.path.join(base_dir, "models", "bilstm_vocab.json"), "w") as f:
        json.dump({"vocab": vocab, "max_len": MAX_LEN,
                   "classes": CLASSES}, f)
    return macro, sev_recall, model


def main():
    print("=== Stage 03 NLP - DL Engineer ===")
    train, test = _load()
    print(f"Train {len(train)} / Test {len(test)}")
    svm_macro, svm_sev, svm_name = train_svm(train, test)
    lstm_macro, lstm_sev, _ = train_bilstm(train, test)

    winner = "severity_stat.joblib" if svm_macro >= lstm_macro else "bilstm_severity.pth"
    wname = f"{svm_name} (TF-IDF)" if winner.startswith("severity_stat") else "BiLSTM + attention"
    with open(os.path.join(base_dir, "models", "winner.txt"), "w", encoding="utf-8") as f:
        f.write(winner + "\n")

    report = f"""# Stage 03 NLP - Model Training Report

## Two-track evidence comparison (held-out test, n={len(test)})

| Track | Model | Macro-F1 | SEVERE recall |
| :--- | :--- | :---: | :---: |
| A (classical) | TF-IDF + {svm_name} | {svm_macro:.4f} | {svm_sev:.4f} |
| B (deep) | BiLSTM + attention | {lstm_macro:.4f} | {lstm_sev:.4f} |

## Decision
Winner: **{wname}** ({winner}) — selected purely on macro-F1 (SEVERE-safe).

> Philosophy: we do not ship the flashiest model by default. The deep track
> must clearly beat the simple, interpretable one; otherwise interpretability
> and speed win. This is baseline gating applied to NLP.
"""
    with open(os.path.join(base_dir, "reports", "model_training_nlp.md"), "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nWinner: {wname} -> {os.path.join(base_dir, 'models', winner)}")


if __name__ == "__main__":
    main()