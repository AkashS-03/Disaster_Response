import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter
import json
import re

# --- Configuration ---
MAX_LEN = 30
VOCAB_SIZE = 1000
EMBED_DIM = 64
HIDDEN_DIM = 32
BATCH_SIZE = 64
EPOCHS = 10

class TranscriptGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(TranscriptGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        x = self.embedding(x)
        _, h = self.gru(x)
        out = self.fc(h[-1])
        return out

def tokenize(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.split()

def train_nlp_model():
    print("--- Starting NLP Deep Learning Pipeline ---")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(base_dir, "data", "nlp", "synthetic_transcripts.csv")
    
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records.")
    
    # Label encoding
    label_map = {"LOW": 0, "MODERATE": 1, "SEVERE": 2}
    df['label'] = df['severity'].map(label_map)
    
    # Build vocabulary
    all_words = []
    for text in df['transcript']:
        all_words.extend(tokenize(text))
        
    word_counts = Counter(all_words)
    common_words = [word for word, count in word_counts.most_common(VOCAB_SIZE - 2)]
    
    word2idx = {word: idx + 2 for idx, word in enumerate(common_words)}
    word2idx["<PAD>"] = 0
    word2idx["<UNK>"] = 1
    
    # Save vocab
    with open(os.path.join(base_dir, "models", "vocab.json"), "w") as f:
        json.dump(word2idx, f)
        
    # Convert text to sequences
    X = []
    for text in df['transcript']:
        seq = [word2idx.get(w, 1) for w in tokenize(text)]
        if len(seq) < MAX_LEN:
            seq = seq + [0] * (MAX_LEN - len(seq))
        else:
            seq = seq[:MAX_LEN]
        X.append(seq)
        
    X = np.array(X)
    y = df['label'].values
    
    # HELD-OUT TEST SPLIT (Rigorous Evaluation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.long), torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TranscriptGRU(len(word2idx), EMBED_DIM, HIDDEN_DIM, 3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("Training GRU Model...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = model(batch_X)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
    # Evaluation on Held-Out Test Set
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(device)
            out = model(batch_X)
            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            
    acc = accuracy_score(y_test, all_preds)
    f1 = f1_score(y_test, all_preds, average='weighted')
    
    print(f"\n--- NLP EVALUATION (Held-Out Test Set) ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score (Weighted): {f1:.4f}")
    
    # Save Model
    torch.save(model.state_dict(), os.path.join(base_dir, "models", "nlp_classifier.pth"))
    
    # Save Report
    with open(os.path.join(base_dir, "reports", "nlp_evaluation_report.md"), "w") as f:
        f.write("# NLP Severity Classifier Evaluation\n\n")
        f.write("## Transparency Notice\n")
        f.write("The dataset used for this model was synthetically generated using rule-based keywords. ")
        f.write("The high accuracy and F1 scores below reflect the GRU's ability to perfectly memorize ")
        f.write("and map these explicit rules, rather than representing organic human distress language understanding.\n\n")
        f.write("## Metrics (Held-Out Test Set: 20%)\n")
        f.write(f"- **Accuracy:** {acc:.4f}\n")
        f.write(f"- **F1 Score:** {f1:.4f}\n")
        
    print("NLP Pipeline Complete. Report saved.")

if __name__ == "__main__":
    train_nlp_model()
