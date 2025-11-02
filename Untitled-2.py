#!/usr/bin/env python3
"""
train_legal_roles_ce.py

Two-stage workflow (recommended for Colab / speed):
  1) Precompute Legal-BERT sentence embeddings for all documents and save to disk.
  2) Train the BiLSTM + CrossEntropy classifier using the saved embeddings (no BERT during training).

Usage examples:
  # Precompute embeddings (writes embeddings.pkl in --output_dir)
  python train_legal_roles_ce.py --mode precompute --data_dir /path/to/folder --output_dir /path/to/out --emb_batch_size 16

  # Train using precomputed embeddings
  python train_legal_roles_ce.py --mode train --emb_file /path/to/out/embeddings.pkl --output_dir /path/to/out --epochs 3 --batch_size 4

  # Do both (precompute then train)
  python train_legal_roles_ce.py --mode all --data_dir /path/to/folder --output_dir /path/to/out

Notes:
- The script supports CSV/TSV/JSONL/plain text inputs (same heuristics as before).
- Precompute is the slow step (one pass through BERT). Training after that is fast.

"""

import os
import glob
import random
import argparse
import json
import pickle
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# -------------------------
# Settings & Device
# -------------------------
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Utilities: read files with heuristics
# -------------------------

def read_label_file(path: str) -> Tuple[List[str], List[str]]:
    ext = os.path.splitext(path)[1].lower()
    sentences: List[str] = []
    labels: List[str] = []

    # CSV/TSV
    if ext in [".csv", ".tsv"]:
        sep = "," if ext == ".csv" else "	"
        try:
            df = pd.read_csv(path, sep=sep, keep_default_na=False)
            cols = [c.lower() for c in df.columns]
            if "sentence" in cols and "label" in cols:
                s_col = df.columns[cols.index("sentence")]
                l_col = df.columns[cols.index("label")]
                sentences = df[s_col].astype(str).tolist()
                labels = df[l_col].astype(str).tolist()
                return sentences, labels
            else:
                if df.shape[1] >= 2:
                    sentences = df.iloc[:, 0].astype(str).tolist()
                    labels = df.iloc[:, 1].astype(str).tolist()
                    return sentences, labels
        except Exception:
            pass

    # JSONL/JSON
    if ext in [".json", ".jsonl"]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    j = json.loads(line)
                    if isinstance(j, dict):
                        if "sentence" in j and "label" in j:
                            sentences.append(str(j["sentence"]))
                            labels.append(str(j["label"]))
                        elif "text" in j and "label" in j:
                            sentences.append(str(j["text"]))
                            labels.append(str(j["label"]))
            if sentences:
                return sentences, labels
        except Exception:
            pass

    # Plain text
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if "	" in line:
                    s, l = line.split("	", 1)
                elif " ||| " in line:
                    s, l = line.split(" ||| ", 1)
                elif "," in line and line.count(",") == 1:
                    s, l = line.split(",", 1)
                else:
                    continue
                sentences.append(s.strip())
                labels.append(l.strip())
        if sentences:
            return sentences, labels
    except Exception:
        pass

    raise ValueError(f"Could not parse file {path} - unsupported or unexpected format.")


# -------------------------
# Embedding precomputation
# -------------------------

def precompute_embeddings(data_dir: str, output_dir: str, model_name: str = MODEL_NAME, emb_batch_size: int = 16, max_length: int = 256):
    print(f"[INFO] Loading tokenizer and BERT model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_model = AutoModel.from_pretrained(model_name).to(device)
    bert_model.eval()

    supported = ["*.csv", "*.tsv", "*.txt", "*.jsonl", "*.json"]
    files = []
    for p in supported:
        files.extend(glob.glob(os.path.join(data_dir, p)))
    files = sorted(files)
    if not files:
        raise ValueError(f"No files found in {data_dir}")

    docs = []
    all_sentences = []
    for f in files:
        sents, labs = read_label_file(f)
        docs.append({"doc_id": os.path.basename(f), "sentences": sents, "labels": labs})
        all_sentences.extend(sents)

    print(f"[INFO] Found {len(docs)} documents and {len(all_sentences)} sentences total.")

    # compute embeddings in batches
    all_embs = []
    for i in tqdm(range(0, len(all_sentences), emb_batch_size), desc="BERT embed batches"):
        batch = all_sentences[i: i + emb_batch_size]
        enc = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = bert_model(**enc)
        last_hidden = out.last_hidden_state
        attn = enc["attention_mask"].unsqueeze(-1)
        summed = (last_hidden * attn).sum(dim=1)
        lengths = attn.sum(dim=1).clamp(min=1)
        sent_vecs = (summed / lengths).cpu()  # move to CPU
        all_embs.append(sent_vecs)

    all_embs = torch.cat(all_embs, dim=0)  # [N, H]

    # Map embeddings back to docs
    idx = 0
    for d in docs:
        n = len(d["sentences"])
        d["embeddings"] = all_embs[idx: idx + n].numpy()
        idx += n

    os.makedirs(output_dir, exist_ok=True)
    emb_path = os.path.join(output_dir, "embeddings.pkl")
    with open(emb_path, "wb") as f:
        pickle.dump(docs, f)

    print(f"[INFO] Saved embeddings to {emb_path}")
    return emb_path


# -------------------------
# Model: HierBiLSTM + CrossEntropy (same as before)
# -------------------------
class HierBiLSTMCE(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, num_labels: int, bidirectional: bool = True):
        super().__init__()
        per_dir = hidden_dim // (2 if bidirectional else 1)
        self.sentence_encoder = nn.LSTM(
            embed_dim, per_dir,
            bidirectional=bidirectional, batch_first=True
        )
        self.hidden2tag = nn.Linear(hidden_dim, num_labels)

    def forward(self, sentence_vectors: torch.Tensor, labels: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None):
        outputs, _ = self.sentence_encoder(sentence_vectors)
        logits = self.hidden2tag(outputs)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            B, L, C = logits.shape
            loss = loss_fct(logits.view(-1, C), labels.view(-1))
            return loss
        else:
            preds = logits.argmax(dim=-1)
            if mask is not None:
                out = []
                for b in range(preds.size(0)):
                    valid = preds[b][mask[b]].tolist()
                    out.append(valid)
                return out
            else:
                return [preds[b].tolist() for b in range(preds.size(0))]


# -------------------------
# Training using precomputed embeddings
# -------------------------

def train_on_embeddings(emb_file: str, output_dir: str, epochs: int = 3, batch_size: int = 4, lr: float = 1e-3, hidden_dim: int = 512, val_frac: float = 0.2):
    with open(emb_file, "rb") as f:
        docs = pickle.load(f)

    # Build label vocab
    unique_labels = sorted({lbl for d in docs for lbl in d["labels"]})
    label2id = {lab: i for i, lab in enumerate(unique_labels)}
    id2label = {v: k for k, v in label2id.items()}
    num_labels = len(label2id)
    print(f"[INFO] {len(docs)} docs, {num_labels} labels: {label2id}")

    # Convert labels
    for d in docs:
        d["label_ids"] = [label2id[l] for l in d["labels"]]

    # Train/val split
    random.shuffle(docs)
    n_train = int(len(docs) * (1.0 - val_frac))
    train_docs = docs[:n_train]
    val_docs = docs[n_train:]
    print(f"Train docs: {len(train_docs)} | Val docs: {len(val_docs)}")

    embed_dim = train_docs[0]["embeddings"].shape[1]
    model = HierBiLSTMCE(embed_dim=embed_dim, hidden_dim=hidden_dim, num_labels=num_labels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def batchify_docs(list_docs, bsz):
        for i in range(0, len(list_docs), bsz):
            yield list_docs[i:i+bsz]

    best_val_loss = float("inf")
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        print(f"=== Epoch {epoch}/{epochs} ===")
        for batch_docs in tqdm(list(batchify_docs(train_docs, batch_size)), desc="train batches"):
            # prepare tensors
            emb_tensors = [torch.tensor(d["embeddings"], dtype=torch.float32) for d in batch_docs]
            label_tensors = [torch.tensor(d["label_ids"], dtype=torch.long) for d in batch_docs]

            padded_emb = nn.utils.rnn.pad_sequence(emb_tensors, batch_first=True).to(device)  # [B, L, H]
            padded_labels = nn.utils.rnn.pad_sequence(label_tensors, batch_first=True, padding_value=-100).to(device)  # [B, L]
            mask = (padded_labels != -100)

            loss = model(padded_emb, labels=padded_labels, mask=mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_loss / max(1, n_batches)
        print(f"Epoch {epoch} - avg train loss: {avg_train_loss:.4f}")

        # validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            v_batches = 0
            for batch_docs in tqdm(list(batchify_docs(val_docs, batch_size)), desc="val batches"):
                emb_tensors = [torch.tensor(d["embeddings"], dtype=torch.float32) for d in batch_docs]
                label_tensors = [torch.tensor(d["label_ids"], dtype=torch.long) for d in batch_docs]
                padded_emb = nn.utils.rnn.pad_sequence(emb_tensors, batch_first=True).to(device)
                padded_labels = nn.utils.rnn.pad_sequence(label_tensors, batch_first=True, padding_value=-100).to(device)
                mask = (padded_labels != -100)

                loss = model(padded_emb, labels=padded_labels, mask=mask)
                val_loss += loss.item()
                v_batches += 1

            avg_val_loss = val_loss / max(1, v_batches)
            print(f"Epoch {epoch} - val loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                os.makedirs(output_dir, exist_ok=True)
                save_path = os.path.join(output_dir, "best_model.pth")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "label2id": label2id,
                    "id2label": id2label,
                    "args": {"hidden_dim": hidden_dim}
                }, save_path)
                print(f"[INFO] Best model saved to {save_path}")

    print("Training finished.")


# -------------------------
# Argparse & main
# -------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Two-stage training: precompute Legal-BERT embeddings, then train BiLSTM on them.")
    ap.add_argument("--mode", choices=["precompute", "train", "all"], default="all",
                    help="Mode: 'precompute' to compute embeddings, 'train' to train from embeddings, 'all' to do both.")
    ap.add_argument("--data_dir", help="Folder with labeled files (required for precompute mode)")
    ap.add_argument("--emb_file", help="Path to embeddings.pkl (required for train mode)")
    ap.add_argument("--output_dir", default="./out", help="Where to save embeddings & model")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=4, help="Number of documents per batch.")
    ap.add_argument("--emb_batch_size", type=int, default=16, help="BERT embedder batch size (sentences at a time).")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--max_length", type=int, default=256, help="Max tokens per sentence for BERT tokenizer")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode in ("precompute", "all"):
        if not args.data_dir:
            raise ValueError("--data_dir is required for precompute mode")
        emb_path = precompute_embeddings(args.data_dir, args.output_dir, emb_batch_size=args.emb_batch_size, model_name=MODEL_NAME, max_length=args.max_length)
    else:
        emb_path = args.emb_file

    if args.mode in ("train", "all"):
        if not emb_path:
            raise ValueError("--emb_file (or run precompute first) is required for train mode")
        train_on_embeddings(emb_path, args.output_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, hidden_dim=args.hidden_dim, val_frac=args.val_frac)

    print("Done.")
