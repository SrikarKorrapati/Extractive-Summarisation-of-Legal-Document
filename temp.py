#!/usr/bin/env python3
"""
train_legal_roles.py

Usage:
    python train_legal_roles.py --data_dir /path/to/folder_with_files --output_dir ./out --epochs 5 --batch_size 4

Assumptions about files in data_dir:
- Each file contains sentence-level labels.
- Supported formats: .csv (with columns 'sentence' and 'label' or first two cols),
                    .tsv, .txt, .jsonl (with keys 'sentence' and 'label'),
                    plain text where each line is "sentence<TAB>label" or "sentence ||| label".
- The script will try to detect the format heuristically.
"""

import os
import glob
import random
import argparse
import json
from typing import List, Tuple, Dict, Any

import pandas as pd
import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# -------------------------
# Settings & Device
# -------------------------
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"  # same as your code
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Utilities: read files with heuristics
# -------------------------
def read_label_file(path: str) -> Tuple[List[str], List[str]]:
    """
    Return: sentences (list[str]), labels (list[str])
    Tries multiple formats:
      - CSV with columns 'sentence' and 'label'
      - CSV/TSV with first two columns = sentence,label
      - JSONL with keys 'sentence' and 'label'
      - plain text lines: sentence<TAB>label, sentence ||| label
    """
    ext = os.path.splitext(path)[1].lower()
    sentences = []
    labels = []

    # 1) CSV/TSV
    if ext in [".csv", ".tsv"]:
        sep = "," if ext == ".csv" else "\t"
        try:
            df = pd.read_csv(path, sep=sep, keep_default_na=False)
            # try to find column names
            cols = [c.lower() for c in df.columns]
            if "sentence" in cols and "label" in cols:
                s_col = df.columns[cols.index("sentence")]
                l_col = df.columns[cols.index("label")]
                sentences = df[s_col].astype(str).tolist()
                labels = df[l_col].astype(str).tolist()
                return sentences, labels
            else:
                # fallback: first two columns
                if df.shape[1] >= 2:
                    sentences = df.iloc[:, 0].astype(str).tolist()
                    labels = df.iloc[:, 1].astype(str).tolist()
                    return sentences, labels
        except Exception:
            pass

    # 2) JSONL
    if ext == ".json" or ext == ".jsonl":
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    j = json.loads(line)
                    # keys heuristic
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

    # 3) Plain text fallback
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # common separators
                if "\t" in line:
                    s, l = line.split("\t", 1)
                elif " ||| " in line:
                    s, l = line.split(" ||| ", 1)
                elif " ||| " in line:
                    s, l = line.split(" ||| ", 1)
                elif " , " in line and line.count(",") == 1:
                    s, l = line.split(",", 1)
                else:
                    # If just a single column per line (maybe sentence only),
                    # we cannot train; so skip or raise.
                    # To be conservative, skip such lines.
                    continue
                sentences.append(s.strip())
                labels.append(l.strip())
        if sentences:
            return sentences, labels
    except Exception:
        pass

    raise ValueError(f"Could not parse file {path} - unsupported or unexpected format.")


# -------------------------
# Legal-BERT embedding utilities
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert_model = AutoModel.from_pretrained(MODEL_NAME).to(device)
bert_model.eval()

def get_sentence_embeddings(sentences: List[str], batch_size: int = 16) -> torch.Tensor:
    """
    sentences: List[str]
    Returns: tensor [len(sentences), hidden_size] on device
    """
    all_vecs = []
    H = bert_model.config.hidden_size
    if len(sentences) == 0:
        return torch.zeros((0, H), device=device)

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        ).to(device)

        with torch.no_grad():
            outputs = bert_model(**enc)
        last_hidden = outputs.last_hidden_state  # [B, T, H]
        attn = enc["attention_mask"].unsqueeze(-1)  # [B, T, 1]
        summed = (last_hidden * attn).sum(dim=1)  # [B, H]
        lengths = attn.sum(dim=1).clamp(min=1)    # [B, 1]
        sent_vecs = summed / lengths              # [B, H]
        all_vecs.append(sent_vecs)

    return torch.cat(all_vecs, dim=0)


def docs_sentlists_to_batch(sent_lists: List[List[str]],
                            label_lists: List[List[int]] = None):
    """
    Given a list of documents, each as a list of sentences, produce:
      - batch_embeddings: [B, L_max, H]
      - mask: [B, L_max] bool (True where sentence exists)
      - padded_labels: [B, L_max] long (if label_lists provided)
      - sentences_per_doc: same as sent_lists (returned for convenience)
    """
    # Flatten sentences to compute embeddings in one go (faster)
    flattened = [s for doc in sent_lists for s in doc]
    embeddings = get_sentence_embeddings(flattened)  # [N, H]
    H = embeddings.shape[1]

    # Reconstruct per-doc tensors
    per_doc_vecs = []
    idx = 0
    for doc in sent_lists:
        n = len(doc)
        per_doc_vecs.append(embeddings[idx: idx + n])
        idx += n

    batch_size = len(per_doc_vecs)
    max_len = max((v.shape[0] for v in per_doc_vecs), default=0)

    batch_embeddings = torch.zeros((batch_size, max_len, H), device=device)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
    padded_labels = None
    if label_lists is not None:
        padded_labels = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)

    for i, vecs in enumerate(per_doc_vecs):
        n = vecs.shape[0]
        if n == 0:
            continue
        batch_embeddings[i, :n, :] = vecs
        mask[i, :n] = True
        if label_lists is not None:
            labs = label_lists[i]
            # Pad with 0 - but mask will prevent loss from using padded positions.
            padded = list(labs)[:n] + [0] * max(0, n - len(labs))
            padded_labels[i, :n] = torch.tensor(padded, dtype=torch.long, device=device)

    return batch_embeddings, mask, padded_labels, sent_lists


# -------------------------
# Model: HierBiLSTMCRF
# -------------------------
class HierBiLSTMCRF(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, num_labels: int, bidirectional: bool = True):
        super().__init__()
        self.sentence_encoder = nn.LSTM(
            embed_dim, hidden_dim // (2 if bidirectional else 1),
            bidirectional=bidirectional, batch_first=True
        )
        self.hidden2tag = nn.Linear(hidden_dim, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, sentence_vectors: torch.Tensor, labels: torch.Tensor = None, mask: torch.Tensor = None):
        """
        sentence_vectors: [B, L, embed_dim]
        labels: [B, L] (long) or None
        mask: [B, L] (bool) or None
        Returns:
          - if labels provided: scalar loss (negative log-likelihood)
          - else: list of decoded sequences (list[list[int]])
        """
        outputs, _ = self.sentence_encoder(sentence_vectors)  # [B, L, hidden_dim]
        emissions = self.hidden2tag(outputs)  # [B, L, num_labels]

        if labels is not None:
            if mask is None:
                mask = torch.ones(labels.shape, dtype=torch.bool, device=labels.device)
            # CRF expects (emissions, tags, mask) and returns log-likelihood
            log_likelihood = self.crf(emissions, labels, mask=mask, reduction="mean")
            return -log_likelihood
        else:
            if mask is None:
                mask = torch.ones(emissions.shape[:2], dtype=torch.bool, device=emissions.device)
            preds = self.crf.decode(emissions, mask=mask)
            return preds


# -------------------------
# Dataset loader over documents
# -------------------------
def load_all_docs_from_folder(folder: str, ext_filter: List[str] = None):
    """
    Walk folder and read all files with supported extensions.
    Returns lists: sent_lists (List[List[str]]), label_lists (List[List[str]]), doc_ids (List[str])
    """
    supported = ["*.csv", "*.tsv", "*.txt", "*.jsonl", "*.json"]
    patterns = ext_filter if ext_filter else supported
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(folder, p)))

    if not files:
        raise ValueError(f"No files found in {folder} matching {patterns}")

    sent_lists = []
    label_lists = []
    doc_ids = []
    for f in sorted(files):
        try:
            sents, labs = read_label_file(f)
            if len(sents) == 0:
                print(f"[WARN] skipping empty file {f}")
                continue
            sent_lists.append(sents)
            label_lists.append(labs)
            doc_ids.append(os.path.basename(f))
        except Exception as e:
            print(f"[ERROR] Could not read {f}: {e}")

    return sent_lists, label_lists, doc_ids


# -------------------------
# Training + evaluation pipeline
# -------------------------
def train(args):
    # 1) load data
    sent_lists, label_lists_str, doc_ids = load_all_docs_from_folder(args.data_dir)
    print(f"Loaded {len(sent_lists)} documents")

    # 2) Build label vocab
    unique_labels = sorted({lbl for doc in label_lists_str for lbl in doc})
    label2id = {lab: i for i, lab in enumerate(unique_labels)}
    id2label = {i: l for l, i in label2id.items()}
    num_labels = len(label2id)
    print(f"Found {num_labels} labels: {label2id}")

    # 3) Convert labels to ids
    label_lists = [[label2id[l] for l in doc] for doc in label_lists_str]

    # 4) Train/val split (by documents)
    combined = list(zip(sent_lists, label_lists, doc_ids))
    random.shuffle(combined)
    n_train = int(len(combined) * (1.0 - args.val_frac))
    train_comb = combined[:n_train]
    val_comb = combined[n_train:]

    train_sents = [c[0] for c in train_comb]
    train_labels = [c[1] for c in train_comb]

    val_sents = [c[0] for c in val_comb]
    val_labels = [c[1] for c in val_comb]

    print(f"Train docs: {len(train_sents)} | Val docs: {len(val_sents)}")

    # 5) Optionally freeze BERT (only tune BiLSTM+CRF)
    if args.freeze_bert:
        bert_model.eval()
        for p in bert_model.parameters():
            p.requires_grad = False
        print("[INFO] BERT frozen. Only training BiLSTM+CRF.")
    else:
        # keep BERT in eval if you don't want to fine-tune it, but embeddings are computed with torch.no_grad() above.
        print("[INFO] BERT will be used for embedding extraction (no gradient by design here).")

    # 6) Model init
    model = HierBiLSTMCRF(embed_dim=bert_model.config.hidden_size, hidden_dim=args.hidden_dim, num_labels=num_labels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 7) Training loop with simple batching by documents
    def batchify(doc_sents, doc_labels, batch_size):
        # yields batches of lists-of-sentences and lists-of-labels
        indices = list(range(len(doc_sents)))
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            sents = [doc_sents[j] for j in batch_idx]
            labs = [doc_labels[j] for j in batch_idx]
            yield sents, labs

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        for sents_batch, labs_batch in tqdm(list(batchify(train_sents, train_labels, args.batch_size)),
                                           desc="train batches"):
            # Build batch embeddings (this computes BERT embeddings under the hood).
            batch_embeddings, mask, padded_labels, _ = docs_sentlists_to_batch(sents_batch, labs_batch)
            # forward
            loss = model(batch_embeddings, labels=padded_labels, mask=mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() if isinstance(loss, torch.Tensor) else float(loss)
            n_batches += 1

        avg_train_loss = total_loss / max(1, n_batches)
        print(f"Epoch {epoch} - avg train loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            v_batches = 0
            for sents_batch, labs_batch in tqdm(list(batchify(val_sents, val_labels, args.batch_size)),
                                               desc="val batches"):
                batch_embeddings, mask, padded_labels, _ = docs_sentlists_to_batch(sents_batch, labs_batch)
                loss = model(batch_embeddings, labels=padded_labels, mask=mask)
                val_loss += (loss.item() if isinstance(loss, torch.Tensor) else float(loss))
                v_batches += 1
            avg_val_loss = val_loss / max(1, v_batches)
            print(f"Epoch {epoch} - val loss: {avg_val_loss:.4f}")

            # Save best
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                os.makedirs(args.output_dir, exist_ok=True)
                save_path = os.path.join(args.output_dir, "best_model.pth")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "label2id": label2id,
                    "id2label": id2label,
                    "args": vars(args)
                }, save_path)
                print(f"[INFO] Best model saved to {save_path}")

    print("Training finished.")


# -------------------------
# Argparse & main
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Folder with labeled files (50 files).")
    ap.add_argument("--output_dir", default="./out", help="Where to save model & maps.")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("1", type=int, default=4, help="Number of documents per batch.")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--freeze_bert", action="store_true", help="If set, freeze BERT (don't finetune it).")
    ap.add_argument("--exts", nargs="*", default=None, help="Optional list of filename patterns to read (e.g. '*.csv')")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
