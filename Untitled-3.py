#!/usr/bin/env python3
"""
predict_folder.py

Usage example:
    python predict_folder.py \
        --input_dir /path/to/text_files \
        --model_path ./out/best_model.pth \
        --output_dir ./preds \
        --emb_batch_size 16 \
        --max_length 256

This script supports plain .txt (one sentence per non-empty line),
.csv/.tsv (takes first column as sentence), .jsonl/.json (expects keys
"sentence" or "text"), or simple "sentence<TAB>label" style files (it will
ignore second column if present).
"""
import os
import glob
import argparse
import json
from typing import List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# -------------------------
# Defaults
# -------------------------
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -------------------------
# Helper: read sentences (prediction files)
# -------------------------

def read_sentences_file(path: str) -> List[str]:
    """
    Read sentences from a file. For CSV/TSV use the first column.
    For txt: read non-empty lines.
    For json/jsonl: look for "sentence" or "text" keys per line/object.
    Returns list of strings (possibly empty).
    """
    ext = os.path.splitext(path)[1].lower()
    sents: List[str] = []

    if ext in [".csv", ".tsv"]:
        sep = "," if ext == ".csv" else "\t"
        try:
            df = pd.read_csv(path, sep=sep, header=None, keep_default_na=False, dtype=str)
            if df.shape[1] >= 1:
                sents = df.iloc[:, 0].astype(str).str.strip().tolist()
                sents = [x for x in sents if x and x.lower() != "nan"]
                return sents
        except Exception:
            pass

    if ext in [".jsonl", ".json"]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                # try jsonl first: each line a JSON
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        j = json.loads(line)
                        if isinstance(j, dict):
                            if "sentence" in j:
                                sents.append(str(j["sentence"]).strip())
                            elif "text" in j:
                                sents.append(str(j["text"]).strip())
                            else:
                                # If only one string value present: grab it
                                vals = [v for v in j.values() if isinstance(v, str)]
                                if vals:
                                    sents.append(vals[0].strip())
                    except json.JSONDecodeError:
                        # last-resort treat as plain line
                        sents.append(line)
            sents = [x for x in sents if x]
            if sents:
                return sents
        except Exception:
            pass

    # Plain text fallback
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # If input contains a tab or ' ||| ' or a comma with single comma, take first part
                if "\t" in line:
                    s = line.split("\t", 1)[0]
                elif " ||| " in line:
                    s = line.split(" ||| ", 1)[0]
                elif "," in line and line.count(",") == 1:
                    s = line.split(",", 1)[0]
                else:
                    s = line
                s = s.strip()
                if s:
                    sents.append(s)
        sents = [x for x in sents if x]
        return sents
    except Exception:
        pass

    return []


# -------------------------
# Load BERT and embedder
# -------------------------
print(f"[INFO] Loading tokenizer and BERT model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert_model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
bert_model.eval()

def get_sentence_embeddings(sentences: List[str], batch_size: int = 16, max_length: int = 256) -> torch.Tensor:
    """
    Compute sentence vectors by mean-pooling the last hidden states weighted by attention mask.
    Returns tensor of shape [len(sentences), hidden_size], on DEVICE.
    """
    if len(sentences) == 0:
        H = bert_model.config.hidden_size
        return torch.zeros((0, H), device=DEVICE)

    all_vecs = []
    H = bert_model.config.hidden_size
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i: i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            outputs = bert_model(**enc)
        last_hidden = outputs.last_hidden_state    # [B, T, H]
        attn = enc["attention_mask"].unsqueeze(-1)  # [B, T, 1]
        summed = (last_hidden * attn).sum(dim=1)  # [B, H]
        lengths = attn.sum(dim=1).clamp(min=1)    # [B, 1]
        sent_vecs = summed / lengths              # [B, H]
        all_vecs.append(sent_vecs)
    return torch.cat(all_vecs, dim=0)


def docs_sentlists_to_batch(sent_lists: List[List[str]], emb_batch_size: int = 16, max_length: int = 256):
    """
    Given a list of documents (each a list of sentences),
    produce batch_embeddings: [B, L_max, H] and mask [B,L_max] (bool).
    """
    flattened = [s for doc in sent_lists for s in doc]
    embeddings = get_sentence_embeddings(flattened, batch_size=emb_batch_size, max_length=max_length)  # [N, H]
    H = embeddings.shape[1]
    per_doc_vecs = []
    idx = 0
    for doc in sent_lists:
        n = len(doc)
        per_doc_vecs.append(embeddings[idx: idx + n])
        idx += n

    batch_size = len(per_doc_vecs)
    max_len = max((v.shape[0] for v in per_doc_vecs), default=0)
    if max_len == 0:
        return torch.zeros((0, 0, H), device=DEVICE), torch.zeros((0, 0), dtype=torch.bool, device=DEVICE)
    batch_embeddings = torch.zeros((batch_size, max_len, H), device=DEVICE)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=DEVICE)
    for i, vecs in enumerate(per_doc_vecs):
        n = vecs.shape[0]
        if n == 0:
            continue
        batch_embeddings[i, :n, :] = vecs
        mask[i, :n] = True
    return batch_embeddings, mask


# -------------------------
# Model class (must match training)
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
        outputs, _ = self.sentence_encoder(sentence_vectors)  # [B, L, hidden_dim]
        logits = self.hidden2tag(outputs)  # [B, L, num_labels]

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            B, L, C = logits.shape
            loss = loss_fct(logits.view(-1, C), labels.view(-1))
            return loss
        else:
            preds = logits.argmax(dim=-1)  # [B, L]
            if mask is not None:
                out = []
                for b in range(preds.size(0)):
                    valid = preds[b][mask[b]].tolist()
                    out.append(valid)
                return out
            else:
                return [preds[b].tolist() for b in range(preds.size(0))]


# -------------------------
# Prediction pipeline
# -------------------------
def predict_folder(input_dir: str, model_path: str, output_dir: str, emb_batch_size: int = 16, max_length: int = 256):
    # 1) load saved model file (state + maps + args)
    saved = torch.load(model_path, map_location="cpu")
    # Extract label maps
    if "label2id" in saved:
        label2id = saved["label2id"]
    else:
        raise ValueError("Saved model doesn't contain 'label2id' mapping.")
    id2label = {int(v): k for k, v in label2id.items()}
    num_labels = len(label2id)

    # Try to recover hidden_dim from saved args if present
    saved_args = saved.get("args", {}) if isinstance(saved.get("args", {}), dict) else {}
    hidden_dim = int(saved_args.get("hidden_dim", 512))

    # 2) init model and load weights
    embed_dim = bert_model.config.hidden_size
    model = HierBiLSTMCE(embed_dim=embed_dim, hidden_dim=hidden_dim, num_labels=num_labels).to(DEVICE)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()

    # 3) gather input files (common text-like extensions)
    patterns = ["*.txt", "*.csv", "*.tsv", "*.jsonl", "*.json"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(input_dir, p)))
    files = sorted(files)
    if not files:
        raise ValueError(f"No files found in {input_dir} with patterns {patterns}")

    os.makedirs(output_dir, exist_ok=True)

    summary = []
    for f in files:
        sents = read_sentences_file(f)
        if len(sents) == 0:
            print(f"[WARN] No sentences found in {f}; skipping.")
            continue

        # model expects a batch of documents; we'll process one document per file
        batch_embeddings, mask = docs_sentlists_to_batch([sents], emb_batch_size=emb_batch_size, max_length=max_length)
        if batch_embeddings.numel() == 0:
            print(f"[WARN] No embeddings produced for {f}; skipping.")
            continue

        with torch.no_grad():
            preds_batch = model(batch_embeddings, labels=None, mask=mask)  # list[list[int]] with one element
        preds = preds_batch[0]  # list[int] of length == len(sents)

        # map ids -> label strings
        mapped_labels = [id2label[int(pid)] if int(pid) in id2label else str(pid) for pid in preds]

        # create dataframe and save
        df = pd.DataFrame({"sentence": sents, "label": mapped_labels})
        base = os.path.splitext(os.path.basename(f))[0]
        out_path = os.path.join(output_dir, base + ".pred.csv")
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"[INFO] Predicted {len(sents)} sentences for {os.path.basename(f)} -> {out_path}")
        summary.append((f, out_path, len(sents)))

    print("Done. Summary:")
    for orig, outp, n in summary:
        print(f"  {os.path.basename(orig)} -> {os.path.basename(outp)} ({n} sentences)")


# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Predict sentence labels for all files in a folder using best_model.pth")
    ap.add_argument("--input_dir", required=True, help="Folder containing files to predict (txt/csv/tsv/jsonl/json).")
    ap.add_argument("--model_path", required=True, help="Path to best_model.pth saved by the training script.")
    ap.add_argument("--output_dir", required=True, help="Where to write per-file CSV predictions.")
    ap.add_argument("--emb_batch_size", type=int, default=16, help="BERT embedder batch size (sentences at a time).")
    ap.add_argument("--max_length", type=int, default=256, help="Max tokens per sentence for BERT tokenizer.")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(f"[INFO] Device: {DEVICE}, input_dir: {args.input_dir}, model: {args.model_path}, out: {args.output_dir}")
    predict_folder(args.input_dir, args.model_path, args.output_dir, emb_batch_size=args.emb_batch_size, max_length=args.max_length)
