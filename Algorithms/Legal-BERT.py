import os
import math
import numpy as np
import pandas as pd
import nltk
from nltk import sent_tokenize
from nltk.corpus import stopwords
from rouge_score import rouge_scorer

# --- Legal-BERT (Indian legal domain) ---
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise ImportError("Please install sentence-transformers: pip install sentence-transformers") from e

# Ensure NLTK data is present
nltk.download("punkt")
nltk.download("stopwords")

# ----------------
# Config
# ----------------
INPUT_DIR = "Judgement"        # folder with original documents
GOLD_DIR = "Summary"           # folder with gold/reference summaries (same filenames)
OUTPUT_CSV = "rouge_scores.csv"
THRESHOLD = 0.2                # tuned for LegalBERT embeddings
EPSILON = 1e-6
SELECT_RATIO = 0.3              # fraction of sentences to include in summary
LEGAL_BERT_MODEL_NAME = "law-ai/InLegalBERT"

STOP_WORDS = set(stopwords.words("english"))
SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

# initialize Legal-BERT once (global)
LEGAL_BERT_MODEL = SentenceTransformer(LEGAL_BERT_MODEL_NAME)


# ----------------
# Summarizer
# ----------------
def lexrank_summarize(text, select_ratio=0.4, threshold=0.2, epsilon=1e-6, per_row_k_small=1):
    """
    LegalBERT-based LexRank (vectorized, adaptive for short docs).
    - threshold: absolute cosine cutoff (used when ts > 5)
    - per_row_k_small: for ts <= 5 keep each sentence's top-k neighbors
    - epsilon: PageRank convergence tolerance (smaller -> more accurate)
    """
    sentences = sent_tokenize(text)
    ts = len(sentences)
    if ts == 0:
        return ""
    if ts == 1:
        return sentences[0]

    # --- encode embeddings ---
    try:
        embeddings = LEGAL_BERT_MODEL.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
    except TypeError:
        embeddings = LEGAL_BERT_MODEL.encode(sentences, convert_to_numpy=True)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        embeddings = embeddings / norms

    # Cosine similarity
    sims = np.dot(embeddings, embeddings.T)
    np.fill_diagonal(sims, 0.0)  # ignore self

    # Build adjacency
    adj = np.zeros((ts, ts), dtype=float)
    if ts <= 5:
        k = min(per_row_k_small, ts - 1)
        for i in range(ts):
            inds = np.argsort(-sims[i])  # descending
            topk = inds[:k]
            adj[i, topk] = 1.0
    else:
        adj = (sims > threshold).astype(float)

    # Handle isolated nodes (add self-loop)
    row_sums = adj.sum(axis=1)
    isolated = (row_sums == 0)
    if isolated.any():
        adj[isolated, :] = 0.0
        idxs = np.where(isolated)[0]
        for i in idxs:
            adj[i, i] = 1.0
        row_sums = adj.sum(axis=1)

    # Normalize rows
    matrix = adj / row_sums[:, None]

    # PageRank
    damping = 0.85
    p_vector = np.ones(ts, dtype=float) / ts
    transposed = matrix.T
    delta = 1.0
    while delta > epsilon:
        next_p = (1 - damping) / ts + damping * (transposed @ p_vector)
        delta = np.linalg.norm(next_p - p_vector)
        p_vector = next_p

    # Select top-k sentences
    k = max(1, int(math.ceil(ts * select_ratio)))
    ranked_indices = np.argsort(-p_vector, kind='mergesort')
    selected_indices = sorted(ranked_indices[:k].tolist())

    summary = " ".join([sentences[i] for i in selected_indices])
    return summary


# ----------------
# Evaluation
# ----------------
def evaluate_folder(input_dir, gold_dir, output_csv, select_ratio=0.5):
    results = []
    files = sorted(os.listdir(input_dir))
    for fname in files:
        input_path = os.path.join(input_dir, fname)
        if not os.path.isfile(input_path):
            continue

        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()

        generated_summary = lexrank_summarize(text, select_ratio=select_ratio,
                                             threshold=THRESHOLD, epsilon=EPSILON)

        gold_path = os.path.join(gold_dir, fname)
        if os.path.exists(gold_path):
            with open(gold_path, "r", encoding="utf-8") as f:
                gold_summary = f.read()
        else:
            gold_summary = ""
            print(f"[Warning] No gold summary for {fname}")

        # Compute ROUGE
        scores = SCORER.score(gold_summary, generated_summary)

        results.append({
            "file": fname,
            "rouge1_precision": scores["rouge1"].precision,
            "rouge1_recall": scores["rouge1"].recall,
            "rouge1_f1": scores["rouge1"].fmeasure,
            "rouge2_precision": scores["rouge2"].precision,
            "rouge2_recall": scores["rouge2"].recall,
            "rouge2_f1": scores["rouge2"].fmeasure,
            "rougeL_precision": scores["rougeL"].precision,
            "rougeL_recall": scores["rougeL"].recall,
            "rougeL_f1": scores["rougeL"].fmeasure
        })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Saved ROUGE results to {output_csv}")
    return df


# ----------------
# Main
# ----------------
if __name__ == "__main__":
    df = evaluate_folder(INPUT_DIR, GOLD_DIR, OUTPUT_CSV, select_ratio=SELECT_RATIO)
    print(df.describe().loc[["mean"]].T)  # quick summary (mean of metrics)
