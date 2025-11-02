import os
import math
import numpy as np
import pandas as pd
import nltk
from nltk import sent_tokenize
from nltk.corpus import stopwords
from rouge_score import rouge_scorer
import re
from typing import List, Tuple, Optional

# --- SBERT ---
from sentence_transformers import SentenceTransformer
SBERT_MODEL = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# -------------
# Setup / Config
# -------------
nltk.download("punkt")
nltk.download("stopwords")

INPUT_DIR = "Labelled_Data"        # folder with original documents
GOLD_DIR = "Summary"               # folder with gold/reference summaries (same filenames)
OUTPUT_CSV = "rouge_scores.csv"
THRESHOLD = 0.1
EPSILON = 0.1
SELECT_RATIO = 0.3        # fraction of sentences to include in summary (50%)
VERBOSE = False           # set True to print warnings about missing roles

STOP_WORDS = set(stopwords.words("english"))
SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

# -------------------------
# Canonical roles & aliases
# -------------------------
CANONICAL = {
    "FACTS": {"role": "Facts", "weight": 0.12},
    "ARGUMENT": {"role": "Argument", "weight": 0.14},
    "STATUTE": {"role": "Statute", "weight": 0.16},
    "PRECEDENT": {"role": "Precedent", "weight": 0.10},
    "RATIO DECIDENDI": {"role": "Ratio Decidendi", "weight": 0.20},
    "RATIO OF THE DECISION": {"role": "Ratio Decidendi", "weight": 0.20},
    "RULING BY LOWER COURT": {"role": "Ruling by Lower Court", "weight": 0.08},
    "RULING BY PRESENT COURT": {"role": "Ruling by Present Court", "weight": 0.20},
}
ALIASES = sorted(CANONICAL.keys(), key=len, reverse=True)


# ---------------------------
# Role-detection helper funcs
# ---------------------------
def _normalize_and_match(token: str) -> Optional[str]:
    norm = re.sub(r'[\s\._\-]+', ' ', token).upper().strip()
    if norm in CANONICAL:
        return norm
    norm2 = re.sub(r'^[\W_]+|[\W_]+$', '', norm)
    if norm2 in CANONICAL:
        return norm2
    norm3 = re.sub(r'[^A-Z0-9 ]+', '', norm)
    if norm3 in CANONICAL:
        return norm3
    return None


def detect_role_from_line(line: str) -> Tuple[str, Optional[str]]:
    """
    Try to detect a rhetorical-role alias at the end of the line.
    Returns: (clean_sentence_body, raw_alias_or_None)
    """
    s = line.rstrip()
    if '\t' in s:
        left, right = s.rsplit('\t', 1)
        cand = right.strip()
        found = _normalize_and_match(cand)
        if found:
            return left.strip().rstrip(' .,:;—-()[]'), cand

    for alias in ALIASES:
        pat = re.compile(
            rf'(?P<body>.+?)\s*(?:[\(\[\-:—]\s*)?{re.escape(alias)}\s*(?:[\)\]\-:—\s]*)?$',
            flags=re.IGNORECASE | re.DOTALL)
        m = pat.match(s)
        if m:
            body = m.group("body").strip()
            return body.rstrip(' .,:;—-()[]'), alias

    toks = re.split(r'\s+', s)
    if len(toks) >= 2:
        last = toks[-1].strip().strip('.,;:()[]')
        found = _normalize_and_match(last)
        if found:
            body = " ".join(toks[:-1]).strip()
            return body.rstrip(' .,:;—-()[]'), last

    return s, None


# ------------------------
# LexRank with SBERT
# ------------------------
def lexrank_summarize(text: str, select_ratio: float = 0.5,
                      threshold: float = 0.1, epsilon: float = 0.1) -> str:
    """
    LexRank with SBERT embeddings + rhetorical role weighting.
    """
    raw_sentences = sent_tokenize(text)
    ts = len(raw_sentences)
    if ts == 0:
        return ""

    # Parse sentences -> (clean_text, role_alias, weight)
    sentences = []
    weights = []
    for s in raw_sentences:
        body, raw_role = detect_role_from_line(s)
        sentences.append(body)
        if raw_role:
            key = _normalize_and_match(raw_role)
            if key and key in CANONICAL:
                weights.append(float(CANONICAL[key]["weight"]))
            else:
                weights.append(1.0)
        else:
            weights.append(1.0)

    # SBERT embeddings
    embeddings = SBERT_MODEL.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)

    # Cosine similarity matrix
    sims = np.dot(embeddings, embeddings.T)
    np.fill_diagonal(sims, 0.0)

    # Apply rhetorical weighting
    weight_array = np.array(weights, dtype=float)
    sims = sims * (weight_array[:, None] * weight_array[None, :])

    # Threshold to build adjacency
    adj = (sims > threshold).astype(float)
    row_sums = adj.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    matrix = adj / row_sums[:, None]

    # Power method for PageRank
    damping = 0.85
    p_vector = np.ones(ts) / ts
    delta = 1.0
    while delta > epsilon:
        next_p = (1 - damping) / ts + damping * (matrix.T @ p_vector)
        delta = np.linalg.norm(next_p - p_vector)
        p_vector = next_p

    # Select top-k sentences
    k = max(1, int(math.ceil(ts * select_ratio)))
    ranked_indices = np.argsort(-p_vector, kind="mergesort")
    selected_indices = sorted(ranked_indices[:k].tolist())

    return " ".join([sentences[i] for i in selected_indices])


# --------------------------
# Evaluation wrapper
# --------------------------
def evaluate_folder(input_dir: str, gold_dir: str, output_csv: str, select_ratio: float = 0.5):
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
            if VERBOSE:
                print(f"[Warning] No gold summary for {fname}")

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


if __name__ == "__main__":
    df = evaluate_folder(INPUT_DIR, GOLD_DIR, OUTPUT_CSV, select_ratio=SELECT_RATIO)
    print(df.describe().loc[["mean"]].T)
