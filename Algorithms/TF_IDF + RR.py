import os
import math
import numpy as np
import pandas as pd
import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from rouge_score import rouge_scorer
import re
from typing import List, Tuple, Optional

# -------------
# Setup / Config
# -------------
nltk.download("punkt")
nltk.download("stopwords")

INPUT_DIR = "Labelled_Data"        # folder with original documents
GOLD_DIR = "Summary"           # folder with gold/reference summaries (same filenames)
OUTPUT_CSV = "rouge_scores.csv"
THRESHOLD = 0.1
EPSILON = 0.1
SELECT_RATIO = 0.5        # fraction of sentences to include in summary (50%)
VERBOSE = False           # set True to print warnings about missing roles

STOP_WORDS = set(stopwords.words("english"))
SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

# -------------------------
# Canonical roles & aliases
# -------------------------
# -------------------------
# Canonical roles & aliases
# -------------------------
CANONICAL = {
    "FACTS": {
        "role": "Facts",
        "weight": 0.25792442243747693,
    },
    "ARGUMENT": {
        "role": "Argument",
        "weight": 0.09202074453854517,
    },
    "STATUTE": {
        "role": "Statute",
        "weight": 0.057173048752764716,
    },
    "PRECEDENT": {
        "role": "Precedent",
        "weight": 0.14170208528977463,
    },
    "RATIO DECIDENDI": {
        "role": "Ratio Decidendi",
        "weight": 0.38549155733794294,
    },
    "RATIO OF THE DECISION": {
        "role": "Ratio Decidendi",
        "weight": 0.38549155733794294,
    },
    "RULING BY PRESENT COURT": {
        "role": "Ruling by Present Court",
        "weight": 0.0319094255952695,
    },
    "RULING BY LOWER COURT": {
        "role": "Ruling by Lower Court",
        "weight": 0.03377871604822617,
    },
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
    # Case: 'text \t ROLE' or 'text\tROLE'
    if '\t' in s:
        left, right = s.rsplit('\t', 1)
        cand = right.strip()
        found = _normalize_and_match(cand)
        if found:
            return left.strip().rstrip(' .,:;—-()[]'), cand

    # Try matching aliases appended in parentheses, brackets or after dash/colon
    for alias in ALIASES:
        pat = re.compile(
            rf'(?P<body>.+?)\s*(?:[\(\[\-:—]\s*)?{re.escape(alias)}\s*(?:[\)\]\-:—\s]*)?$',
            flags=re.IGNORECASE | re.DOTALL)
        m = pat.match(s)
        if m:
            body = m.group("body").strip()
            return body.rstrip(' .,:;—-()[]'), alias

    # Try last token heuristics
    toks = re.split(r'\s+', s)
    if len(toks) >= 2:
        last = toks[-1].strip().strip('.,;:()[]')
        found = _normalize_and_match(last)
        if found:
            body = " ".join(toks[:-1]).strip()
            return body.rstrip(' .,:;—-()[]'), last

    # Nothing found
    return s, None


# ------------------------
# Modified LexRank routine
# ------------------------
def lexrank_summarize(text: str, select_ratio: float = 0.5,
                      threshold: float = 0.1, epsilon: float = 0.1) -> str:
    """
    Index-aligned LexRank that also multiplies sentence-sentence similarity by
    (weight_i * weight_j), where weight_* come from rhetorical roles.
    Input sentences are expected to possibly contain a role label at the end.
    """
    raw_sentences = sent_tokenize(text)
    ts = len(raw_sentences)
    if ts == 0:
        return ""

    # Parse sentences -> (clean_text, role_alias, weight)
    sentences = []
    role_aliases = []
    weights = []
    for s in raw_sentences:
        body, raw_role = detect_role_from_line(s)
        sentences.append(body)
        role_aliases.append(raw_role)
        if raw_role:
            key = _normalize_and_match(raw_role)
            if key and key in CANONICAL:
                weights.append(float(CANONICAL[key]["weight"]))
            else:
                if VERBOSE:
                    print(f"[Warn] Unknown role alias '{raw_role}' -> using weight=1.0")
                weights.append(1.0)
        else:
            # No role detected: neutral weight 1.0
            weights.append(1.0)

    # Build frequency list aligned by sentence index (on cleaned sentences)
    freq_list = []
    for sent in sentences:
        freq_table = {}
        words = word_tokenize(sent)
        for word in words:
            w = word.lower()
            if not w.isalpha() or w in STOP_WORDS:
                continue
            freq_table[w] = freq_table.get(w, 0) + 1
        freq_list.append(freq_table)

    # TF matrix: index -> {word: tf}
    tf_matrix = {}
    for i, f_table in enumerate(freq_list):
        total = sum(f_table.values()) or 1
        tf_matrix[i] = {w: cnt / total for w, cnt in f_table.items()}

    # IDF (per document) computed from freq_list (count how many sentences contain each word)
    word_per_doc = {}
    for f_table in freq_list:
        for w in f_table.keys():
            word_per_doc[w] = word_per_doc.get(w, 0) + 1
    idf_table = {w: math.log10(ts / float(cnt)) if cnt > 0 else 0.0 for w, cnt in word_per_doc.items()}

    # Cosine similarity matrix using tf-idf weighting over intersection of token sets
    matrix = np.zeros((ts, ts))
    for i in range(ts):
        for j in range(ts):
            # tokenize, filter, lowercase, unique (on cleaned sentences)
            u1 = [w.lower() for w in word_tokenize(sentences[i]) if w.isalpha() and w.lower() not in STOP_WORDS]
            u2 = [w.lower() for w in word_tokenize(sentences[j]) if w.isalpha() and w.lower() not in STOP_WORDS]
            u1 = list(set(u1))
            u2 = list(set(u2))
            common = set(u1).intersection(u2)

            d1 = sum((tf_matrix[i].get(t, 0) * idf_table.get(t, 0)) ** 2 for t in u1)
            d2 = sum((tf_matrix[j].get(t, 0) * idf_table.get(t, 0)) ** 2 for t in u2)
            n = sum(tf_matrix[i].get(t, 0) * tf_matrix[j].get(t, 0) * (idf_table.get(t, 0) ** 2) for t in common)

            raw_sim = (n / (math.sqrt(d1) * math.sqrt(d2))) if (d1 > 0 and d2 > 0) else 0.0
            matrix[i][j] = raw_sim

    # --------------------------
    # Apply rhetorical weighting
    # --------------------------
    # Multiply each entry by weight_i * weight_j BEFORE thresholding.
    # This implements: matrix[i][j] := matrix[i][j] * (weight_i * weight_j)
    weight_array = np.array(weights, dtype=float)
    for i in range(ts):
        for j in range(ts):
            matrix[i][j] = matrix[i][j] * (weight_array[i] * weight_array[j])

    # Threshold to build adjacency (binary) then normalize by out-degree
    degrees = np.zeros((ts,))
    for i in range(ts):
        for j in range(ts):
            if matrix[i, j] > threshold:
                matrix[i, j] = 1.0
                degrees[i] += 1
            else:
                matrix[i, j] = 0.0

    # avoid divide-by-zero
    degrees[degrees == 0] = 1.0
    for i in range(ts):
        matrix[i, :] = matrix[i, :] / degrees[i]

    # Power method with damping factor
    damping = 0.85
    transposed = matrix.T
    p_vector = np.array([1.0 / ts] * ts)
    lambda_val = 1.0
    while lambda_val > epsilon:
        next_p = (1 - damping) / ts + damping * np.dot(transposed, p_vector)
        lambda_val = np.linalg.norm(next_p - p_vector)
        p_vector = next_p

    # select top-k sentences (select_ratio by default)
    k = max(1, int(math.ceil(ts * select_ratio)))
    ranked_indices = np.argsort(-p_vector, kind='mergesort')  # stable sort
    topk = ranked_indices[:k]
    selected_indices = sorted(topk.tolist())

    # Return joined original *cleaned* sentences (without role labels)
    summary = " ".join([sentences[i] for i in selected_indices])
    return summary


# --------------------------
# Evaluation wrapper (same)
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


if __name__ == "__main__":
    ratios = [0.3, 0.4, 0.5]
    results = {}

    for r in ratios:
        output_csv = f"rouge_scores_ratio_{int(r*100)}.csv"
        print(f"\n=== Running evaluation with SELECT_RATIO = {r} ===")
        df = evaluate_folder(INPUT_DIR, GOLD_DIR, output_csv, select_ratio=r)
        results[r] = df.describe().loc[["mean"]].T
        print(results[r])

    # Save combined summary statistics to one CSV
    summary_df = pd.concat(
        {f"ratio_{int(r*100)}": stats for r, stats in results.items()},
        axis=1
    )
    summary_df.to_csv("rouge_summary_comparison.csv")
    print("\nSaved summary comparison to rouge_summary_comparison.csv")

