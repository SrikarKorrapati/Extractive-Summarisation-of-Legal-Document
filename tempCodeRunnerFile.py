import os
import math
import numpy as np
import pandas as pd
import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from rouge_score import rouge_scorer

# Ensure NLTK data present
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# Config
INPUT_DIR = "Judgement"        # folder with original documents
GOLD_DIR = "Summary"           # folder with gold/reference summaries (same filenames)
OUTPUT_CSV = "rouge_scores.csv"
THRESHOLD = 0.1
EPSILON = 0.1
SELECT_RATIO = 0.5             # fraction of sentences to include in summary (50%)

STOP_WORDS = set(stopwords.words("english"))
SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def lexrank_summarize(text, select_ratio=0.5, threshold=0.1, epsilon=0.1):
    """
    Index-aligned LexRank implementation (avoids KeyError from duplicate sentences).
    Returns a summary string containing top (select_ratio * num_sentences) sentences.
    """
    sentences = sent_tokenize(text)
    ts = len(sentences)
    if ts == 0:
        return ""

    # Build frequency list aligned by sentence index
    freq_list = []
    for sent in sentences:
        freq_table = {}
        words = word_tokenize(sent)
        for word in words:
            w = word.lower()
            if not w.isalpha() or w in STOP_WORDS:
                continue
            freq_table[w] = freq_table.get(w, 0) + 1
        freq_list.append(freq_table)  # append, preserve index

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
            # tokenize, filter, lowercase, unique
            u1 = [w.lower() for w in word_tokenize(sentences[i]) if w.isalpha() and w.lower() not in STOP_WORDS]
            u2 = [w.lower() for w in word_tokenize(sentences[j]) if w.isalpha() and w.lower() not in STOP_WORDS]
            u1 = list(set(u1))
            u2 = list(set(u2))
            common = set(u1).intersection(u2)

            d1 = sum((tf_matrix[i].get(t, 0) * idf_table.get(t, 0)) ** 2 for t in u1)
            d2 = sum((tf_matrix[j].get(t, 0) * idf_table.get(t, 0)) ** 2 for t in u2)
            n = sum(tf_matrix[i].get(t, 0) * tf_matrix[j].get(t, 0) * (idf_table.get(t, 0) ** 2) for t in common)

            matrix[i][j] = (n / (math.sqrt(d1) * math.sqrt(d2))) if (d1 > 0 and d2 > 0) else 0.0

    # Threshold to build adjacency (binary) then normalize by out-degree
    degrees = np.zeros((ts,))
    for i in range(ts):
        for j in range(ts):
            if matrix[i, j] > threshold:
                matrix[i, j] = 1.0
                degrees[i] += 1
            else:
                matrix[i, j] = 0.0

    degrees[degrees == 0] = 1.0
    for i in range(ts):
        matrix[i, :] = matrix[i, :] / degrees[i]

    # Power method
    transposed = matrix.T
    p_vector = np.array([1.0 / ts] * ts)
    lambda_val = 1.0
    while lambda_val > epsilon:
        next_p = np.dot(transposed, p_vector)
        lambda_val = np.linalg.norm(next_p - p_vector)
        p_vector = next_p

    # select top-k sentences (50% by default)
    k = max(1, int(math.ceil(ts * select_ratio)))
    ranked_indices = np.argsort(-p_vector, kind='mergesort')  # stable sort
    topk = ranked_indices[:k]
    selected_indices = sorted(topk.tolist())

    summary = " ".join([sentences[i] for i in selected_indices])
    return summary


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


if __name__ == "__main__":
    df = evaluate_folder(INPUT_DIR, GOLD_DIR, OUTPUT_CSV, select_ratio=SELECT_RATIO)
    print(df.describe().loc[["mean"]].T)  # quick summary (mean of columns)
