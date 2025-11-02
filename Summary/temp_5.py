"""
Enhanced SBERT summarizer tuned for legal/judgment text.

Features:
- Robust sentence tokenization (spaCy preferred; NLTK/Punkt fallback + regex)
- Cluster (HDBSCAN / KMeans) -> medoid representatives selection
- MMR selection to trade relevance vs novelty
- Near-duplicate removal (embedding cosine threshold)
- Simple sentence compression (truncate long bracketed/semicolon clauses)
- Paragraph-wise summarization fallback for very long documents
- ROUGE evaluation output (same structure as your original script)

Requirements (recommended):
pip install sentence-transformers umap-learn hdbscan scikit-learn scipy rouge-score spacy
python -m spacy download en_core_web_sm

If spaCy isn't installed, the script uses a robust regex fallback for splitting.
"""

import os
import math
import re
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from rouge_score import rouge_scorer

# Optional: spaCy for better sentence segmentation
try:
    import spacy
    SPACY_AVAILABLE = True
    nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "parser"])
except Exception:
    SPACY_AVAILABLE = False

# Clustering & DR
try:
    import umap
    import hdbscan
    from sklearn.cluster import KMeans
except Exception:
    umap = None
    hdbscan = None
    KMeans = None

# SBERT
from sentence_transformers import SentenceTransformer, util

# Ensure NLTK data present (punkt used only as fallback)
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# ---------------- Config (edit as needed) ----------------
INPUT_DIR = "Judgement"
GOLD_DIR = "Summary"
OUTPUT_CSV = "rouge_scores_enhanced.csv"

SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
SELECT_RATIO = 0.5          # fraction of sentences to include in summary
CLUSTER_METHOD = "hdbscan"   # 'hdbscan' or 'kmeans'
UMAP_N_COMPONENTS = 5
MIN_CLUSTER_SIZE_RATIO = 0.12
MMR_LAMBDA = 0.7             # tradeoff relevance vs diversity in MMR (0..1)
DUP_SIM_THRESHOLD = 0.88     # remove near-duplicate sentences with cosine > this
MAX_SENTENCE_WORDS = 45      # if sentence > this, apply light compression
PARAGRAPH_CHUNKING = True    # if doc very long, summarize per paragraph then combine
CHUNK_PARAGRAPH_THRESHOLD = 800  # chars; paragraph longer than this will be chunked
RANDOM_STATE = 42
# --------------------------------------------------------

STOP_WORDS = set(stopwords.words("english"))
SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
SBERT = SentenceTransformer(SBERT_MODEL_NAME)


# ---------------- Utilities ----------------
def normalize_whitespace(text: str) -> str:
    # fix crazy spacing, ensure single space after periods if missing (e.g. "1924the")
    text = re.sub(r"\s+", " ", text)
    # ensure space after period when followed directly by lowercase letter/digit (e.g. "1924when")
    text = re.sub(r"\.([a-z0-9])", r". \1", text)
    return text.strip()


def better_sentence_tokenize(text: str):
    """
    Try spaCy first (best). Fallback to NLTK.sent_tokenize, then regex split on semicolons/newlines
    Returns list of sentences (stripped).
    """
    text = text.strip()
    if not text:
        return []

    if SPACY_AVAILABLE:
        doc = nlp(text)
        sents = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        if len(sents) >= 1:
            return sents

    # fallback: use nltk punkt
    try:
        sents = nltk.sent_tokenize(text)
        # further split very long ones on semicolons or '  ' (double newline)
        out = []
        for s in sents:
            parts = re.split(r";|—|--|\n\n", s)
            for p in parts:
                p = p.strip()
                if p:
                    out.append(p)
        if out:
            return out
    except Exception:
        pass

    # last resort regex: split on period followed by space + lowercase or newline or by double newline
    parts = re.split(r"\.\s+|\n\n", text)
    sents = [p.strip() + ('.' if not p.strip().endswith('.') and len(p.strip().split())>1 else '') for p in parts if p.strip()]
    return sents


def light_compress_sentence(sent: str, max_words=MAX_SENTENCE_WORDS):
    """
    Very light rule-based compression:
    - remove parenthetical clauses and trailing semicolon clauses
    - truncate after first main clause if extremely long
    This is intentionally conservative (keeps factual content).
    """
    s = sent.strip()
    # remove parentheticals
    s = re.sub(r"\([^)]*\)", "", s)
    # split by semicolon or '—' and keep first part if too long
    if len(s.split()) > max_words:
        parts = re.split(r";|—|--|:|\.\s", s)
        if parts:
            s = parts[0].strip()
    # final safety: if still long, truncate to nearest sentence boundary before max_words
    words = s.split()
    if len(words) > max_words:
        s = " ".join(words[:max_words]) + "..."
    return s.strip()


def canonicalize_sentence(sent: str):
    # normalize spacing and capitalization of first character (sentences often all lower-case in legal texts)
    s = re.sub(r"\s+", " ", sent).strip()
    if not s:
        return s
    # ensure first char uppercase
    s = s[0].upper() + s[1:]
    # ensure it ends with period-like punctuation
    if not re.search(r"[.!?]$", s):
        s = s + "."
    return s


def dedupe_selected(sentences, embeddings, threshold=DUP_SIM_THRESHOLD):
    """
    Remove sentences that are near-duplicates based on cosine sim. Keep earlier one.
    sentences: list[str] (order preserved)
    embeddings: numpy array same order
    """
    kept = []
    kept_embs = []
    for i, s in enumerate(sentences):
        if not kept:
            kept.append(s); kept_embs.append(embeddings[i])
            continue
        sims = util.cos_sim(embeddings[i], np.vstack(kept_embs)).numpy().ravel()
        if np.any(sims >= threshold):
            # skip near-duplicate
            continue
        kept.append(s); kept_embs.append(embeddings[i])
    return kept


# ---------------- MMR ----------------
def mmr_selection(candidate_embs, candidate_indices, doc_centroid_emb, k, lambda_param=MMR_LAMBDA):
    """
    candidate_embs: numpy array (n_candidates, dim)
    candidate_indices: list of original indices corresponding to each candidate
    doc_centroid_emb: numpy (dim,)
    returns: list of chosen candidate original indices (not positions in candidate list)
    """
    chosen = []
    chosen_pos = []
    n = candidate_embs.shape[0]
    if n == 0:
        return []

    sims_to_doc = util.cos_sim(candidate_embs, doc_centroid_emb).numpy().ravel()  # shape (n,)
    # initialize by picking the best single sentence
    first = int(np.argmax(sims_to_doc))
    chosen_pos.append(first)
    while len(chosen_pos) < k and len(chosen_pos) < n:
        remaining = [i for i in range(n) if i not in chosen_pos]
        mmr_scores = []
        for idx in remaining:
            sim_doc = sims_to_doc[idx]
            sim_selected = 0.0
            if chosen_pos:
                sim_sel_vec = util.cos_sim(candidate_embs[idx], candidate_embs[chosen_pos]).numpy().ravel()
                sim_selected = float(np.max(sim_sel_vec))
            score = lambda_param * sim_doc - (1 - lambda_param) * sim_selected
            mmr_scores.append((score, idx))
        # choose max
        best = max(mmr_scores, key=lambda x: x[0])[1]
        chosen_pos.append(best)

    return [candidate_indices[i] for i in chosen_pos]


# ---------------- Summarizers ----------------
def cluster_then_mmr_summary(text: str, select_ratio=SELECT_RATIO, method=CLUSTER_METHOD,
                             umap_components=UMAP_N_COMPONENTS, min_cluster_size_ratio=MIN_CLUSTER_SIZE_RATIO,
                             rep_selection="medoid", mmr_lambda=MMR_LAMBDA):
    """
    Robust pipeline:
    - Single-level paragraph chunking (no deep recursion)
    - For each paragraph, call _summarize_no_chunk (which does not chunk again)
    - Combine para summaries and produce final summary via _summarize_no_chunk
    """
    text = normalize_whitespace(text)

    # If very long and paragraph-chunking enabled, split into paragraphs but only if split yields multiple paras.
    if PARAGRAPH_CHUNKING and len(text) > 3000:
        paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        # Only chunk if we actually got more than one paragraph; otherwise skip (prevents infinite recursion)
        if len(paras) > 1:
            para_summaries = []
            # Summarize each paragraph without allowing further paragraph-chunk recursion
            for p in paras:
                # if paragraph is extremely long, reduce select_ratio for that paragraph
                local_ratio = max(0.12, select_ratio * (0.6 if len(p) > CHUNK_PARAGRAPH_THRESHOLD else 0.4))
                para_summaries.append(_summarize_no_chunk(p, select_ratio=local_ratio,
                                                          method=method,
                                                          umap_components=umap_components,
                                                          min_cluster_size_ratio=min_cluster_size_ratio,
                                                          rep_selection=rep_selection,
                                                          mmr_lambda=mmr_lambda))
            joined = " ".join([ps for ps in para_summaries if ps.strip()])
            if not joined.strip():
                return ""
            # Final pass on the combined short document (no further chunking)
            return _summarize_no_chunk(joined, select_ratio=select_ratio,
                                      method=method,
                                      umap_components=umap_components,
                                      min_cluster_size_ratio=min_cluster_size_ratio,
                                      rep_selection=rep_selection,
                                      mmr_lambda=mmr_lambda)
        # else fall through to normal (no chunking) summary

    # Default: no paragraph-chunking (or too short to chunk)
    return _summarize_no_chunk(text, select_ratio=select_ratio,
                               method=method,
                               umap_components=umap_components,
                               min_cluster_size_ratio=min_cluster_size_ratio,
                               rep_selection=rep_selection,
                               mmr_lambda=mmr_lambda)


def _summarize_no_chunk(text: str, select_ratio=SELECT_RATIO, method=CLUSTER_METHOD,
                        umap_components=UMAP_N_COMPONENTS, min_cluster_size_ratio=MIN_CLUSTER_SIZE_RATIO,
                        rep_selection="medoid", mmr_lambda=MMR_LAMBDA):
    """
    The core summarizer that DOES NOT do paragraph-chunking.
    This is the function that performs tokenization, embedding, clustering, MMR, dedupe, compression.
    (This code is essentially the inner part of your old function after paragraph-chunking.)
    """
    sents = better_sentence_tokenize(text)
    ts = len(sents)
    if ts == 0:
        return ""
    if ts == 1:
        return canonicalize_sentence(light_compress_sentence(sents[0]))

    k_target = max(1, int(math.ceil(ts * select_ratio)))

    # encode
    try:
        emb = SBERT.encode(sents, convert_to_numpy=True, normalize_embeddings=True)
    except TypeError:
        emb = SBERT.encode(sents, convert_to_numpy=True)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)

    # prepare embeddings for clustering
    emb_for_clustering = emb
    if method == "hdbscan" and umap is not None and hdbscan is not None:
        n_comp = min(umap_components, emb.shape[1])
        reducer = umap.UMAP(n_components=n_comp, random_state=RANDOM_STATE)
        emb_for_clustering = reducer.fit_transform(emb)

    # clustering
    clusters = {}
    if method == "hdbscan" and hdbscan is not None:
        min_cluster_size = max(2, int(ts * min_cluster_size_ratio))
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
        labels = clusterer.fit_predict(emb_for_clustering)
        for idx, lab in enumerate(labels):
            key = lab if lab != -1 else ("noise", idx)
            clusters.setdefault(key, []).append(idx)
    else:
        # fallback to kmeans
        cluster_k = min(max(1, int(round(math.sqrt(ts)))), ts)
        if KMeans is None:
            clusters[0] = list(range(ts))
        else:
            km = KMeans(n_clusters=cluster_k, random_state=RANDOM_STATE)
            labels = km.fit_predict(emb_for_clustering)
            for idx, lab in enumerate(labels):
                clusters.setdefault(int(lab), []).append(idx)

    # choose candidates (medoid/centroid)
    candidate_indices = []
    candidate_embs = []
    from scipy.spatial.distance import cdist
    for label, idxs in clusters.items():
        idxs_arr = np.array(idxs)
        if len(idxs_arr) == 1:
            candidate_indices.append(int(idxs_arr[0]))
            candidate_embs.append(emb[int(idxs_arr[0])])
            continue
        cluster_embs = emb[idxs_arr]
        if rep_selection == "centroid":
            centroid = cluster_embs.mean(axis=0, keepdims=True)
            sims = util.cos_sim(centroid[0], cluster_embs).numpy().ravel()
            best = int(idxs_arr[int(np.argmax(sims))])
            candidate_indices.append(best)
            candidate_embs.append(emb[best])
        else:
            dists = cdist(cluster_embs, cluster_embs, metric="cosine").sum(axis=1)
            medoid_idx = int(idxs_arr[int(np.argmin(dists))])
            candidate_indices.append(medoid_idx)
            candidate_embs.append(emb[medoid_idx])

    candidate_embs = np.vstack(candidate_embs) if candidate_embs else np.zeros((0, emb.shape[1]))

    # supplement candidates if fewer than target
    if len(candidate_indices) < k_target:
        doc_centroid = emb.mean(axis=0, keepdims=True)
        sims = util.cos_sim(emb, doc_centroid[0]).numpy().ravel()
        sorted_idx = list(np.argsort(-sims))
        for idx in sorted_idx:
            if idx not in candidate_indices:
                candidate_indices.append(int(idx))
                candidate_embs = np.vstack([candidate_embs, emb[int(idx)]]) if candidate_embs.size else emb[int(idx)][None, :]
            if len(candidate_indices) >= k_target:
                break

    # select final using MMR
    doc_centroid = emb.mean(axis=0)
    selected_orig_indices = mmr_selection(candidate_embs, candidate_indices, doc_centroid, k_target, lambda_param=mmr_lambda)

    selected_sorted = sorted(set(selected_orig_indices))
    selected_sents = [sents[i] for i in selected_sorted]
    selected_embs = np.vstack([emb[i] for i in selected_sorted])

    # dedupe and finalize
    selected_sents = dedupe_selected(selected_sents, selected_embs, threshold=DUP_SIM_THRESHOLD)
    final_sents = [canonicalize_sentence(light_compress_sentence(s)) for s in selected_sents]
    summary = " ".join(final_sents)
    return summary

# ---------------- Evaluation loop ----------------
def evaluate_folder(input_dir, gold_dir, output_csv, select_ratio=SELECT_RATIO):
    results = []
    files = sorted(os.listdir(input_dir))
    for fname in files:
        input_path = os.path.join(input_dir, fname)
        if not os.path.isfile(input_path):
            continue

        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()

        generated_summary = cluster_then_mmr_summary(text, select_ratio=select_ratio,
                                                     method=CLUSTER_METHOD,
                                                     umap_components=UMAP_N_COMPONENTS,
                                                     min_cluster_size_ratio=MIN_CLUSTER_SIZE_RATIO,
                                                     rep_selection="medoid",
                                                     mmr_lambda=MMR_LAMBDA)

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
            "rougeL_f1": scores["rougeL"].fmeasure,
            "generated_summary": generated_summary[:200]  # small preview for debugging
        })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Saved ROUGE results to {output_csv}")
    return df


if __name__ == "__main__":
    df = evaluate_folder(INPUT_DIR, GOLD_DIR, OUTPUT_CSV, select_ratio=SELECT_RATIO)
    print(df.describe().loc[["mean"]].T)
