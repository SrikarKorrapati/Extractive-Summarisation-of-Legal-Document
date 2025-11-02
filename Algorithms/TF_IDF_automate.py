import os
import pandas as pd
from nltk.tokenize import sent_tokenize
from lexrank import LexRank
from rouge_score import rouge_scorer
import nltk

# Directories
input_dir = "Judgement"         # Original text files
gold_dir = "Summary"            # Reference summaries
output_csv = "rouge_scores.csv"

# Prepare corpus for LexRank (list of lists of sentences per document)
corpus = []
for fname in os.listdir(input_dir):
    with open(os.path.join(input_dir, fname), "r", encoding="utf-8") as f:
        text = f.read()
        sentences = sent_tokenize(text)
        corpus.append(sentences)

lxr = LexRank(corpus)
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

results = []

for fname in os.listdir(input_dir):
    with open(os.path.join(input_dir, fname), "r", encoding="utf-8") as f:
        text = f.read()

    sentences = sent_tokenize(text)

    # Calculate 50% of total sentences (at least 1 sentence)
    summary_size = max(1, len(sentences) // 2)

    summary_sentences = lxr.get_summary(sentences, summary_size=summary_size, threshold=0.1)
    generated_summary = " ".join(summary_sentences)

    gold_path = os.path.join(gold_dir, fname)
    if os.path.exists(gold_path):
        with open(gold_path, "r", encoding="utf-8") as f:
            gold_summary = f.read()
    else:
        gold_summary = ""
        print(f"[Warning] No gold summary for {fname}")

    scores = scorer.score(gold_summary, generated_summary)

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
print(f"ROUGE scores saved to {output_csv}")
    