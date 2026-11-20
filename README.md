# Extractive summarisation of legal judgments

This repository compares four ways to represent sentences in a role-weighted LexRank graph. Each encoder sees the same judgment sentences, uses the same predicted rhetorical roles, and is scored against the same reference summaries.

## Findings

Legal-BERT with Chebyshev-center weights had the highest validation ROUGE-L F1 among the twelve weighted configurations (0.3719 averaged over three summary lengths). On the ten held-out judgments it scored 0.3885 on the same measure. TF-IDF with Chebyshev-center weights had the highest observed test average across all sixteen configurations (0.3987). That second comparison is descriptive because the test set was reserved for final scoring.

| Encoder | Weight method chosen on validation | Test ROUGE-L F1 at 30% | At 40% | At 50% | Mean | Unweighted mean |
|---|---|---:|---:|---:|---:|---:|
| TF-IDF | Geometric median | 0.3758 | 0.3824 | 0.3836 | 0.3806 | 0.3888 |
| Sentence-BERT | Arithmetic mean | 0.3684 | 0.3764 | 0.3811 | 0.3753 | 0.3914 |
| Legal-BERT | Chebyshev center | 0.3722 | 0.3908 | 0.4026 | 0.3885 | 0.3751 |
| OpenAI | Geometric median | 0.3624 | 0.3704 | 0.3751 | 0.3693 | 0.3853 |

For the validation-selected Legal-BERT configuration, the final role weights were **30.44% precedent, 27.61% facts, 24.25% reasoning, 9.32% statute, 5.46% argument, 1.87% present-court ruling, and 1.03% lower-court ruling**. These are estimates from the 40 training judgments.

All four encoders were run. Weighted and unweighted scores are available at every ratio. [Validation comparison](results/selection.csv) · [All ROUGE scores](results/metrics.csv)

## Data and fitting

- The corpus contains 50 judgments and 99 distinct reference summaries. A seeded shuffle of case IDs reserves 10 judgments for testing. Of the 40 training judgments, 32 are used to fit candidate methods and 8 are used for validation. The exact lists and encoder revisions are recorded in the [protocol](results/protocol.json).
- Each nonempty line in `Judgement/` is one extractable sentence. Each annotation in `Labelled_Data/` pairs a sentence with one of seven roles. These two text sources have different sentence counts in 49 cases. Supervised training uses the annotated sequence. Summarization predicts a fresh role for every sentence of the raw judgment.
- The hierarchical BiLSTM-CRF learns word embeddings on the fit cases. A word-level bidirectional LSTM encodes each sentence, a second bidirectional LSTM reads the judgment in order, and a linear-chain CRF decodes the role sequence with Viterbi. Both fits use eight epochs, with up to 96 tokens per sentence for classification. The final fit on all 40 training cases achieved 58.82% role accuracy and 0.4668 macro-F1 on 2,120 held-out annotations. Lower-court rulings had zero F1 on this split. [Per-role evaluation](results/role_metrics.json)
- Each training case contributes a vector of seven role frequencies. The **arithmetic mean** averages those vectors equally. The **Chebyshev center** minimizes the largest Euclidean distance to a case vector. The **geometric median** minimizes the sum of distances, limiting the influence of an unusual judgment. Final weights are refit on all 40 training cases after validation. The `validation` and `test` keys in the weights file identify where the weights are applied. [Role weights](results/weights.json)

## Ranking and evaluation

1. Compute sentence cosine similarities using per-judgment unigram TF-IDF, Sentence-BERT (`sentence-transformers/all-MiniLM-L6-v2`), Legal-BERT (`nlpaueb/legal-bert-base-uncased`), or OpenAI (`text-embedding-3-large`). The two local transformers use attention-masked mean pooling.
2. Connect each pair of distinct sentences by the nonnegative cosine similarity multiplied by both predicted role weights. The baseline uses cosine alone. Weighted PageRank uses damping 0.85 and distributes dangling-node mass uniformly.
3. At each requested ratio, select the highest-ranked sentences, round the count up, and restore original judgment order. For example, a judgment with 211 sentences contributes 64 at 30%.
4. Compute stemmed English ROUGE-1, ROUGE-2, and ROUGE-L precision, recall, and F1 against each distinct A1/A2 reference. Average within each case and then equally across cases. Select one of the three weighting methods per encoder by validation ROUGE-L F1 averaged over cases and the 30%, 40%, and 50% ratios. Score all methods and the baseline on the same ten test cases.

Each encoder JSON in `results/` records the zero-based indices of the selected source sentences and the nine ROUGE scores for every case, method, and ratio. The checkpoint in that directory is the final 40-case role labeller.

## Run

Use Python 3.11 from the repository root.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

Generate a summary using the published checkpoint and validation-selected weights. The input format is UTF-8 text with one sentence per nonempty line.

```sh
python -m legal_lexrank.summarize Judgement/1971_S_1.txt --backend legalbert --ratio 0.4
```

Run the full experiment into a fresh directory. For OpenAI, set `OPENAI_API_KEY` in the environment or in a local ignored `.env` file. The OpenAI backend sends sentence tokens for embedding. Downloaded local model weights and computed similarities are reused on subsequent runs.

```sh
python -m legal_lexrank --output runs/reproduction
```

To run only local encoders, append `--backends tfidf sbert legalbert`. Individual backends can also be added to the same output directory under the same protocol. The maintained implementation is in [legal_lexrank](legal_lexrank/).
