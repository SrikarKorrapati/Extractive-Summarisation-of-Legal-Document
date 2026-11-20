# Extractive summarisation of legal judgments

This project compares TF-IDF, Sentence-BERT, Legal-BERT, and OpenAI embeddings in a role-weighted LexRank sentence graph. It includes 50 judgments, seven rhetorical roles, and 99 distinct reference summaries.

## Train the role labeller on Google Colab

[Open the training notebook in Colab](https://colab.research.google.com/github/SrikarKorrapati/Extractive-Summarisation-of-Legal-Document/blob/main/notebooks/train_role_labeller.ipynb). Connect to a hosted CPU runtime and choose **Runtime → Run all**. The notebook fetches a pinned revision of the public corpus and model code. Each run trains a new hierarchical BiLSTM-CRF from random initialization.

The seeded split reserves 10 whole judgments for testing. The notebook first trains on 32 cases and evaluates on 8 validation cases. It then trains a fresh final model on all 40 training cases and evaluates its role predictions on the 10 test cases. Each fit uses eight epochs. Download `/content/legal_lexrank_role_labeller.zip` from Colab’s Files pane before disconnecting. The archive contains the final checkpoint, per-role metrics, case IDs, and runtime details.

## Reproduce the full summarization experiment

From the repository root, use Python 3.11 to install the dependencies and run the checks.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python -m legal_lexrank --output runs/experiment
```

The full benchmark fits its own labellers and three candidate role-weight profiles. It selects a profile for each encoder on validation ROUGE-L F1 and evaluates all three profiles plus an unweighted baseline on the held-out cases. All encoders use the same sentence budgets (30%, 40%, and 50%) and stemmed ROUGE-1, ROUGE-2, and ROUGE-L precision, recall, and F1. Scores are averaged over distinct reference summaries within each case, then across cases.

The OpenAI backend needs `OPENAI_API_KEY` in the environment or an ignored local `.env` file. To run the other three backends, append `--backends tfidf sbert legalbert`. To summarise a judgment after running the benchmark, specify its output directory:

```sh
python -m legal_lexrank.summarize Judgement/1971_S_1.txt --backend legalbert --ratio 0.4 --results runs/experiment
```

## Repository map

| Path | Contents |
|---|---|
| `Judgement/` | Source judgments, one sentence per nonempty line |
| `Labelled_Data/` | Annotated sentence sequences for role training |
| `Summary/`, `Summary_2/` | The two reference summary sets |
| `legal_lexrank/` | Corpus loading, modelling, evaluation, and command-line tools |
| `notebooks/` | Hosted Colab training notebook |
| `tests/` | Automated checks |
| `docs/references/` | Research papers |
