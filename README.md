# Extractive summarisation of legal judgments

This project compares TF-IDF, Sentence-BERT, Legal-BERT, and OpenAI embeddings in a role-weighted LexRank sentence graph. It includes 50 judgments, seven rhetorical roles, and 99 distinct reference summaries.

## Train the role labeller on Google Colab

The classifier learns from labelled training judgments and assigns labels to new documents. Summarization applies the fixed rhetorical-role weights listed below.

The [validation tuning notebook](https://colab.research.google.com/github/SrikarKorrapati/Extractive-Summarisation-of-Legal-Document/blob/main/notebooks/tune_role_labeller.ipynb) compares pretrained Legal-BERT and OpenAI sentence representations followed by a document BiLSTM-CRF. It fits on 32 judgments, measures classification accuracy and per-role recall on eight validation judgments, and selects the training configuration and epoch. The final classifier is then fitted on all 40 training judgments before evaluating the ten reserved judgments.

The following paragraph describes the initial classifier baseline.

[Open the training notebook in Colab](https://colab.research.google.com/github/SrikarKorrapati/Extractive-Summarisation-of-Legal-Document/blob/main/notebooks/train_role_labeller.ipynb). Connect to a hosted CPU runtime and choose **Runtime → Run all**. The notebook fetches a pinned revision of the public corpus and model code. Each run trains a new hierarchical BiLSTM-CRF from random initialization.

The seeded split reserves 10 whole judgments for testing. The notebook first trains on 32 cases and evaluates on 8 validation cases. It then trains a fresh final model on all 40 training cases and evaluates its role predictions on the 10 test cases. Each fit uses eight epochs. Download `/content/legal_lexrank_role_labeller.zip` from Colab’s Files pane before disconnecting. The archive contains both checkpoints, their checksums, per-role metrics, case IDs, and runtime details. The revised trainer reads every token of each sentence and learns transitions within contiguous labelled spans.

The [verified Colab run](runs/colab-roles/) includes both trained models. The final model correctly labelled 1,227 of 2,120 annotated sentences in the ten test judgments (57.88% accuracy, 0.4374 macro F1). Always predicting the most common training role, *Ratio of the decision*, gives 41.79% accuracy on the same cases. The model correctly identified zero of the 54 lower-court ruling sentences. [Per-role scores](runs/colab-roles/role_metrics.json) · [Training split and checksums](runs/colab-roles/protocol.json) · [Hosted runtime](runs/colab-roles/runtime.json)

## Reproduce the full summarization experiment

From the repository root, use Python 3.11 to install the dependencies and run the checks.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python -m legal_lexrank --output runs/experiment
```

To run the benchmark with the published Colab checkpoints, pass their directory explicitly. For a new Colab fit, extract your downloaded archive into a separate directory and pass its path. The runner verifies the split, training annotations, model implementation and checkpoint hashes before scoring.

```sh
python -m legal_lexrank --role-run runs/colab-roles --output runs/reproduction
```

## Label a new judgment

Provide a UTF-8 file with one sentence per nonempty line and a fitted classifier checkpoint. The command prints each original sentence followed by its predicted role, separated by a tab.

```sh
python -m legal_lexrank.label_roles new_judgment.txt --checkpoint runs/colab-roles/role_labeller.pt
```

## Fixed rhetorical-role weights

Every encoder uses the following default weights at all three summary lengths.

| Rhetorical role | Weight |
|---|---:|
| Facts | 0.10 |
| Argument | 0.10 |
| Statute | 0.10 |
| Precedent | 0.15 |
| Ratio of the decision | 0.15 |
| Ruling by Present Court | 0.20 |
| Ruling by Lower Court | 0.20 |

The seven weights sum to 1. They are defined in `legal_lexrank/ranking.py` and recorded in each benchmark's `protocol.json`. Each graph edge is the nonnegative sentence cosine similarity multiplied by the weights of the two predicted roles.

The summarization benchmark evaluates these defaults alongside a similarity-only baseline on the validation and test judgments. All encoders use the same sentence budgets (30%, 40%, and 50%) and stemmed ROUGE-1, ROUGE-2, and ROUGE-L precision, recall, and F1. Scores are averaged over distinct reference summaries within each case, then across cases. Reports identify the weighting as `fixed` or `unweighted`.

Sentence-BERT uses the published 256-token input window, and Legal-BERT uses 512 tokens. In this corpus the Sentence-BERT limit truncates 11 of 8,656 raw judgment sentences. Legal-BERT fits all of them. The OpenAI backend needs `OPENAI_API_KEY` in the environment or an ignored local `.env` file. To run the other three backends, append `--backends tfidf sbert legalbert`. To summarise a judgment, point to a directory containing a fitted `role_labeller.pt` checkpoint. The default applies the fixed weights. Use `--unweighted` for the similarity-only baseline.

```sh
python -m legal_lexrank.summarize Judgement/1971_S_1.txt --backend legalbert --ratio 0.4 --results runs/colab-roles
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
