# Extractive summarisation of legal judgments

This project compares TF-IDF, Sentence-BERT, Legal-BERT, and OpenAI embeddings in a role-weighted LexRank sentence graph. It includes 50 judgments, seven rhetorical roles, and 99 distinct reference summaries.

## Train the role labeller on Google Colab

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

Each training judgment contributes one vector of seven proportions, one per rhetorical role among its labelled sentences. The arithmetic mean averages those vectors. The Chebyshev center minimizes the distance to the farthest judgment. The geometric median minimizes the sum of distances, reducing the influence of atypical cases. Both distance-based methods use Euclidean distance. These weights capture how common each role is in source judgments. Validation ROUGE tests how well each profile works for summarization.

Each encoder selects its best of the three profiles by average validation ROUGE-L F1. All three profiles and an unweighted baseline are evaluated on the ten held-out cases. All encoders use the same sentence budgets (30%, 40%, and 50%) and stemmed ROUGE-1, ROUGE-2, and ROUGE-L precision, recall, and F1. Scores are averaged over distinct reference summaries within each case, then across cases.

### Results on the ten held-out judgments

The table reports ROUGE-L F1 averaged across the 30%, 40%, and 50% sentence budgets. The weighted and unweighted test columns use the same ten judgments and twenty reference summaries. The eight validation judgments determined the choice of weight method before test scoring.

| LexRank encoder | Weight method selected on validation | Validation, weighted | Test, weighted | Test, unweighted |
|---|---|---:|---:|---:|
| TF-IDF | Arithmetic mean | 0.3581 | 0.3739 | 0.3888 |
| Sentence-BERT | Chebyshev center | 0.3784 | 0.3767 | **0.3914** |
| Legal-BERT | Chebyshev center | **0.3850** | 0.3560 | 0.3751 |
| OpenAI `text-embedding-3-large` | Chebyshev center | 0.3797 | 0.3583 | 0.3850 |

Unweighted Sentence-BERT had the highest observed test average. Each encoder’s unweighted baseline outscored all three of its role-weighted profiles on that average. This comparison covers one 10-case test split. [Scores by summary length and weight method](runs/colab-benchmark/metrics.csv) · [Validation selection](runs/colab-benchmark/selection.csv) · [Seven role weights for each method](runs/colab-benchmark/weights.json)

Sentence-BERT uses the published 256-token input window, and Legal-BERT uses 512 tokens. In this corpus the Sentence-BERT limit truncates 11 of 8,656 raw judgment sentences. Legal-BERT fits all of them. The OpenAI backend needs `OPENAI_API_KEY` in the environment or an ignored local `.env` file. To run the other three backends, append `--backends tfidf sbert legalbert`. To summarise a judgment after running the benchmark, specify its output directory:

```sh
python -m legal_lexrank.summarize Judgement/1971_S_1.txt --backend legalbert --ratio 0.4 --results runs/colab-benchmark
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
