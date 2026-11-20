# Extractive summarisation of legal judgments

This project compares TF-IDF, Sentence-BERT, Legal-BERT, and OpenAI embeddings in a role-weighted LexRank sentence graph. It includes 50 judgments, seven rhetorical roles, and 99 distinct reference summaries.

## Train the role labeller on Google Colab

The primary classifier follows the **pretrained Hier-BiLSTM-CRF** variant referenced by the Legal LexRank paper: raw 200-dimensional legal sent2vec sentence vectors → document BiLSTM (100 units per direction) → dropout 0.5 → linear seven-role scores → CRF. It uses the original legal-domain encoder. It has no added projection, normalization, position features, or auxiliary classification loss. [Original model and pretrained variant](https://github.com/Law-AI/semantic-segmentation/tree/16d16b2c9bad824b527d4ffa3e94e029e282b133).

[Open the paper-model training notebook in Colab](https://colab.research.google.com/github/SrikarKorrapati/Extractive-Summarisation-of-Legal-Document/blob/main/notebooks/paper_role_labeller.ipynb). It downloads verified sentence vectors for the 40 training judgments; Colab needs neither an API key nor the large sent2vec binary. The notebook fits on 32 judgments and uses eight validation judgments to select an optimizer schedule and epoch, by accuracy then macro F1. A fresh final model is fitted on all 40 training judgments for that fixed schedule. Download its checkpoint bundle and extract it into `runs/paper-colab`. Evaluate the ten reserved judgments separately after the model is frozen.

The four summarization encoders—TF-IDF, S-BERT, Legal-BERT and OpenAI—share this same role classifier. Their vectors determine graph similarities; they do not replace the classifier's legal sent2vec encoder. The graph uses the fixed rhetorical-role importance weights below.

The earlier random-word baseline in `runs/colab-roles` reached 57.88% test accuracy and missed all 54 lower-court sentences. The contextual labeller modules and `tune_role_labeller.ipynb` are exploratory extensions with additional architectural components; they are not the primary paper reproduction. Current paper-model results will be reported after hosted training and frozen test evaluation finish.

## Reproduce the full summarization experiment

From the repository root, use Python 3.11 to install the dependencies and run the checks.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

After extracting the paper-model Colab bundle, evaluate its role predictions and then run the summarization benchmark. The runner verifies the split, training annotations, model implementation and checkpoint hashes before scoring. Install the native inference dependency below first.

```sh
python -m legal_lexrank.evaluate_roles --role-run runs/paper-colab --output runs/paper-role-evaluation
python -m legal_lexrank --role-run runs/paper-colab --output runs/paper-benchmark
```

## Install the legal sentence encoder for inference

Training in Colab uses the small exported feature archive. Classifying previously unseen text locally also needs the official 2.25 GB legal sent2vec binary and its native C++ binding. A C++ compiler is required. These commands use the pinned native source, not the unrelated PyPI package named `sent2vec`.

```sh
python -m pip install Cython==0.29.37
python -m pip install --no-deps --no-build-isolation https://github.com/epfml/sent2vec/archive/9efbc2dd69f6c737c3a752c9dc5fbb4843d578b6.zip
mkdir -p .cache/role-original
curl -L https://cse.iitkgp.ac.in/~saptarshi/models/sent2vec.bin -o .cache/role-original/sent2vec.bin
```

The adapter checks the exact file size and SHA256 `1e7f7f1c182a11444085262896ac833243ca4d8bd1f7168446f2219a0be909f0` before loading. Set `LEGAL_SENT2VEC_MODEL` to use a different file location. The binary stays outside Git. Sentence vectors are regenerated consistently for training and inference; the published text-vector files are not mixed with native outputs.

## Label a new judgment

Provide a UTF-8 file with one sentence per nonempty line and a fitted classifier checkpoint. The command prints each original sentence followed by its predicted role, separated by a tab.

```sh
python -m legal_lexrank.label_roles new_judgment.txt --checkpoint runs/paper-colab/role_labeller.pt
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
python -m legal_lexrank.summarize Judgement/1971_S_1.txt --backend legalbert --ratio 0.4 --results runs/paper-colab
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
