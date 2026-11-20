# Extractive summarisation of legal judgments

This project compares TF-IDF, Sentence-BERT, Legal-BERT, and OpenAI embeddings in a role-weighted LexRank sentence graph. It includes 50 judgments, seven rhetorical roles, and 99 distinct reference summaries.

## Train the role labeller on Google Colab

The primary role classifier fully fine-tunes a **non-generative transformer encoder** and a new seven-role classification head. It compares [InLegalBERT](https://huggingface.co/law-ai/InLegalBERT) with [DeBERTa-v3-base](https://huggingface.co/microsoft/deberta-v3-base). These are pretrained language encoders used to predict role labels; the classifier does not generate text. Every backbone layer is trained. The separate LexRank summarizer extracts original sentences.

[Open the transformer training notebook in Colab](https://colab.research.google.com/github/SrikarKorrapati/Extractive-Summarisation-of-Legal-Document/blob/main/notebooks/transformer_role_labeller.ipynb) and select a GPU runtime, such as a T4. The notebook uses Colab's preinstalled PyTorch with `transformers==4.57.1` and `sentencepiece==0.2.1`; it does not install the local Python 3.11 dependency lock. Role training needs no API key or native sent2vec installation.

Each target sentence is encoded with its immediately preceding and following sentences from the same judgment. The classifier mean-pools only target-token representations, then applies dropout and a linear seven-role head. Inputs are limited to 512 tokens. Long targets are split into chunks of at most 384 tokens, preserving every target token; surrounding context is trimmed to fit. Chunk losses and sentence-level predictions are weighted by each chunk's share of target tokens.

The notebook runs three separate stages with the fixed seed-42 judgment split:

1. Fit each of the two configurations on 32 judgments. Select the encoder and epoch using macro F1 across all seven roles on eight validation judgments, with accuracy breaking ties. Training is bounded to six epochs per candidate with early-stopping patience two. Reports include lower-court precision, recall, F1 and support.
2. Reload the selected original pretrained encoder and initialize a new head. Refit on all 40 training judgments for the selected epoch count, retaining the six-epoch learning-rate schedule horizon. Class weights are recomputed from these 40 judgments. This stage performs no evaluation.
3. Freeze the final model and evaluate the ten reserved judgments once in a separate cell. Keep the model choice fixed after seeing these results.

Both configurations use AdamW, square-root inverse-frequency class weights from their fitting data, FP16, gradient checkpointing and an effective batch of 32 chunks. InLegalBERT uses encoder/head learning rates `2e-5`/`1e-4`; DeBERTa uses `1e-5`/`1e-4`. The source revision, encoder revisions, split, selection and checkpoint hashes are recorded for verification. The downloaded bundle contains the final model, provenance, histories and held-out evaluation; validation checkpoints stay outside the deployment bundle. Extract it into `runs/transformer-colab`.

An optional notebook cell also runs the complete summary benchmark on Colab after role evaluation. It verifies a published cache of 40 similarity matrices covering the four backends and ten test judgments, then uses the frozen role classifier on the GPU. This cached benchmark needs no API key and includes its completed JSON/CSV reports in the download under `summary-benchmark/`. The cache contains similarities and provenance, with no role labels or credentials.

The classifier's frequency-based loss weights address label imbalance. LexRank's fixed role weights below determine sentence importance in the summary graph.

The completed Colab run selected InLegalBERT and refitted it on all 40 training judgments. Its reserved-test accuracy is **61.46%**, macro F1 is **0.5390**, and lower-court recall is **9/54 (16.67%)**. **The requested 90% accuracy target was not reached.** See the [measured results and provenance](docs/transformer-colab-results.md). The completed model export is awaiting manual download from Colab before local verification and release publication.

The earlier experimental paper-model reproduction uses legal sent2vec vectors and a Hier-BiLSTM-CRF. Its validation accuracy was 65.06%, with zero of 99 lower-court sentences recognized. Its [notebook](notebooks/paper_role_labeller.ipynb) and implementation remain available as a baseline.

The four summarization encoders—TF-IDF, S-BERT, Legal-BERT and OpenAI—share the selected role classifier. Their vectors determine graph similarities. The graph uses the fixed rhetorical-role importance weights below.

## Reproduce the full summarization experiment

From the repository root, use Python 3.11 to install the dependencies and run the checks.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

After extracting the transformer Colab bundle, inspect the role report in `runs/transformer-colab/test-evaluation/role_metrics.json`. If you ran the optional hosted summary benchmark, its comparison is already in `runs/transformer-colab/summary-benchmark/metrics.csv`. Otherwise, run the summarization benchmark below. The runner verifies the split, training annotations, model implementation, selection and checkpoint hashes before scoring. The transformer bundle contains the complete fine-tuned weights; its pinned tokenizer files are downloaded on first local use and then cached. Add `--device cuda` for GPU role inference.

```sh
python -m legal_lexrank --role-run runs/transformer-colab --output runs/transformer-benchmark
```

If the separate held-out evaluation cell has not yet been run, evaluate the already frozen model once before the benchmark. Use `--device cuda` when a CUDA GPU is available; CPU is the default.

```sh
python -m legal_lexrank.evaluate_roles --role-run runs/transformer-colab --output runs/transformer-colab/test-evaluation
```

## Label a new judgment

Provide a UTF-8 file with one sentence per nonempty line and a fitted classifier checkpoint. The command prints each original sentence followed by its predicted role, separated by a tab.

```sh
python -m legal_lexrank.label_roles new_judgment.txt --checkpoint runs/transformer-colab/role_labeller.pt
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

For the final transformer classifier, the summarization benchmark evaluates these defaults alongside a similarity-only baseline on the ten test judgments only. All four encoders use the same sentence budgets (30%, 40%, and 50%) and stemmed ROUGE-1, ROUGE-2, and ROUGE-L precision, recall, and F1. Scores are averaged over distinct reference summaries within each case, then across cases. Reports identify the weighting as `fixed` or `unweighted`. The eight validation judgments are part of the final classifier's 40-case training set and are not scored as held-out summaries.

Sentence-BERT uses the published 256-token input window, and Legal-BERT uses 512 tokens. In this corpus the Sentence-BERT limit truncates 11 of 8,656 raw judgment sentences. Legal-BERT fits all of them. Uncached OpenAI embeddings need `OPENAI_API_KEY` in the environment or an ignored local `.env` file; the optional Colab benchmark reuses verified similarities for all ten test judgments and needs no key. To run only the other three backends locally, append `--backends tfidf sbert legalbert`. To summarise a judgment, point to a directory containing a fitted `role_labeller.pt` checkpoint. The default applies the fixed weights. Use `--unweighted` for the similarity-only baseline.

```sh
python -m legal_lexrank.summarize Judgement/1971_S_1.txt --backend legalbert --ratio 0.4 --results runs/transformer-colab
```

## Repository map

| Path | Contents |
|---|---|
| `Judgement/` | Source judgments, one sentence per nonempty line |
| `Labelled_Data/` | Annotated sentence sequences for role training |
| `Summary/`, `Summary_2/` | The two reference summary sets |
| `legal_lexrank/` | Corpus loading, modelling, evaluation, and command-line tools |
| `notebooks/` | Transformer Colab workflow and earlier experimental notebooks |
| `tests/` | Automated checks |
| `docs/references/` | Research papers |
