# Transformer role-classifier results

The Google Colab run completed on 20 November 2026. The selected non-generative InLegalBERT classifier achieved **61.46% sentence accuracy** and **0.5390 macro F1** on the ten reserved judgments. It correctly classified **9 of 54** lower-court sentences. **The requested 90% accuracy target was not reached.**

## Training and selection

Both candidates started from pinned, unsupervised pretrained encoder weights with a newly initialized seven-role classification head. All encoder layers were fine-tuned on a Tesla T4; no generative model was used. Each target sentence received the immediately preceding and following sentences as context. Training used square-root inverse-frequency loss weights computed only from the fitting judgments.

The seed-42 split contains 32 fitting judgments, eight validation judgments and ten reserved test judgments. Candidate and epoch selection used validation macro F1 across all seven labels, with accuracy breaking ties. The test judgments were not used for selection.

| Candidate | Selected epoch | Validation accuracy | Validation macro F1 | Lower-court correct |
|---|---:|---:|---:|---:|
| InLegalBERT | 2 | 63.85% | 0.6007 | 19 / 99 |
| DeBERTa-v3-base | 4 | 58.71% | 0.5744 | 25 / 99 |

InLegalBERT was then reloaded from its original pretrained weights, given a fresh classification head, and refitted on all 40 training judgments for two epochs. The six-epoch learning-rate schedule horizon was retained. The frozen final model was evaluated on 2,120 annotated sentences across the ten reserved judgments.

Validation and test results refer to different fitted checkpoints and different judgments. The test report is not a new validation score. No further model selection was performed using these test results.

## Reserved-test results

| Role | Precision | Recall | F1 | Gold sentences |
|---|---:|---:|---:|---:|
| Facts | 58.43% | 80.61% | 0.6775 | 490 |
| Argument | 67.92% | 55.10% | 0.6085 | 196 |
| Statute | 74.79% | 78.76% | 0.7672 | 113 |
| Precedent | 57.48% | 52.16% | 0.5469 | 324 |
| Ratio of the decision | 68.99% | 57.00% | 0.6242 | 886 |
| Ruling by Present Court | 28.57% | 49.12% | 0.3613 | 57 |
| Ruling by Lower Court | 21.43% | 16.67% | 0.1875 | 54 |

Exact overall accuracy: `0.6146226415094339`. Exact macro F1: `0.5390236329853687`.

The classifier now predicts the lower-court class, but its precision and recall remain poor. The [training/validation annotation audit](role-label-audit.md) found label imbalance, concentrated examples and possible annotation ambiguities; it did not find dropped or misaligned labels, and did not change annotations. These observations do not establish a single cause of the model's errors. A larger supervised role corpus and independently adjudicated annotations are possible next experiments; they do not guarantee 90% accuracy.

## Summarization evaluation

The completed Colab pipeline also evaluated TF-IDF, S-BERT, Legal-BERT and OpenAI LexRank on the ten test judgments. Every backend used the same frozen role classifier, fixed role weights and unweighted comparison, 30%/40%/50% sentence budgets, and stemmed ROUGE-1/2/L precision, recall and F1. Distinct reference scores were averaged within each case before averaging across cases. The verified similarity cache required no new embedding API calls. Detailed reports are included under `summary-benchmark/` in the export.

## Provenance and export

- Platform: Google Colab, Tesla T4, CUDA.
- Python: 3.13.16; PyTorch: 2.14.1+cu130; Transformers: 4.57.1.
- Corrected candidate-training source: `88ec2403cc32bece0ac0a3f3c79182cf906a24a2`.
- Final refit and evaluation source: `3c861b6d46438406f62ff0845993ed19bcf3fab8`.
- Training started: `2026-11-20T12:31:10.766272+00:00`.
- Final model, evaluation, benchmark and export completed: `2026-11-20T13:49:31.255148+00:00`.
- Export: `transformer_role_labeller.zip`, 405,833,223 bytes.
- Export SHA-256: `ccf58a3dca7029808f843895bc4958fe98b5879c9da30538e8f86ebceb077a95`.

The export contains the final model, split and checkpoint verification metadata, candidate histories, test predictions and metrics, and all four summarization reports. At the time this report was written, the completed export remained in the Colab runtime awaiting manual download; a locally verified model release was not yet published. The values above were read from the completed Colab pipeline's status output.
