# Lower-court role annotation audit

This read-only audit examined **only the 32 fit and eight validation cases** from
the canonical seed-42 split. No test-case examples or labels were inspected. No
annotations were changed, and no model was trained or evaluated for this audit.

## Parsing and distribution

All **7,260 nonempty annotation rows** in these two splits were independently
checked against their raw sentence text and canonical role-label suffixes. There
were **zero parsing/alignment mismatches, missing labels, or unknown roles**.

| Split | Ruling by Lower Court (RLC) | Cases containing RLC | Contiguous RLC runs | Longest run |
| --- | ---: | ---: | ---: | ---: |
| Fit | 163 / 5,022 sentences (3.25%) | 20 / 32 | 37 | 19 sentences |
| Validation | 99 / 2,238 sentences (4.42%) | 7 / 8 | 11 | 29 sentences |

Validation positives are concentrated: **69 of 99** occur in just two cases,
`1963_S_59` (47) and `2009_B_16` (22). Most RLC sentences continue an existing
RLC passage: **126 of 163 fit sentences and 88 of 99 validation sentences** have
an immediately preceding RLC label. Consequently, many positive examples are
fragments or continuations whose attribution depends on earlier text.

## Representative examples

1. **Explicit lower-court ruling.** “The learned Magistrate … held the accused
   guilty…” is labelled RLC in
   [fit `2010_S_431`, line 21](../Labelled_Data/2010_S_431.txt#L21).
2. **Sentence fragment.** “500/- each with default clause” is labelled RLC,
   continuing a conviction and sentencing passage in
   [fit `2015_J_10`, line 30](../Labelled_Data/2015_J_10.txt#L30). The fragment
   alone does not identify a court or ruling.
3. **Overlap with Facts.** “the Industrial Court … allowed the appeal and set
   aside the award” is labelled **Facts** in
   [fit `1954_M_25`, line 26](../Labelled_Data/1954_M_25.txt#L26). Court names and
   decision verbs therefore do not uniquely identify RLC.
4. **Apparent ambiguity with Precedent.** In fit `2010_S_431`,
   [lines 38–47](../Labelled_Data/2010_S_431.txt#L38) introduce and quote
   *Willie Slaney* as **Precedent**. The continuation “This Court then examined…”
   and the following quotation, [lines 48–65](../Labelled_Data/2010_S_431.txt#L48),
   are labelled **RLC**; line 66 returns to Precedent. This boundary merits
   annotation review.
5. **Apparent ambiguity with Ratio/Precedent.** In validation `1963_S_59`,
   [lines 367–376](../Labelled_Data/1963_S_59.txt#L367) discuss a cited decision
   under **Ratio of the decision**. Its quotation beginning “Dealing with this
   matter this Court held…” is labelled **RLC** for 29 sentences,
   [lines 377–405](../Labelled_Data/1963_S_59.txt#L377).

The last two examples are **apparent annotation ambiguities, not confirmed
errors**. No relabelling is proposed without reviewing the annotation guidance
and the surrounding passages.

## Interpretation

The audit finds no evidence that label parsing dropped or shifted the RLC
examples. It does identify plausible learning difficulties: a rare class,
concentration in a few cases, fragmented sentences, attribution requiring
context, and overlapping or ambiguous annotation boundaries. For comparison,
the fit split contains 1,739 Ratio and 1,321 Facts examples versus 163 RLC
examples.

Class imbalance could contribute to a model predicting other roles for every
RLC sentence. These observations are **not causal proof** of the previous
model's zero RLC recall, nor do they establish its particular confusion
destinations. They support examining per-role predictions and attribution
context while preserving the held-out test split.
