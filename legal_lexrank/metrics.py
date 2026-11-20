"""Case-level ROUGE-1, ROUGE-2 and ROUGE-L with English stemming."""

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

from rapidfuzz.distance import LCSseq
from rouge_score.scoring import fmeasure
from rouge_score.tokenizers import DefaultTokenizer

SELECTION_METRIC = "rougeL_f1"

_TOKENIZER = DefaultTokenizer(use_stemmer=True)


@dataclass
class _PreparedText:
    tokens: list[str]
    unigrams: Counter[str]
    bigrams: Counter[tuple[str, str]]


def _prepare(text: str) -> _PreparedText:
    tokens = _TOKENIZER.tokenize(text)
    return _PreparedText(tokens, Counter(tokens), Counter(zip(tokens, tokens[1:])))


# The same references recur across encoders, compression ratios and role weights.
_prepare_reference = lru_cache(maxsize=128)(_prepare)


@lru_cache(maxsize=512)
def _score_distinct(summary: str, references: tuple[str, ...]) -> dict[str, float]:
    candidate = _prepare(summary)
    candidate_size = len(candidate.tokens)
    totals: defaultdict[str, float] = defaultdict(float)

    for text in references:
        reference = _prepare_reference(text)
        reference_size = len(reference.tokens)
        unigram_matches = sum((candidate.unigrams & reference.unigrams).values())
        bigram_matches = sum((candidate.bigrams & reference.bigrams).values())
        lcs_length = LCSseq.similarity(candidate.tokens, reference.tokens)
        for metric, matches, predicted, expected in (
            ("rouge1", unigram_matches, candidate_size, reference_size),
            ("rouge2", bigram_matches, candidate_size - 1, reference_size - 1),
            ("rougeL", lcs_length, candidate_size, reference_size),
        ):
            precision = matches / max(1, predicted)
            recall = matches / max(1, expected)
            totals[f"{metric}_precision"] += precision
            totals[f"{metric}_recall"] += recall
            totals[f"{metric}_f1"] += fmeasure(precision, recall)

    return {key: total / len(references) for key, total in totals.items()}


def score_summaries(summary: str, references: Sequence[str]) -> dict[str, float]:
    """Return precision, recall and F1 averaged over distinct reference texts.

    Empty texts score zero. At least one reference is required to define a mean.
    """
    if isinstance(references, str):
        raise TypeError("Pass references as a sequence of complete reference texts")
    distinct = tuple(dict.fromkeys(references))
    if not distinct:
        raise ValueError("At least one reference is required")
    return _score_distinct(summary, distinct).copy()
