"""Rank judgment sentences with fixed rhetorical-role weights."""

from collections.abc import Mapping, Sequence
import math

import numpy as np


DEFAULT_ROLE_WEIGHTS = {
    "Facts": 0.10,
    "Argument": 0.10,
    "Statute": 0.10,
    "Precedent": 0.15,
    "Ratio of the decision": 0.15,
    "Ruling by Present Court": 0.20,
    "Ruling by Lower Court": 0.20,
}


def rank_sentences(
    cosine: np.ndarray, labels: Sequence[str] | None,
    weights: Mapping[str, float] | None = DEFAULT_ROLE_WEIGHTS,
) -> np.ndarray:
    """Rank sentence indices using the paper's weighted graph and uniform PageRank."""
    similarities = np.asarray(cosine, dtype=float)
    if similarities.ndim != 2 or similarities.shape[0] != similarities.shape[1] or not np.isfinite(similarities).all():
        raise ValueError("Expected a finite square cosine matrix")
    count = similarities.shape[0]
    if labels is not None and len(labels) != count:
        raise ValueError("Expected one label per sentence")
    if weights is not None and labels is None:
        raise ValueError("Role-weighted ranking requires predicted sentence labels")
    if count == 0:
        return np.empty(0, dtype=int)
    # Cosine may be negative, but transition probabilities must be nonnegative.
    graph = np.maximum(similarities, 0)
    np.fill_diagonal(graph, 0)
    if weights is not None:
        role_weights = np.array([weights[label] for label in labels], dtype=float)
        if np.any(role_weights < 0) or not np.isfinite(role_weights).all():
            raise ValueError("Rhetorical-role weights must be finite and nonnegative")
        graph *= role_weights[:, None]
        graph *= role_weights[None, :]

    outgoing = graph.sum(axis=1)
    dangling = outgoing == 0
    np.divide(graph, outgoing[:, None], where=~dangling[:, None], out=graph)
    damping = 0.85
    scores = np.full(count, 1 / count)
    for _ in range(1000):
        teleport = (1 - damping + damping * scores[dangling].sum()) / count
        updated = teleport + damping * np.einsum("i,ij->j", scores, graph)
        if np.abs(updated - scores).sum() < 1e-12:
            # Ignore roundoff when breaking mathematically equal scores by source order.
            return np.argsort(-np.round(updated, 14), kind="stable")
        scores = updated
    raise RuntimeError("Weighted PageRank did not converge")


def select_indices(ranking: Sequence[int], ratio: float) -> list[int]:
    """Select the requested fraction, rounded up, then restore source order."""
    if not 0 < ratio <= 1:
        raise ValueError("ratio must be in (0, 1]")
    count = math.ceil(len(ranking) * ratio)
    return sorted(int(index) for index in ranking[:count])
