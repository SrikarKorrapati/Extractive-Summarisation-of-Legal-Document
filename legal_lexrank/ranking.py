"""Fit rhetorical-role profiles on training cases and rank a document's sentences."""

from collections import Counter
from collections.abc import Mapping, Sequence
import math

import numpy as np
from scipy.optimize import minimize


def _chebyshev_center(profiles: np.ndarray) -> np.ndarray:
    """Find the simplex center minimizing the largest Euclidean distance to a case."""
    if np.all(profiles == profiles[0]):
        return profiles[0].copy()

    # Minimize the squared radius. For each case the corresponding constraint is
    # radius_squared - ||center - case||^2 >= 0, which is smooth at zero.
    def margins(params):
        offsets = params[:-1] - profiles
        return params[-1] - np.einsum("ij,ij->i", offsets, offsets)

    def margin_jacobian(params):
        return np.column_stack((2 * (profiles - params[:-1]), np.ones(len(profiles))))

    mean = profiles.mean(axis=0)
    start = np.r_[mean, np.max(np.sum((profiles - mean) ** 2, axis=1))]
    objective_gradient = np.zeros_like(start)
    objective_gradient[-1] = 1
    sum_gradient = np.r_[np.ones_like(mean), 0]
    fit = minimize(
        lambda params: params[-1],
        x0=start,
        jac=lambda params: objective_gradient,
        method="SLSQP",
        bounds=[(0, 1)] * len(mean) + [(0, None)],
        constraints=[
            {"type": "ineq", "fun": margins, "jac": margin_jacobian},
            {"type": "eq", "fun": lambda p: p[:-1].sum() - 1, "jac": lambda p: sum_gradient},
        ],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not fit.success or np.min(margins(fit.x)) < -1e-8:
        raise RuntimeError(f"Minimum enclosing ball fit failed: {fit.message}")
    center = np.maximum(fit.x[:-1], 0)
    return center / center.sum()


def _geometric_median(profiles: np.ndarray) -> np.ndarray:
    """Minimize the sum of Euclidean distances using modified Weiszfeld iterations."""
    center = profiles.mean(axis=0)
    for _ in range(10000):
        offsets = profiles - center
        distances = np.linalg.norm(offsets, axis=1)
        away = distances > 1e-12
        if not away.any():
            return center
        inv_distance = 1 / distances[away]
        target = np.average(profiles[away], axis=0, weights=inv_distance)
        coincident = len(profiles) - away.sum()
        if coincident:
            # A data point is optimal only if its copies can cancel the gradient
            # contributed by the other cases. Otherwise take a shortened step.
            residual = np.linalg.norm((offsets[away] * inv_distance[:, None]).sum(axis=0))
            if residual <= coincident:
                return center
            target = center + (1 - coincident / residual) * (target - center)
        if np.linalg.norm(target - center) <= 1e-11:
            return target
        center = target
    raise RuntimeError("Geometric median did not converge")


def fit_weights(
    documents: Sequence[Sequence[str]], roles: Sequence[str]
) -> dict[str, dict[str, float]]:
    """Estimate three weight vectors from the labelled sentences in training cases."""
    roles = tuple(roles)
    if not roles or any(not role for role in roles) or len(set(roles)) != len(roles):
        raise ValueError("roles must contain unique, nonempty role names")
    role_set = set(roles)
    rows = []
    for document in documents:
        counts = Counter(document)
        if not counts or counts.keys() - role_set:
            raise ValueError(f"Expected a nonempty case with known roles, got {list(counts)}")
        rows.append([counts[role] / counts.total() for role in roles])
    if not rows:
        raise ValueError("At least one labelled training case is required")
    profiles = np.asarray(rows, dtype=float)
    estimates = {
        "mean": profiles.mean(axis=0),
        "chebyshev": _chebyshev_center(profiles),
        "geometric_median": _geometric_median(profiles),
    }
    return {
        name: {role: float(weight) for role, weight in zip(roles, values)}
        for name, values in estimates.items()
    }


def rank_sentences(
    cosine: np.ndarray, labels: Sequence[str], weights: Mapping[str, float] | None
) -> np.ndarray:
    """Rank sentence indices using the paper's weighted graph and uniform PageRank."""
    similarities = np.asarray(cosine, dtype=float)
    count = len(labels)
    if similarities.shape != (count, count) or not np.isfinite(similarities).all():
        raise ValueError("Expected a finite square cosine matrix aligned with labels")
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
