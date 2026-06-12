"""Ranking / rank-correlation metrics (NumPy only).

Houses the shared tie-aware ranking primitive (:func:`average_ranks`, also used
by the AUROC metric) plus Spearman's rank correlation and top-fraction precision
— the rank-based metrics on the ProteinGym DMS leaderboard.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["average_ranks", "spearman_corr", "top_k_precision"]


def average_ranks(scores: np.ndarray) -> np.ndarray:
    """Return 1-based average ("midrank") ranks of ``scores``.

    Tied values all receive the mean of the ranks they span, so tied pairs
    contribute 0.5 to rank-sum statistics. Sorting is stable for determinism.

    Args:
        scores: 1-D array of real-valued scores, shape ``(N,)``.

    Returns:
        Float array of shape ``(N,)`` with the average rank of each element,
        aligned to the input order.
    """
    n = scores.shape[0]
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]

    ranks_sorted = np.arange(1, n + 1, dtype=np.float64)
    group_ids = np.concatenate(([0], np.cumsum(sorted_scores[1:] != sorted_scores[:-1])))
    group_mean_rank = np.bincount(group_ids, weights=ranks_sorted) / np.bincount(group_ids)
    ranks_sorted = group_mean_rank[group_ids]

    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = ranks_sorted
    return ranks


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Spearman's rank correlation coefficient between ``a`` and ``b``.

    Equals the Pearson correlation of the (tie-aware) ranks of the two inputs.

    Args:
        a: 1-D array, shape ``(N,)``.
        b: 1-D array, shape ``(N,)`` (same shape as ``a``).

    Returns:
        The Spearman correlation in ``[-1, 1]``.

    Raises:
        ValueError: If the arrays are not equally-shaped 1-D arrays, have fewer
            than two elements, or either is constant (zero rank variance, so the
            correlation is undefined).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError(f"inputs must be 1-D, got {a.ndim}-D and {b.ndim}-D")
    if a.shape != b.shape:
        raise ValueError(f"inputs must have equal shape, got {a.shape} vs {b.shape}")
    if a.size < 2:
        raise ValueError("Spearman correlation needs at least 2 points")

    ra = average_ranks(a)
    rb = average_ranks(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = math.sqrt(float((ra * ra).sum()) * float((rb * rb).sum()))
    if denom == 0.0:
        raise ValueError("Spearman correlation undefined: an input has zero rank variance")
    return float((ra * rb).sum() / denom)


def top_k_precision(y_true: np.ndarray, y_pred: np.ndarray, *, fraction: float = 0.1) -> float:
    """Fraction of the top-``fraction`` items by ``y_true`` recovered by ``y_pred``.

    Ranks both arrays in descending order, takes the top ``k = ceil(fraction * N)``
    indices of each, and returns ``|true_top ∩ pred_top| / k``. Because both sets
    have size ``k`` this equals top-``fraction`` recall (the ProteinGym leaderboard
    metric). Ties are broken by index (stable sort) for determinism.

    Args:
        y_true: 1-D array of ground-truth scores (higher = better), shape ``(N,)``.
        y_pred: 1-D array of predicted scores (higher = better), shape ``(N,)``.
        fraction: Top fraction to score, in ``(0, 1]``. Default ``0.1``.

    Returns:
        Precision in ``[0, 1]``.

    Raises:
        ValueError: If the arrays are not equally-shaped 1-D arrays, are empty,
            or ``fraction`` is outside ``(0, 1]``.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError(f"inputs must be 1-D, got {y_true.ndim}-D and {y_pred.ndim}-D")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"inputs must have equal shape, got {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("inputs must be non-empty")
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    k = max(1, math.ceil(fraction * y_true.size))
    true_top = set(np.argsort(-y_true, kind="stable")[:k].tolist())
    pred_top = set(np.argsort(-y_pred, kind="stable")[:k].tolist())
    return len(true_top & pred_top) / k
