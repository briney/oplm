"""Binary classification metrics.

Implemented in NumPy only (no scipy/sklearn dependency). The single entry point
is :func:`roc_auc_score`, used by the ProteinGym clinical-variant eval task to
score Pathogenic-vs-Benign discrimination.
"""

from __future__ import annotations

import numpy as np

from oplm.eval.metrics.ranking import average_ranks

__all__ = ["roc_auc_score"]


def roc_auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute binary ROC AUROC via the rank-based Mann–Whitney U statistic.

    ``AUROC = (sum_of_ranks_of_positives - n_pos*(n_pos+1)/2) / (n_pos * n_neg)``
    using average ("midrank") ranks, so tied scores contribute 0.5 pairwise.
    Higher ``scores`` are treated as more positive.

    Args:
        labels: 1-D array of ``{0.0, 1.0}`` where ``1`` is the positive class,
            shape ``(N,)``.
        scores: 1-D array of real-valued scores, shape ``(N,)`` (higher = more
            positive).

    Returns:
        The AUROC in ``[0, 1]``.

    Raises:
        ValueError: If the arrays are not 1-D, differ in shape, are empty, hold
            labels outside ``{0, 1}``, or contain only a single class (AUROC is
            undefined without both a positive and a negative example).
    """
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)

    if labels.ndim != 1 or scores.ndim != 1:
        raise ValueError(f"labels and scores must be 1-D, got {labels.ndim}-D and {scores.ndim}-D")
    if labels.shape != scores.shape:
        raise ValueError(
            f"labels and scores must have equal shape, got {labels.shape} vs {scores.shape}"
        )
    if labels.size == 0:
        raise ValueError("labels and scores must be non-empty")
    if not np.isin(labels, (0.0, 1.0)).all():
        raise ValueError("labels must be binary (0 or 1)")

    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"AUROC is undefined with a single class present (n_pos={n_pos}, n_neg={n_neg})"
        )

    ranks = average_ranks(scores)
    sum_ranks_pos = ranks[labels == 1.0].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)
