"""Tests for the ranking metrics (``src/oplm/eval/metrics/ranking.py``)."""

from __future__ import annotations

import numpy as np
import pytest

from oplm.eval.metrics.ranking import spearman_corr, top_k_precision

# --- spearman_corr -------------------------------------------------------------


def test_spearman_perfect_monotonic() -> None:
    """A strictly increasing relationship gives +1."""
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([10.0, 20.0, 30.0, 40.0])
    assert spearman_corr(a, b) == pytest.approx(1.0)


def test_spearman_perfect_inverse() -> None:
    """A strictly decreasing relationship gives -1."""
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([40.0, 30.0, 20.0, 10.0])
    assert spearman_corr(a, b) == pytest.approx(-1.0)


def test_spearman_known_value() -> None:
    """A hand-computed case: 1 - 6*sum(d^2)/(n(n^2-1)) = 0.8."""
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([2.0, 1.0, 4.0, 3.0, 5.0])
    assert spearman_corr(a, b) == pytest.approx(0.8)


def test_spearman_handles_ties() -> None:
    """Tied values use midranks; correlation stays in range and is symmetric."""
    a = np.array([1.0, 2.0, 2.0, 3.0])
    b = np.array([5.0, 6.0, 7.0, 8.0])
    value = spearman_corr(a, b)
    assert -1.0 <= value <= 1.0
    assert spearman_corr(b, a) == pytest.approx(value)


def test_spearman_constant_input_raises() -> None:
    """A constant input has zero rank variance → undefined."""
    with pytest.raises(ValueError, match="zero rank variance"):
        spearman_corr(np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0]))


def test_spearman_too_few_points_raises() -> None:
    """Fewer than two points raises."""
    with pytest.raises(ValueError, match="at least 2"):
        spearman_corr(np.array([1.0]), np.array([2.0]))


def test_spearman_shape_mismatch_raises() -> None:
    """Mismatched shapes raise."""
    with pytest.raises(ValueError, match="equal shape"):
        spearman_corr(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))


# --- top_k_precision -----------------------------------------------------------


def test_top_k_precision_perfect() -> None:
    """Identical rankings recover the full top-k."""
    y_true = np.array([4.0, 3.0, 2.0, 1.0])
    y_pred = np.array([4.0, 3.0, 2.0, 1.0])
    assert top_k_precision(y_true, y_pred, fraction=0.5) == pytest.approx(1.0)


def test_top_k_precision_disjoint() -> None:
    """A reversed ranking recovers none of the true top-k."""
    y_true = np.array([4.0, 3.0, 2.0, 1.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])
    assert top_k_precision(y_true, y_pred, fraction=0.5) == pytest.approx(0.0)


def test_top_k_precision_partial() -> None:
    """k=2 of 8 with exactly one shared top item → 0.5."""
    # true top-2 = indices {0, 1}; pred top-2 = indices {0, 7}; overlap {0}.
    y_true = np.array([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    y_pred = np.array([8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0])
    assert top_k_precision(y_true, y_pred, fraction=0.25) == pytest.approx(0.5)


def test_top_k_precision_small_n_uses_at_least_one() -> None:
    """A tiny fraction still selects at least one item (k >= 1)."""
    y_true = np.array([3.0, 2.0, 1.0])
    y_pred = np.array([3.0, 2.0, 1.0])
    assert top_k_precision(y_true, y_pred, fraction=0.1) == pytest.approx(1.0)


@pytest.mark.parametrize("fraction", [0.0, 1.5, -0.1])
def test_top_k_precision_bad_fraction_raises(fraction: float) -> None:
    """A fraction outside (0, 1] raises."""
    with pytest.raises(ValueError, match="fraction"):
        top_k_precision(np.array([1.0, 2.0]), np.array([1.0, 2.0]), fraction=fraction)


def test_top_k_precision_empty_raises() -> None:
    """Empty inputs raise."""
    with pytest.raises(ValueError, match="non-empty"):
        top_k_precision(np.array([]), np.array([]))
