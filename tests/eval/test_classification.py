"""Tests for the NumPy AUROC metric (``src/oplm/eval/metrics/classification.py``)."""

from __future__ import annotations

import numpy as np
import pytest

from oplm.eval.metrics.classification import roc_auc_score


def test_perfect_separation() -> None:
    """Positives strictly above negatives → AUROC 1.0."""
    labels = np.array([0, 0, 1, 1], dtype=float)
    scores = np.array([1.0, 2.0, 3.0, 4.0])
    assert roc_auc_score(labels, scores) == pytest.approx(1.0)


def test_perfectly_inverted() -> None:
    """Positives strictly below negatives → AUROC 0.0."""
    labels = np.array([0, 0, 1, 1], dtype=float)
    scores = np.array([4.0, 3.0, 2.0, 1.0])
    assert roc_auc_score(labels, scores) == pytest.approx(0.0)


def test_hand_computed_partial() -> None:
    """A known 0.75 case (one mis-ranked pair out of four)."""
    labels = np.array([0, 0, 1, 1], dtype=float)
    scores = np.array([1.0, 3.0, 2.0, 4.0])
    assert roc_auc_score(labels, scores) == pytest.approx(0.75)


def test_all_tied_is_chance() -> None:
    """Identical scores give a chance-level 0.5 via midranks."""
    labels = np.array([0, 1], dtype=float)
    scores = np.array([5.0, 5.0])
    assert roc_auc_score(labels, scores) == pytest.approx(0.5)


def test_midrank_tie_case() -> None:
    """A tie spanning one positive and one negative contributes 0.5 pairwise."""
    # positives {2, 3}, negatives {1, 2}: 3 wins + 1 tie out of 4 → 3.5/4.
    labels = np.array([0, 0, 1, 1], dtype=float)
    scores = np.array([1.0, 2.0, 2.0, 3.0])
    assert roc_auc_score(labels, scores) == pytest.approx(0.875)


def test_good_model_above_chance() -> None:
    """The task's sign convention: higher pathogenicity score for positives → > 0.5."""
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    pathogenicity = np.array([0.1, 0.2, 0.6, 0.5, 0.8, 0.9])
    assert roc_auc_score(labels, pathogenicity) > 0.5


@pytest.mark.parametrize("labels", [np.array([0.0, 0.0]), np.array([1.0, 1.0])])
def test_single_class_raises(labels: np.ndarray) -> None:
    """AUROC is undefined when only one class is present."""
    with pytest.raises(ValueError, match="single class"):
        roc_auc_score(labels, np.array([0.3, 0.7]))


def test_empty_raises() -> None:
    """Empty inputs raise."""
    with pytest.raises(ValueError, match="non-empty"):
        roc_auc_score(np.array([]), np.array([]))


def test_shape_mismatch_raises() -> None:
    """Mismatched shapes raise."""
    with pytest.raises(ValueError, match="equal shape"):
        roc_auc_score(np.array([0.0, 1.0]), np.array([0.1, 0.2, 0.3]))


def test_non_binary_labels_raise() -> None:
    """Labels outside {0, 1} raise."""
    with pytest.raises(ValueError, match="binary"):
        roc_auc_score(np.array([0.0, 2.0]), np.array([0.1, 0.2]))


def test_matches_brute_force_with_ties() -> None:
    """Rank formula equals the brute-force pairwise definition (ties = 0.5)."""
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=40).astype(float)
    if labels.sum() in (0, len(labels)):  # guarantee both classes
        labels[0], labels[1] = 0.0, 1.0
    scores = rng.integers(0, 5, size=40).astype(float)  # many ties

    pos = scores[labels == 1.0]
    neg = scores[labels == 0.0]
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    brute = wins / (len(pos) * len(neg))

    assert roc_auc_score(labels, scores) == pytest.approx(brute)
