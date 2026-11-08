"""Unit tests for the coevolution metrics (Jacobian pair scores vs Potts couplings)."""

from __future__ import annotations

import math

import pytest
import torch

from oplm.eval.metrics.categorical_jacobian import (
    StructurePairScoreData,
    compute_mean_pair_score_precision_at_l,
)
from oplm.eval.metrics.coevolution import (
    COEVOLUTION_METRIC_DIVISORS,
    compute_coevolution_metrics,
)


def _data(scores, potts, labels, seq_len, neff_bin="high"):
    n = len(scores)
    return StructurePairScoreData(
        scores=torch.tensor(scores, dtype=torch.float32),
        labels=torch.tensor(labels),
        seq_len=seq_len,
        pair_indices=(torch.zeros(n, dtype=torch.long), torch.zeros(n, dtype=torch.long)),
        potts_scores=None if potts is None else torch.tensor(potts, dtype=torch.float32),
        neff_bin=neff_bin,
    )


def test_hand_built_counts():
    # 6 pairs, k = 2. M = {0,1}, P = {1,2}, T = {0,1,2}
    d = _data([0.9, 0.8, 0.1, 0.1, 0.1, 0.1], [0.1, 0.9, 0.8, 0.1, 0.1, 0.1], [1, 1, 1, 0, 0, 0], 2)
    m = compute_coevolution_metrics([d], COEVOLUTION_METRIC_DIVISORS)
    assert m["coevo_precision_at_L"] == 0.5
    assert m["p_at_L_shared"] == 0.5
    assert m["p_at_L_beyond"] == 0.5
    assert m["p_at_L_potts_only"] == 0.5
    assert m["potts_precision_at_L"] == 1.0
    assert m["p_at_L_shared_neff_high"] == 0.5


def test_shared_plus_beyond_equals_structure_precision():
    g = torch.Generator().manual_seed(0)
    bins = ["low", "mid", "high"]
    ds = [
        _data(
            torch.rand(50, generator=g).tolist(),
            torch.rand(50, generator=g).tolist(),
            (torch.rand(50, generator=g) < 0.3).long().tolist(),
            seq_len=int(torch.randint(4, 30, (1,), generator=g)),
            neff_bin=bins[i % 3],
        )
        for i in range(20)
    ]
    m = compute_coevolution_metrics(ds, COEVOLUTION_METRIC_DIVISORS)
    for suffix, div in (("at_L", 1), ("at_L_2", 2), ("at_L_5", 5)):
        total = m[f"p_{suffix}_shared"] + m[f"p_{suffix}_beyond"]
        assert total == pytest.approx(compute_mean_pair_score_precision_at_l(ds, l_divisor=div))


def test_coevo_headline_excludes_low_bin():
    low = _data([0.9, 0.8, 0.1, 0.1], [0.1, 0.1, 0.9, 0.8], [1, 1, 0, 0], 2, "low")
    high = _data([0.9, 0.8, 0.1, 0.1], [0.9, 0.8, 0.1, 0.1], [1, 1, 0, 0], 2, "high")
    m = compute_coevolution_metrics([low, high], COEVOLUTION_METRIC_DIVISORS)
    assert m["coevo_precision_at_L"] == 1.0
    assert m["coevo_precision_at_L_neff_low"] == 0.0
    assert m["p_at_L_beyond"] == pytest.approx(0.5)  # (1.0 low + 0.0 high) / 2


def test_empty_groups_are_omitted():
    low = _data([0.9, 0.8, 0.1, 0.1], [0.9, 0.8, 0.1, 0.1], [1, 1, 0, 0], 2, "low")
    m = compute_coevolution_metrics([low], COEVOLUTION_METRIC_DIVISORS)
    assert "coevo_precision_at_L" not in m
    assert "coevo_precision_at_L_neff_mid" not in m
    assert "p_at_L_beyond" in m


def test_nan_potts_pairs_rank_last_and_skip_spearman():
    nan = float("nan")
    d = _data([0.9, 0.8, 0.5, 0.4, 0.3, 0.2], [nan, nan, 0.3, 0.2, 0.9, 0.1], [1, 1, 1, 0, 0, 0], 2)
    m = compute_coevolution_metrics([d], COEVOLUTION_METRIC_DIVISORS)
    assert all(math.isfinite(v) for v in m.values())
    # P = {4, 2}: avoids the NaN pairs 0 and 1, so M (= {0, 1}) and P are disjoint.
    assert m["coevo_precision_at_L"] == 0.0


def test_spearman_perfect_and_inverse():
    scores = [0.1, 0.5, 0.3, 0.9, 0.7]
    same = _data(scores, [2 * x for x in scores], [0] * 5, 2)
    rev = _data(scores, [-x for x in scores], [0] * 5, 2)
    assert compute_coevolution_metrics([same], ["coevo_spearman"])["coevo_spearman"] == (
        pytest.approx(1.0)
    )
    assert compute_coevolution_metrics([rev], ["coevo_spearman"])["coevo_spearman"] == (
        pytest.approx(-1.0)
    )


def test_structures_without_potts_are_ignored_and_only_requested_metrics_emitted():
    with_potts = _data([0.9, 0.1, 0.2], [0.9, 0.1, 0.2], [1, 0, 0], 1)
    without = _data([0.9, 0.1, 0.2], None, [1, 0, 0], 1)
    m = compute_coevolution_metrics([with_potts, without], ["p_at_L_beyond"])
    assert set(m) == {"p_at_L_beyond", "p_at_L_beyond_neff_high"}
