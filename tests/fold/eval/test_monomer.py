"""score_result / score_target / summarize / CSV round trip / paired comparison (addendum §6)."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from oplm.fold.eval import monomer
from oplm.fold.eval.monomer import (
    METRICS,
    SamplingSettings,
    TargetScore,
    compare_scores,
    read_scores,
    score_result,
    score_target,
    summarize,
    write_scores,
)
from oplm.fold.eval.targets import load_targets
from tests.fold.helpers import tiny_folder

if TYPE_CHECKING:
    from pathlib import Path


def _result(coords: np.ndarray, names: list[str], n_res: int) -> SimpleNamespace:
    return SimpleNamespace(
        coords=torch.tensor(coords), atom_names=names, plddt=torch.full((n_res,), 0.8), ptm=0.6
    )


def _row(key: str, tm: float, status: str = "ok") -> TargetScore:
    return TargetScore(
        target_set="cameo",
        entry_id=key,
        chain_id="A",
        auth_chain_id="A",
        length=50,
        resolved_residues=50,
        status=status,
        tm_score=tm,
        lddt_ca=tm,
        lddt=tm,
        plddt=0.5,
        ptm=0.5,
    )


def test_score_result_of_the_reference_itself_is_perfect(fixture_dataset: Path) -> None:
    (t,), _ = load_targets(fixture_dataset, "cameo")
    scores = score_result(t, _result(t.coords, t.atom_names, len(t.sequence)))
    assert scores["tm_score"] == pytest.approx(1.0) and scores["lddt"] == pytest.approx(1.0)
    assert scores["lddt_ca"] == pytest.approx(1.0)
    assert scores["plddt"] == pytest.approx(0.8) and scores["ptm"] == pytest.approx(0.6)


def test_score_result_ignores_unresolved_reference_atoms(
    fixture_dataset_early_cutoff: Path,
) -> None:
    (t,), _ = load_targets(fixture_dataset_early_cutoff, "valid")  # 2ds7: 40 of 51 CA resolved
    pred = t.coords.copy()
    pred[~t.resolved] = 999.0  # wherever the reference has nothing, the prediction is absurd
    scores = score_result(t, _result(pred, t.atom_names, len(t.sequence)))
    assert scores["tm_score"] == pytest.approx(1.0) and scores["lddt"] == pytest.approx(1.0)


def test_score_result_rejects_a_different_atom_layout(fixture_dataset: Path) -> None:
    (t,), _ = load_targets(fixture_dataset, "cameo")
    names = list(t.atom_names)
    names[1] = "CB"
    with pytest.raises(ValueError, match="atom layout"):
        score_result(t, _result(t.coords, names, len(t.sequence)))


def test_score_target_folds_and_scores_with_a_tiny_model(fixture_dataset: Path) -> None:
    (t,), _ = load_targets(fixture_dataset, "cameo")
    row = score_target(tiny_folder(), t, SamplingSettings(seed=0))
    assert row.status == "ok", row.error
    assert (row.entry_id, row.chain_id, row.length, row.resolved_residues) == ("1crn", "A", 46, 46)
    for name in METRICS:
        assert 0.0 <= getattr(row, name) <= 1.0, name
    assert row.seconds > 0


def test_score_target_records_failures(
    fixture_dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (t,), _ = load_targets(fixture_dataset, "cameo")

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(monomer, "fold", boom)
    row = score_target(tiny_folder(), t, SamplingSettings())
    assert row.status == "failed" and row.error == "RuntimeError: boom"
    assert all(math.isnan(getattr(row, name)) for name in METRICS)


def test_summarize_averages_ok_rows_and_counts_failures() -> None:
    summary = summarize([_row("a", 0.4), _row("b", 0.6), _row("c", math.nan, status="failed")])
    assert summary["tm_score"] == pytest.approx(0.5)
    assert summary["targets"] == 2.0 and summary["failed"] == 1.0
    empty = summarize([])
    assert math.isnan(empty["tm_score"]) and empty["targets"] == 0.0


def test_score_table_round_trip(tmp_path: Path) -> None:
    rows = [_row("a", 0.4), _row("b", math.nan, status="failed")]
    rows[1].error = "RuntimeError: boom, with a comma"
    write_scores(tmp_path / "targets.csv", rows)
    back = read_scores(tmp_path / "targets.csv")
    assert back[0] == rows[0]
    assert back[1].error == rows[1].error and math.isnan(back[1].tm_score)


def test_compare_scores_is_paired_and_reproducible() -> None:
    a = [_row(k, v) for k, v in [("a", 0.40), ("b", 0.50), ("c", 0.60), ("x", 0.9)]]
    b = [_row(k, v) for k, v in [("a", 0.50), ("b", 0.62), ("c", 0.68), ("y", 0.1)]]
    b.append(_row("d", math.nan, status="failed"))
    out = compare_scores(a, b, samples=2000, seed=0)
    assert out["n"] == 3.0
    assert out["mean_diff"] == pytest.approx(0.1) and out["b_better_fraction"] == 1.0
    assert out["ci_low"] <= 0.1 <= out["ci_high"] and out["ci_low"] > 0.0
    assert compare_scores(a, b, samples=2000, seed=0) == out
    with pytest.raises(ValueError, match="metric"):
        compare_scores(a, b, metric="rmsd")
    with pytest.raises(ValueError, match="no target"):
        compare_scores(a[:1], b[1:2])
