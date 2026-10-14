"""Target sets from the preprocessed fixtures: lists, splits, entity rule, coverage (addendum §6)."""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING

import numpy as np
import pytest

from oplm.fold.eval.targets import load_targets

if TYPE_CHECKING:
    from pathlib import Path


def _with_list(dataset: Path, tmp_path: Path, name: str, items: list[dict]) -> Path:
    root = tmp_path / "ds"
    shutil.copytree(dataset, root)
    (root / "targets" / f"{name}.json").write_text(json.dumps(items))
    return root


def test_list_targets_pick_the_named_author_chain(fixture_dataset: Path) -> None:
    targets, coverage = load_targets(fixture_dataset, "cameo")
    assert [t.key for t in targets] == ["1crn_A"]
    t = targets[0]
    assert t.target_set == "cameo" and t.auth_chain_id == "A" and len(t.sequence) == 46
    assert t.coords.shape == (len(t.atom_names), 3) and t.coords.dtype == np.float32
    assert [t.atom_names[i] for i in t.ca_index] == ["CA"] * 46
    assert t.resolved.all() and int(t.residue_index[-1]) == 45
    assert (coverage.listed, coverage.selected, coverage.missing, coverage.skipped) == (
        1,
        1,
        [],
        {},
    )


def test_split_targets_keep_unresolved_residues(fixture_dataset_early_cutoff: Path) -> None:
    targets, coverage = load_targets(fixture_dataset_early_cutoff, "valid")
    (t,) = targets
    assert t.key == "2ds7_A" and len(t.sequence) == 51 and len(t.ca_index) == 51
    assert int(t.resolved[t.ca_index].sum()) == 40  # 11 residues unmodeled, still tokens
    assert np.all(t.coords[~t.resolved] == 0.0)
    assert coverage.listed == 1 and coverage.selected == 1


def test_entry_items_take_one_chain_per_entity_and_report_coverage(
    fixture_dataset: Path, tmp_path: Path
) -> None:
    items = [
        {"id": "1jb6", "chain": None},  # homodimer: one entity -> chain A only
        {"id": "4hhb", "chain": None},  # two entities -> A (alpha) and B (beta)
        {"id": "9zzz", "chain": None},  # not in the snapshot
        {"id": "1crn", "chain": "Q"},  # no such author chain
    ]
    targets, coverage = load_targets(
        _with_list(fixture_dataset, tmp_path, "entries", items), "entries"
    )
    assert [t.key for t in targets] == ["1jb6_A", "4hhb_A", "4hhb_B"]
    assert coverage.listed == 4 and coverage.selected == 3
    assert coverage.missing == ["9zzz", "1crn_Q"]


def test_max_length_and_limit(fixture_dataset: Path, tmp_path: Path) -> None:
    items = [
        {"id": "1jb6", "chain": None},
        {"id": "4hhb", "chain": None},
        {"id": "1crn", "chain": "A"},
    ]
    root = _with_list(fixture_dataset, tmp_path, "entries", items)
    targets, coverage = load_targets(root, "entries", max_length=100)
    assert [t.key for t in targets] == ["1crn_A", "1jb6_A"]
    assert coverage.skipped == {
        "4hhb_A": "length 141 > max_length 100",
        "4hhb_B": "length 146 > max_length 100",
    }
    limited, _ = load_targets(root, "entries", max_length=100, limit=1)
    assert [t.key for t in limited] == ["1crn_A"]


def test_unknown_target_set_names_the_splits(fixture_dataset: Path) -> None:
    with pytest.raises(ValueError, match=r"unknown target set 'cmaeo'.*train"):
        load_targets(fixture_dataset, "cmaeo")
