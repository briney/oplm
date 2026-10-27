"""EvalChain: fold-dataset chains to eval PDB files, and the shared curation rules."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from oplm.data import load_structures
from oplm.eval.metrics.contact import contact_positions_valid
from oplm.fold.eval.targets import load_targets
from oplm.fold.evalsets.chain import (
    EvalChain,
    count_long_range_contacts,
    from_eval_target,
    rejection,
    write_pdb,
)

if TYPE_CHECKING:
    from pathlib import Path

_RULES = {"min_length": 20, "max_length": 1024, "min_modeled_fraction": 0.7}


def _chain(dataset: Path, split: str, key: str) -> EvalChain:
    targets, _ = load_targets(dataset, split)
    (target,) = [t for t in targets if t.key == key]
    return from_eval_target(target, meta={})


def test_from_eval_target_keeps_unmodeled_residues(fixture_dataset_early_cutoff: Path) -> None:
    """2ds7_A: 51 residues in the sequence, 11 without atoms, 40 modeled."""
    chain = _chain(fixture_dataset_early_cutoff, "valid", "2ds7_A")
    assert chain.name == "2ds7_A" and len(chain.sequence) == 51
    assert chain.modeled == 40
    assert sum(not residue for residue in chain.atoms) == 11
    assert np.isnan(chain.backbone()).any(axis=(1, 2)).sum() == 11


def test_write_pdb_round_trips_through_the_loader(
    fixture_dataset_early_cutoff: Path, tmp_path: Path
) -> None:
    """The written file loads as the full sequence with NaN rows exactly where unmodeled."""
    chain = _chain(fixture_dataset_early_cutoff, "valid", "2ds7_A")
    write_pdb(chain, tmp_path / f"{chain.name}.pdb")
    (loaded,) = load_structures(tmp_path)
    assert loaded.name == "2ds7_A" and loaded.sequence == chain.sequence
    built = chain.backbone()
    assert np.array_equal(np.isnan(loaded.coords.numpy()), np.isnan(built))
    assert np.allclose(loaded.coords.numpy(), built, atol=1e-3, equal_nan=True)
    assert contact_positions_valid(loaded.coords).sum().item() == chain.modeled
    assert count_long_range_contacts(loaded.coords.numpy()) == count_long_range_contacts(built)


def test_atoms_hold_the_precision_a_pdb_file_stores(
    fixture_dataset_early_cutoff: Path, tmp_path: Path
) -> None:
    """Coordinates are rounded to the 0.001 Å a PDB file stores, so curation and the
    manifest's contact count see the coordinates the eval reads (7aal_A had a virtual-Cβ
    pair within 0.0005 Å of the 8 Å cutoff)."""
    chain = _chain(fixture_dataset_early_cutoff, "valid", "2ds7_A")
    jittered = EvalChain(
        name=chain.name,
        sequence=chain.sequence,
        atoms=[{a: xyz + np.float32(4e-4) for a, xyz in r.items()} for r in chain.atoms],
    )
    write_pdb(jittered, tmp_path / f"{chain.name}.pdb")
    (loaded,) = load_structures(tmp_path)
    np.testing.assert_array_equal(loaded.coords.numpy(), jittered.backbone())


def test_count_long_range_contacts(fixture_dataset: Path) -> None:
    """A lone helix has none; a globin has well over a hundred."""
    assert count_long_range_contacts(_chain(fixture_dataset, "casp14", "1jb6_A").backbone()) == 0
    assert count_long_range_contacts(_chain(fixture_dataset, "train", "4hhb_A").backbone()) > 100


@pytest.mark.parametrize(
    ("split", "key", "overrides", "reason"),
    [
        ("train", "4hhb_A", {}, None),
        ("cameo", "1crn_A", {"min_length": 50}, "length"),
        ("casp14", "1jb6_A", {}, "contacts"),
    ],
)
def test_rejection_reasons(
    fixture_dataset: Path, split: str, key: str, overrides: dict, reason: str | None
) -> None:
    assert rejection(_chain(fixture_dataset, split, key), **(_RULES | overrides)) == reason


def test_rejection_modeled_fraction(fixture_dataset_early_cutoff: Path) -> None:
    """2ds7_A is 40/51 = 78% modeled: fails a 0.8 floor."""
    chain = _chain(fixture_dataset_early_cutoff, "valid", "2ds7_A")
    assert rejection(chain, **(_RULES | {"min_modeled_fraction": 0.8})) == "modeled"


def test_nonstandard_residues_are_rejected_and_not_written(tmp_path: Path) -> None:
    chain = EvalChain(name="x", sequence="AXA", atoms=[{}, {}, {}])
    assert rejection(chain, **_RULES) == "nonstandard"
    with pytest.raises(ValueError, match="non-standard"):
        write_pdb(chain, tmp_path / "x.pdb")


def test_atoms_must_match_the_sequence() -> None:
    with pytest.raises(ValueError, match="atoms for 1 residues"):
        EvalChain(name="x", sequence="AA", atoms=[{}])
