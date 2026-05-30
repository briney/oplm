"""Tests for the structure loader (Phase 6, docs/DATA_TOOLING.md §5).

Parses a real PDB fixture (crambin, ``1CRN``) into :class:`StructureData` and
checks the backbone-coordinate contract, modified-residue mapping, missing-atom
``NaN`` handling, directory loading, and the biopython-missing ``ImportError``
path. Marked ``slow`` because biopython parsing is comparatively heavy.

Real structures live under ``tests/data/fixtures/structures/`` (see that
directory's README); tests ``pytest.skip`` when the fixture is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from oplm.data.structure.loader import (
    StructureData,
    _parse_single_structure,
    _residue_to_one_letter,
    load_structures,
)

pytestmark = pytest.mark.slow

_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "structures"
_CRAMBIN = _FIXTURE_DIR / "1CRN.pdb"

# Known properties of crambin (1CRN): a single chain A of 46 residues, all
# backbone atoms present (a high-resolution X-ray structure).
_CRAMBIN_SEQUENCE = "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN"


def _require_crambin() -> Path:
    """Return the crambin fixture path, skipping the test if it is absent."""
    if not _CRAMBIN.is_file():
        pytest.skip(f"structure fixture missing: {_CRAMBIN} (see tests/data/fixtures/README.md)")
    return _CRAMBIN


@pytest.fixture(scope="module")
def crambin() -> StructureData:
    """The parsed crambin structure (shared, read-only)."""
    parsed = _parse_single_structure(_require_crambin())
    assert parsed is not None  # crambin parses cleanly
    return parsed


# --------------------------------------------------------------------------- #
# Parsing contract
# --------------------------------------------------------------------------- #


def test_parse_contract(crambin: StructureData) -> None:
    """A real PDB parses to the documented StructureData contract."""
    L = len(crambin.sequence)
    assert crambin.name == "1CRN"
    assert crambin.chain_id == "A"
    assert crambin.sequence == _CRAMBIN_SEQUENCE
    assert crambin.coords.shape == (L, 3, 3)  # (L, 3, 3) backbone N, CA, C
    assert crambin.coords.dtype == torch.float32


def test_coords_align_with_sequence(crambin: StructureData) -> None:
    """The coordinate first axis matches the sequence length exactly."""
    assert crambin.coords.shape[0] == len(crambin.sequence)


def test_crambin_has_no_missing_atoms(crambin: StructureData) -> None:
    """Crambin is complete, so no backbone atom is NaN."""
    assert not torch.isnan(crambin.coords).any()


@pytest.mark.parametrize(
    ("resname", "expected"),
    [
        ("MSE", "M"),  # selenomethionine (modified-residue map)
        ("SEC", "C"),  # selenocysteine
        ("CSE", "C"),  # selenocysteine (alternate)
        ("SEP", "S"),  # phosphoserine
        ("TPO", "T"),  # phosphothreonine
        ("PTR", "Y"),  # phosphotyrosine
        ("ALA", "A"),  # standard (biopython table)
        ("TRP", "W"),  # standard
        ("HOH", "X"),  # water -> unknown
        ("UNK", "X"),  # explicit unknown
    ],
)
def test_residue_to_one_letter(resname: str, expected: str) -> None:
    """Modified residues, standard residues, and unknowns map as documented."""
    assert _residue_to_one_letter(resname) == expected


def test_missing_atom_becomes_nan(tmp_path: Path) -> None:
    """A residue missing a backbone atom yields a NaN row for that atom only.

    Builds a real-data variant of crambin with the first residue's backbone C
    atom deleted, then asserts the C row is NaN while N and CA stay finite.
    """
    source = _require_crambin()
    lines = source.read_text().splitlines(keepends=True)

    # Identify the first residue's sequence number from its first ATOM record.
    first_atom = next(line for line in lines if line.startswith("ATOM"))
    first_resseq = first_atom[22:26].strip()

    def _is_first_residue_backbone_c(line: str) -> bool:
        return (
            line.startswith("ATOM")
            and line[12:16].strip() == "C"  # backbone carbonyl C (not CA/CB/CG…)
            and line[22:26].strip() == first_resseq
        )

    kept = [line for line in lines if not _is_first_residue_backbone_c(line)]
    assert len(kept) == len(lines) - 1  # exactly one atom removed

    modified = tmp_path / "1CRN_missing_c.pdb"
    modified.write_text("".join(kept))

    parsed = _parse_single_structure(modified)
    assert parsed is not None
    assert len(parsed.sequence) == len(_CRAMBIN_SEQUENCE)  # residue still present

    n_row, ca_row, c_row = parsed.coords[0]
    assert not torch.isnan(n_row).any()  # N present
    assert not torch.isnan(ca_row).any()  # CA present
    assert torch.isnan(c_row).all()  # C deleted -> NaN row


# --------------------------------------------------------------------------- #
# Directory loading
# --------------------------------------------------------------------------- #


def test_load_structures_directory() -> None:
    """load_structures parses every structure in a directory."""
    _require_crambin()
    structures = load_structures(_FIXTURE_DIR)
    assert [s.name for s in structures] == sorted(s.name for s in structures)  # deterministic order
    assert "1CRN" in {s.name for s in structures}


def test_load_structures_max_cap() -> None:
    """max_structures caps the number returned."""
    _require_crambin()
    assert load_structures(_FIXTURE_DIR, max_structures=0) == []
    assert len(load_structures(_FIXTURE_DIR, max_structures=1)) == 1


def test_load_structures_missing_directory(tmp_path: Path) -> None:
    """A nonexistent directory raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_structures(tmp_path / "does_not_exist")


# --------------------------------------------------------------------------- #
# Optional-dependency gating
# --------------------------------------------------------------------------- #


def test_biopython_missing_raises_import_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When biopython is unavailable, parsing raises a clear ImportError.

    Simulated by poisoning ``sys.modules`` so ``from Bio.PDB import ...`` fails,
    without requiring biopython to actually be uninstalled.
    """
    monkeypatch.setitem(sys.modules, "Bio.PDB", None)

    (tmp_path / "fake.pdb").touch()  # a file to attempt parsing (content irrelevant)
    with pytest.raises(ImportError, match=r"oplm\[train\]"):
        load_structures(tmp_path)


def test_biopython_missing_in_residue_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    """The residue-name mapping also surfaces a clear ImportError without biopython."""
    monkeypatch.setitem(sys.modules, "Bio.Data.IUPACData", None)

    # A name not in the modified-residue map forces the biopython table lookup.
    with pytest.raises(ImportError, match=r"oplm\[train\]"):
        _residue_to_one_letter("ALA")
