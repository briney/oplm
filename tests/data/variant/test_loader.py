"""Tests for the variant loader (Phase 7, docs/DATA_TOOLING.md §6).

Covers mutation parsing, wild-type-consistency validation, multi-mutant
splitting, CSV column requirements, and wild-type resolution (explicit mapping
vs. a ``wildtype`` column). The contract tests build temporary CSVs against a
**real** reference sequence (human ubiquitin) — the loader only loads and
validates structure/mutations, so the ``DMS_score`` values are opaque to it.

A real ProteinGym assay CSV is committed at
``tests/data/fixtures/variant/<assay>.csv`` (see that directory's README) and the
``test_real_proteingym_fixture`` test loads it directly.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow.csv as pa_csv
import pytest

from oplm.data.variant.loader import (
    Mutation,
    VariantAssay,
    load_clinical_variant_assays,
    load_variant_assays,
    parse_mutation,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

# Real human ubiquitin (76 aa) — used as a wild-type reference so mutations have
# real residues to validate against. Position (1-based) -> residue spot-checks:
# 1=M, 6=K, 42=R, 76=G.
_UBQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"

_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "variant"


def _write_assay_csv(
    directory: Path,
    name: str,
    rows: Sequence[tuple[str, float]],
    *,
    wildtype_col: str | None = None,
) -> Path:
    """Write a ``<name>.csv`` with ``mutant``/``DMS_score`` (+ optional ``wildtype``).

    Args:
        directory: Destination directory.
        name: Assay name (becomes the filename stem).
        rows: ``(mutant, DMS_score)`` pairs.
        wildtype_col: If given, also write a constant ``wildtype`` column.

    Returns:
        The written CSV path.
    """
    path = directory / f"{name}.csv"
    header = ["mutant", "DMS_score"] + (["wildtype"] if wildtype_col is not None else [])
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for mutant, score in rows:
            writer.writerow([mutant, score] + ([wildtype_col] if wildtype_col is not None else []))
    return path


# --------------------------------------------------------------------------- #
# parse_mutation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("A42T", Mutation("A", 42, "T")),
        ("M1A", Mutation("M", 1, "A")),
        ("G76S", Mutation("G", 76, "S")),
        (" a42t ", Mutation("A", 42, "T")),  # whitespace + lower-case tolerated
    ],
)
def test_parse_mutation_valid(token: str, expected: Mutation) -> None:
    """Well-formed tokens parse to the documented (wt, pos, mut), 1-based."""
    assert parse_mutation(token) == expected


@pytest.mark.parametrize("token", ["", "A42", "42T", "AA42T", "A42TT", "AT", "A-1T", "4242"])
def test_parse_mutation_malformed_raises(token: str) -> None:
    """Tokens that are not a single ``<wt><pos><mut>`` substitution raise."""
    with pytest.raises(ValueError, match="malformed mutation token"):
        parse_mutation(token)


def test_parse_mutation_zero_position_raises() -> None:
    """A non-positive position is rejected (positions are 1-based)."""
    with pytest.raises(ValueError, match="1-based"):
        parse_mutation("A0T")


# --------------------------------------------------------------------------- #
# load_variant_assays — contract & wild-type resolution
# --------------------------------------------------------------------------- #


def test_load_contract_with_wildtype_column(tmp_path: Path) -> None:
    """A CSV with a constant ``wildtype`` column yields the documented contract."""
    _write_assay_csv(
        tmp_path,
        "UBQ_assay",
        [("M1A", 0.12), ("K6R", -0.5), ("G76S", 1.3)],
        wildtype_col=_UBQ,
    )
    assays = load_variant_assays(tmp_path)
    assert len(assays) == 1

    assay = assays[0]
    assert isinstance(assay, VariantAssay)
    assert assay.name == "UBQ_assay"
    assert assay.wildtype == _UBQ
    assert assay.mutations == ["M1A", "K6R", "G76S"]
    assert assay.labels == pytest.approx([0.12, -0.5, 1.3])
    assert all(isinstance(x, float) for x in assay.labels)


def test_load_wildtype_from_mapping(tmp_path: Path) -> None:
    """The wild-type is resolvable from the ``wildtypes`` mapping (no column)."""
    _write_assay_csv(tmp_path, "NoCol", [("K6R", 1.0)])
    assays = load_variant_assays(tmp_path, wildtypes={"NoCol": _UBQ})
    assert assays[0].wildtype == _UBQ


def test_mapping_takes_precedence_over_column(tmp_path: Path) -> None:
    """An explicit ``wildtypes`` entry overrides a ``wildtype`` column."""
    # Column carries a truncated WT that would make K6R invalid; mapping wins.
    _write_assay_csv(tmp_path, "Both", [("K6R", 1.0)], wildtype_col=_UBQ)
    assays = load_variant_assays(tmp_path, wildtypes={"Both": _UBQ})
    assert assays[0].wildtype == _UBQ


def test_labels_coerced_to_float(tmp_path: Path) -> None:
    """Integer-looking DMS scores still load as Python floats."""
    _write_assay_csv(tmp_path, "Ints", [("M1A", 1), ("K6R", 0)], wildtype_col=_UBQ)
    labels = load_variant_assays(tmp_path)[0].labels
    assert labels == pytest.approx([1.0, 0.0])
    assert all(isinstance(x, float) for x in labels)


# --------------------------------------------------------------------------- #
# Multi-mutant handling
# --------------------------------------------------------------------------- #


def test_multi_mutant_split_and_validated(tmp_path: Path) -> None:
    """A ``:``-joined multi-mutant is kept verbatim and each part validated."""
    _write_assay_csv(tmp_path, "Multi", [("M1A:K6R:G76S", 0.7)], wildtype_col=_UBQ)
    assays = load_variant_assays(tmp_path)
    assert assays[0].mutations == ["M1A:K6R:G76S"]  # raw string preserved


def test_multi_mutant_with_bad_part_raises(tmp_path: Path) -> None:
    """A multi-mutant raises if any single substitution mismatches the WT."""
    # Position 6 is K, not A, so "A6R" is inconsistent with the wild-type.
    _write_assay_csv(tmp_path, "MultiBad", [("M1A:A6R", 0.1)], wildtype_col=_UBQ)
    with pytest.raises(ValueError, match="A6R"):
        load_variant_assays(tmp_path)


# --------------------------------------------------------------------------- #
# Validation failures
# --------------------------------------------------------------------------- #


def test_wildtype_mismatch_raises(tmp_path: Path) -> None:
    """A wild-type residue mismatch raises (position 1 is M, not A)."""
    _write_assay_csv(tmp_path, "Mismatch", [("A1G", 1.0)], wildtype_col=_UBQ)
    with pytest.raises(ValueError, match="expects wild-type"):
        load_variant_assays(tmp_path)


def test_position_out_of_range_raises(tmp_path: Path) -> None:
    """A position beyond the wild-type length raises."""
    _write_assay_csv(tmp_path, "OOR", [("M999A", 1.0)], wildtype_col=_UBQ)
    with pytest.raises(ValueError, match="out of range"):
        load_variant_assays(tmp_path)


@pytest.mark.parametrize("drop", ["mutant", "DMS_score"])
def test_missing_required_column_raises(tmp_path: Path, drop: str) -> None:
    """Dropping either required column is an error."""
    path = tmp_path / "Missing.csv"
    kept = [c for c in ("mutant", "DMS_score") if c != drop]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(kept)
        writer.writerow(["M1A" if c == "mutant" else 1.0 for c in kept])
    with pytest.raises(ValueError, match="missing required column"):
        load_variant_assays(tmp_path)


def test_missing_wildtype_source_raises(tmp_path: Path) -> None:
    """No mapping entry and no ``wildtype`` column -> error."""
    _write_assay_csv(tmp_path, "NoWT", [("M1A", 1.0)])
    with pytest.raises(ValueError, match="no wild-type sequence available"):
        load_variant_assays(tmp_path)


def test_nonconstant_wildtype_column_raises(tmp_path: Path) -> None:
    """A ``wildtype`` column with differing values per row is rejected."""
    path = tmp_path / "Varying.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mutant", "DMS_score", "wildtype"])
        writer.writerow(["M1A", 1.0, _UBQ])
        writer.writerow(["K6R", 2.0, _UBQ[:-1]])  # different sequence
    with pytest.raises(ValueError, match="single constant sequence"):
        load_variant_assays(tmp_path)


# --------------------------------------------------------------------------- #
# Directory handling
# --------------------------------------------------------------------------- #


def test_assays_sorted_by_filename(tmp_path: Path) -> None:
    """Assays are returned in deterministic (filename-sorted) order."""
    for name in ("c_assay", "a_assay", "b_assay"):
        _write_assay_csv(tmp_path, name, [("M1A", 1.0)], wildtype_col=_UBQ)
    names = [a.name for a in load_variant_assays(tmp_path)]
    assert names == ["a_assay", "b_assay", "c_assay"]


def test_max_assays_caps_count(tmp_path: Path) -> None:
    """``max_assays`` caps the number returned, after sorting."""
    for name in ("a", "b", "c"):
        _write_assay_csv(tmp_path, name, [("M1A", 1.0)], wildtype_col=_UBQ)
    assert load_variant_assays(tmp_path, max_assays=0) == []
    assert [a.name for a in load_variant_assays(tmp_path, max_assays=2)] == ["a", "b"]


def test_missing_directory_raises(tmp_path: Path) -> None:
    """A nonexistent directory raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_variant_assays(tmp_path / "does_not_exist")


# --------------------------------------------------------------------------- #
# Real ProteinGym fixture (optional)
# --------------------------------------------------------------------------- #


def _real_fixture_csvs() -> list[Path]:
    """Return the committed real ProteinGym fixture CSVs."""
    assert _FIXTURE_DIR.is_dir(), (
        f"variant fixture dir missing: {_FIXTURE_DIR} (see fixtures/README.md)"
    )
    csvs = sorted(_FIXTURE_DIR.glob("*.csv"))
    assert csvs, f"no variant fixture CSVs in {_FIXTURE_DIR} (see fixtures/README.md)"
    return csvs


def test_real_proteingym_fixture() -> None:
    """A real ProteinGym CSV loads, reconstructing the WT from ``mutated_sequence``.

    Real ProteinGym substitution CSVs carry ``mutant``, ``DMS_score``, and
    ``mutated_sequence`` columns but no standalone wild-type. The wild-type is
    reconstructed by reverting a single-substitution row's mutant residue.
    """
    path = _real_fixture_csvs()[0]
    table = pa_csv.read_csv(str(path))
    assert "mutated_sequence" in table.column_names, (
        f"{path.name} has no 'mutated_sequence' column to reconstruct the WT"
    )

    mutants = [str(m) for m in table.column("mutant").to_pylist()]
    mutated_seqs = [str(s) for s in table.column("mutated_sequence").to_pylist()]

    # Reconstruct the WT from the first single substitution: revert that residue.
    wildtype: str | None = None
    for token, seq in zip(mutants, mutated_seqs, strict=True):
        if ":" in token:
            continue
        mutation = parse_mutation(token)
        chars = list(seq)
        chars[mutation.pos - 1] = mutation.wt
        wildtype = "".join(chars)
        break
    assert wildtype is not None, f"{path.name} has no single-substitution row to reconstruct the WT"

    assays = load_variant_assays(_FIXTURE_DIR, wildtypes={path.stem: wildtype})
    assay = next(a for a in assays if a.name == path.stem)
    assert assay.wildtype == wildtype
    assert len(assay.mutations) == len(mutants)
    assert len(assay.labels) == len(assay.mutations)


# --------------------------------------------------------------------------- #
# Clinical variant loader (Pathogenic/Benign + protein_sequence WT column)
# --------------------------------------------------------------------------- #

_CLINICAL_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "variant_clinical"


def _write_clinical_csv(
    directory: Path,
    name: str,
    rows: Sequence[tuple[str, str]],
    *,
    protein_sequence: str = _UBQ,
) -> Path:
    """Write a clinical-substitution ``<name>.csv`` (mutant/protein_sequence/DMS_bin_score).

    Args:
        directory: Destination directory.
        name: Assay name (becomes the filename stem).
        rows: ``(mutant, DMS_bin_score)`` pairs, e.g. ``("K6R", "Pathogenic")``.
        protein_sequence: Constant wild-type written to every row.

    Returns:
        The written CSV path.
    """
    path = directory / f"{name}.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mutant", "protein_sequence", "DMS_bin_score"])
        for mutant, label in rows:
            writer.writerow([mutant, protein_sequence, label])
    return path


def test_clinical_labels_mapped_to_floats(tmp_path: Path) -> None:
    """Pathogenic/Benign map to 1.0/0.0 (case-insensitive) and WT comes from the column."""
    _write_clinical_csv(
        tmp_path,
        "ACAD",
        [("M1A", "Pathogenic"), ("K6R", "benign"), ("G76S", "PATHOGENIC")],
    )
    assays = load_clinical_variant_assays(tmp_path)
    assert len(assays) == 1

    assay = assays[0]
    assert isinstance(assay, VariantAssay)
    assert assay.name == "ACAD"
    assert assay.wildtype == _UBQ
    assert assay.mutations == ["M1A", "K6R", "G76S"]
    assert assay.labels == pytest.approx([1.0, 0.0, 1.0])
    assert all(isinstance(x, float) for x in assay.labels)


def test_clinical_unknown_label_raises(tmp_path: Path) -> None:
    """A label outside the Pathogenic/Benign vocabulary raises."""
    _write_clinical_csv(tmp_path, "Bad", [("K6R", "Likely_pathogenic")])
    with pytest.raises(ValueError, match="unknown clinical label"):
        load_clinical_variant_assays(tmp_path)


@pytest.mark.parametrize("drop", ["mutant", "protein_sequence", "DMS_bin_score"])
def test_clinical_missing_required_column_raises(tmp_path: Path, drop: str) -> None:
    """Dropping any required clinical column is an error."""
    path = tmp_path / "Missing.csv"
    all_cols = {"mutant": "M1A", "protein_sequence": _UBQ, "DMS_bin_score": "Benign"}
    kept = [c for c in all_cols if c != drop]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(kept)
        writer.writerow([all_cols[c] for c in kept])
    # Dropping the WT column surfaces as a "no wild-type" error; the others as
    # a missing-required-column error.
    expected = (
        "no wild-type sequence available"
        if drop == "protein_sequence"
        else "missing required column"
    )
    with pytest.raises(ValueError, match=expected):
        load_clinical_variant_assays(tmp_path)


def test_clinical_wildtype_mismatch_raises(tmp_path: Path) -> None:
    """Mutation validation against the protein_sequence WT is reused (position 1 is M)."""
    _write_clinical_csv(tmp_path, "Mismatch", [("A1G", "Pathogenic")])
    with pytest.raises(ValueError, match="expects wild-type"):
        load_clinical_variant_assays(tmp_path)


def test_clinical_position_out_of_range_raises(tmp_path: Path) -> None:
    """A position beyond the WT length raises."""
    _write_clinical_csv(tmp_path, "OOR", [("M999A", "Benign")])
    with pytest.raises(ValueError, match="out of range"):
        load_clinical_variant_assays(tmp_path)


def test_clinical_multi_mutant_kept_and_validated(tmp_path: Path) -> None:
    """A ``:``-joined multi-mutant is kept verbatim and each part validated."""
    _write_clinical_csv(tmp_path, "Multi", [("M1A:K6R", "Pathogenic")])
    assays = load_clinical_variant_assays(tmp_path)
    assert assays[0].mutations == ["M1A:K6R"]


def test_clinical_mapping_overrides_protein_sequence(tmp_path: Path) -> None:
    """An explicit ``wildtypes`` entry takes precedence over the protein_sequence column."""
    _write_clinical_csv(tmp_path, "Both", [("K6R", "Benign")], protein_sequence="MAAA")
    assays = load_clinical_variant_assays(tmp_path, wildtypes={"Both": _UBQ})
    assert assays[0].wildtype == _UBQ


def test_clinical_max_assays_caps_count(tmp_path: Path) -> None:
    """``max_assays`` caps the number returned, after filename sorting."""
    for name in ("a", "b", "c"):
        _write_clinical_csv(tmp_path, name, [("M1A", "Benign")])
    assert [a.name for a in load_clinical_variant_assays(tmp_path, max_assays=2)] == ["a", "b"]


def test_real_clinical_fixture() -> None:
    """The committed real clinical fixture loads with both classes present."""
    assert _CLINICAL_FIXTURE_DIR.is_dir(), (
        f"clinical fixture dir missing: {_CLINICAL_FIXTURE_DIR} (see fixtures/README.md)"
    )
    csvs = sorted(_CLINICAL_FIXTURE_DIR.glob("*.csv"))
    assert csvs, f"no clinical fixture CSVs in {_CLINICAL_FIXTURE_DIR}"

    assays = load_clinical_variant_assays(_CLINICAL_FIXTURE_DIR)
    assert assays
    assay = assays[0]
    assert set(assay.labels) == {0.0, 1.0}  # both Pathogenic and Benign present
    assert len(assay.mutations) == len(assay.labels)
