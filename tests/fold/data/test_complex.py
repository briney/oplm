"""The flat all-atom representation and its Parquet row (spec §4.2, §4.7)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from oplm.fold.data.complex import (
    COMPLEX_SCHEMA,
    ChainInfo,
    ComplexMetadata,
    MolecularComplex,
    complex_to_row,
    row_to_complex,
    rows_to_table,
)


def two_chain_complex() -> MolecularComplex:
    """Chain A: GLY (4 atoms), ALA (5 atoms); chain B: GLY (4 atoms). 13 atoms, 3 tokens."""
    counts = np.array([4, 5, 4], dtype=np.int32)
    starts = np.array([0, 4, 9], dtype=np.int32)
    names = ["N", "CA", "C", "O", "N", "CA", "C", "O", "CB", "N", "CA", "C", "O"]
    return MolecularComplex(
        atom_coords=np.arange(39, dtype=np.float32).reshape(13, 3),
        atom_resolved=np.array([True] * 8 + [False] + [True] * 4),
        atom_element=np.array([7, 6, 6, 8, 7, 6, 6, 8, 6, 7, 6, 6, 8], dtype=np.uint8),
        atom_name=names,
        atom_charge=np.zeros(13, dtype=np.int8),
        atom_bfactor=np.full(13, 20.0, dtype=np.float32),
        atom_to_token=np.array([0] * 4 + [1] * 5 + [2] * 4, dtype=np.int32),
        token_res_type=np.array([9, 2, 9], dtype=np.int8),
        token_mol_type=np.zeros(3, dtype=np.int8),
        token_asym_id=np.array([0, 0, 1], dtype=np.int32),
        token_entity_id=np.array([0, 0, 1], dtype=np.int32),
        token_sym_id=np.zeros(3, dtype=np.int32),
        token_residue_index=np.array([0, 1, 0], dtype=np.int32),
        token_center_atom=starts + 1,
        token_rep_atom=np.array([1, 8, 10], dtype=np.int32),
        token_atom_start=starts,
        token_atom_count=counts,
        bonds=np.zeros((0, 2), dtype=np.int32),
        chains=[
            ChainInfo("A", "A", 0, 0, 0, "GA"),
            ChainInfo("B", "B", 1, 0, 0, "G", mean_plddt=0.5),
        ],
        metadata=ComplexMetadata(
            "test", "pdb", "X-RAY DIFFRACTION", 1.5, "2000-01-01", {"water": 1}
        ),
    )


def assert_complexes_equal(a: MolecularComplex, b: MolecularComplex) -> None:
    for f in dataclasses.fields(MolecularComplex):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if isinstance(x, np.ndarray):
            assert x.dtype == y.dtype, f.name
            assert np.array_equal(x, y), f.name
        else:
            assert x == y, f.name


def test_validate_accepts_a_consistent_complex() -> None:
    c = two_chain_complex()
    c.validate()
    assert c.num_atoms == 13 and c.num_tokens == 3
    assert c.chain_tokens(0).tolist() == [0, 1] and c.chain_tokens(1).tolist() == [2]


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("atom_to_token", np.array([0] * 13, dtype=np.int32), "atom_to_token"),
        ("token_atom_start", np.array([0, 5, 9], dtype=np.int32), "contiguous"),
        ("token_rep_atom", np.array([1, 8, 3], dtype=np.int32), "representative"),
        ("token_asym_id", np.array([0, 0, 2], dtype=np.int32), "asym_id"),
        ("atom_name", ["N"] * 12, "atom count"),
    ],
)
def test_validate_rejects_inconsistent_indices(field: str, value: object, match: str) -> None:
    c = dataclasses.replace(two_chain_complex(), **{field: value})
    with pytest.raises(ValueError, match=match):
        c.validate()


def test_select_tokens_remaps_atoms_and_compacts_chains() -> None:
    c = two_chain_complex()
    sub = c.select_tokens(np.array([2, 1]))  # unsorted input: result is in token order
    sub.validate()
    assert sub.num_tokens == 2 and sub.num_atoms == 9
    assert sub.token_atom_start.tolist() == [0, 5] and sub.token_atom_count.tolist() == [5, 4]
    assert sub.atom_name == c.atom_name[4:13]
    assert np.array_equal(sub.atom_coords, c.atom_coords[4:13])
    assert sub.token_center_atom.tolist() == [1, 6] and sub.token_rep_atom.tolist() == [4, 6]
    assert sub.token_asym_id.tolist() == [0, 1]  # both chains still present
    assert sub.token_residue_index.tolist() == [1, 0]  # original identities preserved
    only_b = c.select_tokens(np.array([2]))
    assert only_b.token_asym_id.tolist() == [0] and [ch.chain_id for ch in only_b.chains] == ["B"]
    assert only_b.token_entity_id.tolist() == [1] and only_b.chains[0].entity_id == 1
    assert only_b.metadata is c.metadata


def test_arrow_row_round_trip(tmp_path: Path) -> None:
    c = two_chain_complex()
    table = rows_to_table([complex_to_row(c), complex_to_row(c.select_tokens(np.array([2])))])
    assert table.schema.equals(COMPLEX_SCHEMA) and table.num_rows == 2
    pq.write_table(table, tmp_path / "shard.parquet", row_group_size=1)
    back = pq.read_table(tmp_path / "shard.parquet")
    assert_complexes_equal(row_to_complex(back.slice(0, 1).to_pylist()[0]), c)
    assert_complexes_equal(
        row_to_complex(back.slice(1, 1).to_pylist()[0]), c.select_tokens(np.array([2]))
    )


def test_round_trip_keeps_none_resolution_and_plddt() -> None:
    c = two_chain_complex()
    c.metadata = ComplexMetadata("nmr", "pdb", "SOLUTION NMR", None, None, {})
    back = row_to_complex(rows_to_table([complex_to_row(c)]).to_pylist()[0])
    assert back.metadata.resolution is None and back.metadata.release_date is None
    assert back.chains[0].mean_plddt is None and back.chains[1].mean_plddt == 0.5


def test_select_tokens_empty_selection() -> None:
    c = two_chain_complex()
    empty = c.select_tokens(np.array([], dtype=np.int64))
    empty.validate()
    assert empty.num_tokens == 0 and empty.num_atoms == 0 and empty.chains == []
    assert empty.bonds.shape == (0, 2)
