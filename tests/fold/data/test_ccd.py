"""Residue vocabulary, atom-name encoding and the reference conformer table."""

from __future__ import annotations

import pytest
import torch

from oplm.fold.data.ccd import (
    LM_UNKNOWN_ID,
    NUM_RES_TYPES,
    PROTEIN_1TO3,
    PROTEIN_RES_TYPES,
    UNK_RES_TYPE,
    ReferenceConformers,
    encode_atom_name,
)
from oplm.model.tokenization_oplm import VOCAB


def test_res_type_ids_are_alphabetical_three_letter_codes_from_two() -> None:
    assert PROTEIN_RES_TYPES["ALA"] == 2 and PROTEIN_RES_TYPES["VAL"] == 21
    assert PROTEIN_RES_TYPES["UNK"] == UNK_RES_TYPE == 22
    assert len(PROTEIN_RES_TYPES) == 21 and NUM_RES_TYPES == 33
    assert PROTEIN_1TO3["M"] == "MET" and PROTEIN_1TO3["X"] == "UNK"
    assert VOCAB["X"] == LM_UNKNOWN_ID == 24  # the LM id ESMFold2 gives X/unknown residues


@pytest.mark.parametrize(
    ("name", "codes"),
    [
        ("CA", (35, 33, 0, 0)),
        ("N", (46, 0, 0, 0)),
        ("OXT", (47, 56, 52, 0)),
        ("CD1", (35, 36, 17, 0)),
    ],
)
def test_encode_atom_name_is_left_aligned_ord_minus_32(name: str, codes: tuple[int, ...]) -> None:
    assert encode_atom_name(name) == codes


def test_reference_conformers_table() -> None:
    table = ReferenceConformers.load()
    assert set(table.codes) == set(PROTEIN_RES_TYPES)
    assert sum(len(table[c].atoms) for c in table.codes) == 171
    trp = table["TRP"]
    assert trp.atoms[:5] == ("N", "CA", "C", "O", "CB") and len(trp.atoms) == 14
    assert trp.positions.shape == (14, 3) and trp.positions.dtype == torch.float32
    assert trp.elements[:5] == (7, 6, 6, 8, 6)
    assert table["GLY"].representative_atom == 1  # CA when there is no CB
    assert table["ALA"].representative_atom == 4  # CB
    assert table["LYS"].charges[table["LYS"].atoms.index("NZ")] == 1
    assert table["ARG"].charges[table["ARG"].atoms.index("NH2")] == 1
    assert table["HIS"].charges[table["HIS"].atoms.index("ND1")] == 1
    assert table["UNK"].atoms == ("N", "CA", "C", "O")
    assert torch.equal(table["UNK"].positions, torch.zeros(4, 3))
