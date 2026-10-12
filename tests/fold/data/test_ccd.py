"""Residue vocabulary, atom-name encoding and the reference conformer table."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from oplm.fold.data.ccd import (
    LM_UNKNOWN_ID,
    NUM_RES_TYPES,
    PROTEIN_1TO3,
    PROTEIN_3TO1,
    PROTEIN_RES_TYPES,
    RES_TYPE_TO_CODE,
    UNK_RES_TYPE,
    ChemCompTable,
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
    assert table.source.startswith("esm 3.4.1.post1 get_idealized_atom_pos")  # upstream CCD dump
    assert torch.isfinite(table["UNK"].positions).all()  # the CCD UNK conformer, as upstream uses


CCD_SUBSET = Path(__file__).parents[2] / "fixtures" / "fold" / "ccd_subset.cif"


def test_chem_comp_table_parses_parents_types_and_letters() -> None:
    table = ChemCompTable.parse(CCD_SUBSET)
    assert len(table) == 45 and "MSE" in table and "ZZZ" not in table
    assert table["MSE"].parent == "MET" and table["MSE"].one_letter == "M"
    assert table["MSE"].type == "L-PEPTIDE LINKING"
    assert table["0YG"].type == "L-PEPTIDE NH3 AMINO TERMINUS"  # upper-cased
    assert table["0YG"].parent is None and table["175"].parent is None  # multi-parent
    assert table["ALA"].parent is None and table["HOH"].parent is None
    assert table["HOH"].one_letter is None
    assert table.source.startswith("ccd_subset.cif")


@pytest.mark.parametrize(
    ("code", "parent"),
    [
        ("ALA", "ALA"),  # standard: itself
        ("UNK", "UNK"),
        ("MSE", "MET"),
        ("SEP", "SER"),
        ("NLE", "LEU"),
        ("PCA", "GLN"),
        ("HYP", "PRO"),
        ("0YG", "UNK"),  # multi-parent
        ("DAL", "UNK"),  # D-amino acid, no parent
        ("HOH", "UNK"),  # not a peptide at all
        ("ZZZ", "UNK"),  # unknown code
    ],
)
def test_protein_parent(code: str, parent: str) -> None:
    assert ChemCompTable.parse(CCD_SUBSET).protein_parent(code) == parent


def test_is_peptide() -> None:
    table = ChemCompTable.parse(CCD_SUBSET)
    assert table.is_peptide("MSE") and table.is_peptide("DAL") and table.is_peptide("GLY")
    assert (
        not table.is_peptide("HOH") and not table.is_peptide("DA") and not table.is_peptide("ZZZ")
    )


def test_chem_comp_table_json_round_trip(tmp_path: Path) -> None:
    table = ChemCompTable.parse(CCD_SUBSET)
    table.save(tmp_path / "ccd.json")
    loaded = ChemCompTable.load(tmp_path / "ccd.json")
    assert len(loaded) == len(table) and loaded.source == table.source
    assert all(loaded[c] == table[c] for c in ("MSE", "0YG", "HOH", "ALA"))


def test_inverse_tables() -> None:
    assert PROTEIN_3TO1["MET"] == "M" and PROTEIN_3TO1["UNK"] == "X" and len(PROTEIN_3TO1) == 21
    assert (
        RES_TYPE_TO_CODE[2] == "ALA"
        and RES_TYPE_TO_CODE[22] == "UNK"
        and len(RES_TYPE_TO_CODE) == 21
    )
