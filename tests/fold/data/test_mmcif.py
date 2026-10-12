"""wwPDB mmCIF -> MolecularComplex on real entries (spec §4.2, §4.4; addendum §4)."""

from __future__ import annotations

import gzip
from typing import TYPE_CHECKING

import gemmi
import numpy as np
import pytest

from oplm.fold.data.ccd import PROTEIN_RES_TYPES
from oplm.fold.data.mmcif import Rejection, parse_mmcif
from tests.fold.data.conftest import MMCIF_DIR

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from oplm.fold.data.ccd import ChemCompTable, ReferenceConformers
    from oplm.fold.data.complex import MolecularComplex

CRAMBIN = "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN"
# 1ejg's entity sequence carries the first alternative of the LEU,ILE site at 25: L, not 1crn's I.
CRAMBIN_1EJG = CRAMBIN[:24] + "L" + CRAMBIN[25:]


def token_atoms(c: MolecularComplex, token: int) -> slice:
    start = int(c.token_atom_start[token])
    return slice(start, start + int(c.token_atom_count[token]))


def test_monomer_1crn(parsed: Callable[[str], MolecularComplex]) -> None:
    c = parsed("1crn")
    c.validate()
    m = c.metadata
    assert m.id == "1crn" and m.source == "pdb" and m.method == "X-RAY DIFFRACTION"
    assert m.resolution == pytest.approx(1.5) and m.release_date == "1981-07-28"
    assert m.dropped == {}
    assert len(c.chains) == 1 and c.chains[0].chain_id == "A" and c.chains[0].auth_chain_id == "A"
    assert (
        c.chains[0].sequence == CRAMBIN and c.chains[0].entity_id == 0 and c.chains[0].sym_id == 0
    )
    assert c.num_tokens == 46 and c.atom_resolved.all() and c.token_mol_type.tolist() == [0] * 46
    assert c.token_residue_index.tolist() == list(range(46))
    assert c.atom_name[:4] == ["N", "CA", "C", "O"] and c.token_center_atom[0] == 1
    assert np.allclose(c.atom_coords[0], [17.047, 14.099, 3.625], atol=1e-3)  # THR 1 N
    assert c.token_rep_atom[0] == 4  # THR CB
    assert c.token_rep_atom[19] == c.token_atom_start[19] + 1  # GLY 20 -> CA
    assert c.token_res_type[19] == PROTEIN_RES_TYPES["GLY"] and c.token_atom_count[19] == 4
    assert c.bonds.shape == (0, 2) and c.atom_bfactor[0] > 0
    assert c.atom_element[c.token_atom_start[2] + 5] == 16  # CYS 3 SG


def test_altlocs_hydrogens_and_microheterogeneity_1ejg(
    parsed: Callable[[str], MolecularComplex],
) -> None:
    c = parsed("1ejg")
    c.validate()
    assert c.chains[0].sequence == CRAMBIN_1EJG and c.metadata.resolution == pytest.approx(0.54)
    assert not np.any(c.atom_element == 1)  # hydrogens dropped
    # Highest-occupancy altloc per atom: THR 1 N has A (0.82) and B (0.18).
    st = gemmi.read_structure(str(MMCIF_DIR / "1ejg.cif.gz"))
    thr1 = st[0]["A"][0]
    alt_a = next(a for a in thr1 if a.name == "N" and a.altloc == "A")
    assert np.allclose(c.atom_coords[0], [alt_a.pos.x, alt_a.pos.y, alt_a.pos.z], atol=1e-4)
    assert c.atom_bfactor[0] == pytest.approx(alt_a.b_iso)
    # Microheterogeneity keeps the entity sequence's first alternative.
    assert c.token_res_type[21] == PROTEIN_RES_TYPES["PRO"] and c.token_atom_count[21] == 7
    assert c.token_res_type[24] == PROTEIN_RES_TYPES["LEU"] and c.token_atom_count[24] == 8
    assert c.atom_resolved.all()


def test_entities_copies_and_dropped_4hhb(parsed: Callable[[str], MolecularComplex]) -> None:
    c = parsed("4hhb")
    c.validate()
    assert [ch.chain_id for ch in c.chains] == ["A", "B", "C", "D"]
    assert [ch.entity_id for ch in c.chains] == [0, 1, 0, 1]
    assert [ch.sym_id for ch in c.chains] == [0, 0, 1, 1]
    assert [len(ch.sequence) for ch in c.chains] == [141, 146, 141, 146]
    assert c.chains[0].sequence.startswith("VLSPADKTNV") and c.chains[1].sequence.startswith(
        "VHLTPEEKSA"
    )
    assert c.metadata.dropped == {"ligand": 2, "water": 1}  # HEM, PO4 entities; water
    assert c.num_tokens == 574 and c.metadata.resolution == pytest.approx(1.74)
    assert c.chain_tokens(2).tolist() == list(range(287, 428))
    assert np.array_equal(c.token_entity_id[c.chain_tokens(2)], np.zeros(141, dtype=np.int32))
    assert np.array_equal(c.token_sym_id[c.chain_tokens(2)], np.ones(141, dtype=np.int32))


def test_unresolved_residues_and_selenomethionine_2ds7(
    parsed: Callable[[str], MolecularComplex],
) -> None:
    c = parsed("2ds7")
    c.validate()
    assert len(c.chains) == 1 and len(c.chains[0].sequence) == 51 and c.num_tokens == 51
    for token in range(11):  # residues 1-11 are not modeled: tokens exist, coordinates masked
        atoms = token_atoms(c, token)
        assert not c.atom_resolved[atoms].any()
        assert not c.atom_coords[atoms].any() and not c.atom_bfactor[atoms].any()
    assert c.atom_resolved[token_atoms(c, 11)].any()
    mse = 42  # label_seq 43
    assert c.token_res_type[mse] == PROTEIN_RES_TYPES["MET"] and c.chains[0].sequence[mse] == "M"
    names = c.atom_name[token_atoms(c, mse)]
    assert names == ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"] and "SE" not in names
    assert c.atom_resolved[token_atoms(c, mse)].all()  # SE matched SD
    assert c.atom_element[c.token_atom_start[mse] + 6] == 16  # template element (S), not Se


def test_homodimer_norleucine_and_partial_chain_1jb6(
    parsed: Callable[[str], MolecularComplex],
) -> None:
    c = parsed("1jb6")
    c.validate()
    assert [ch.chain_id for ch in c.chains] == ["A", "B"]
    assert [ch.entity_id for ch in c.chains] == [0, 0] and [ch.sym_id for ch in c.chains] == [0, 1]
    assert c.chains[0].sequence == c.chains[1].sequence and len(c.chains[0].sequence) == 33
    # NLE collapses to LEU: CB and CG match by name, CD1/CD2 do not exist in norleucine.
    st = gemmi.read_structure(str(MMCIF_DIR / "1jb6.cif.gz"))
    nle_seq = next(r.label_seq for r in st[0]["A"].subchains()[0] if r.name == "NLE")
    nle = int(np.flatnonzero((c.token_asym_id == 0) & (c.token_residue_index == nle_seq - 1))[0])
    assert (
        c.token_res_type[nle] == PROTEIN_RES_TYPES["LEU"]
        and c.chains[0].sequence[nle_seq - 1] == "L"
    )
    names = c.atom_name[token_atoms(c, nle)]
    resolved = c.atom_resolved[token_atoms(c, nle)].tolist()
    assert dict(zip(names, resolved, strict=True)) == {
        "N": True,
        "CA": True,
        "C": True,
        "O": True,
        "CB": True,
        "CG": True,
        "CD1": False,
        "CD2": False,
    }
    b = c.chain_tokens(1)
    unresolved_b = [
        int(c.token_residue_index[t]) for t in b if not c.atom_resolved[token_atoms(c, t)].any()
    ]
    assert unresolved_b == [0, 1, 30, 31, 32]


def test_nmr_first_model_1l2y(parsed: Callable[[str], MolecularComplex]) -> None:
    c = parsed("1l2y")
    c.validate()
    assert c.metadata.method == "SOLUTION NMR" and c.metadata.resolution is None
    assert c.num_tokens == 20 and c.atom_resolved.all()
    st = gemmi.read_structure(str(MMCIF_DIR / "1l2y.cif.gz"))
    first, second = st[0]["A"][0][0].pos, st[1]["A"][0][0].pos  # first atom of models 0 and 1
    xyz = np.asarray(c.atom_coords[0])
    assert np.allclose(xyz, [first.x, first.y, first.z], atol=1e-4)
    assert not np.allclose(xyz, [second.x, second.y, second.z], atol=1e-3)


def test_resolution_and_short_chain_filters(
    ccd_table: ChemCompTable, conformers: ReferenceConformers
) -> None:
    hhb = MMCIF_DIR / "4hhb.cif.gz"
    rej = parse_mmcif(hhb, ccd=ccd_table, conformers=conformers, max_resolution=1.0)
    assert isinstance(rej, Rejection) and rej.id == "4hhb" and rej.reason == "resolution"
    c = parse_mmcif(
        MMCIF_DIR / "1jb6.cif.gz", ccd=ccd_table, conformers=conformers, min_resolved_residues=30
    )
    assert not isinstance(c, Rejection)
    assert [ch.chain_id for ch in c.chains] == ["A"] and c.metadata.dropped == {
        "water": 1,
        "short_chain": 1,
    }
    rej = parse_mmcif(
        MMCIF_DIR / "1crn.cif.gz", ccd=ccd_table, conformers=conformers, min_resolved_residues=1000
    )
    assert isinstance(rej, Rejection) and rej.reason == "no_resolved_chain"


def test_no_protein_chains_and_parse_error(
    tmp_path: Path, ccd_table: ChemCompTable, conformers: ReferenceConformers
) -> None:
    st = gemmi.read_structure(str(MMCIF_DIR / "1crn.cif.gz"))
    st.setup_entities()
    st.entities[0].polymer_type = gemmi.PolymerType.Dna
    dna = tmp_path / "dna.cif"
    st.make_mmcif_document().write_file(str(dna))
    rej = parse_mmcif(dna, ccd=ccd_table, conformers=conformers)
    assert isinstance(rej, Rejection) and rej.reason == "no_protein_chains"
    bad = tmp_path / "bad.cif.gz"
    with gzip.open(bad, "wt") as fh:
        fh.write("data_bad\n_entry.id bad\nloop_ broken\n")
    rej = parse_mmcif(bad, ccd=ccd_table, conformers=conformers)
    assert isinstance(rej, Rejection) and rej.reason == "parse_error" and rej.id == "bad.cif.gz"
