"""CASP domain tarballs -> eval chains on the full target sequence."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from oplm.fold.evalsets import casp
from tests.fold.evalsets.conftest import CRAMBIN, GAP, crambin_domain, make_tarball

if TYPE_CHECKING:
    from pathlib import Path


def test_domains_sit_on_the_full_target_sequence(tmp_path: Path) -> None:
    """A gapped domain keeps all 46 residues; the five unmodeled ones have no atoms."""
    tarball = make_tarball(tmp_path / "d.tgz", {"T0201-D1.pdb": crambin_domain(GAP)})
    (chain,), stats = casp.casp_domain_chains(tarball, {"T1201": CRAMBIN}, "16")
    assert chain.name == "casp16_T1201-D1" and chain.sequence == CRAMBIN
    assert chain.modeled == 41
    assert [i for i, residue in enumerate(chain.atoms, 1) if not residue] == sorted(GAP)
    assert chain.meta == {
        "source": "casp16",
        "entry": "T1201",
        "chain": "D1",
        "source_file": "T0201-D1.pdb",
    }
    assert stats["files"] == 1


def test_duplicates_versions_combined_units_and_mismatches_are_skipped(tmp_path: Path) -> None:
    domain = crambin_domain()
    files = {
        "T0201-D1.pdb": domain,
        "T2201-D1.pdb": domain,  # phase-2 copy of T1201-D1
        "T1228v1-D1.pdb": domain,
        "T1228v2-D1.pdb": domain,  # later version of the same domain
        "T1300-D12.pdb": domain,  # combined unit
        "T1301-D1.pdb": domain,  # no sequence
        "T1302-D1.pdb": crambin_domain(renumber=100),  # numbering outside the sequence
    }
    sequences = {name: CRAMBIN for name in ("T1201", "T1228v1", "T1300", "T1302")}
    chains, stats = casp.casp_domain_chains(
        make_tarball(tmp_path / "d.tgz", files), sequences, "16"
    )
    assert [c.name for c in chains] == ["casp16_T1201-D1", "casp16_T1228v1-D1"]
    assert stats == {
        "files": 7,
        "multi_domain": 1,
        "duplicate": 2,
        "no_sequence": 1,
        "numbering_mismatch": 1,
    }


def test_hetatm_and_altloc_lines_are_ignored(tmp_path: Path) -> None:
    """Waters and a second alternate location don't break placement or add atoms."""
    lines = crambin_domain().decode().splitlines()
    first_ca = next(line for line in lines if line[12:16] == " CA ")
    altloc_b = f"{first_ca[:16]}B{first_ca[17:30]}{99.0:>8.3f}{first_ca[38:]}"
    water = "HETATM  999  O   HOH A 101      10.000  10.000  10.000  1.00 20.00           O"
    data = "\n".join([*lines[:-1], altloc_b, water, "END"]).encode()
    tarball = make_tarball(tmp_path / "d.tgz", {"T1104-D1.pdb": data})
    (chain,), _ = casp.casp_domain_chains(tarball, {"T1104": CRAMBIN}, "15")
    assert chain.modeled == 46
    assert chain.atoms[0]["CA"][0] != pytest.approx(99.0)


def test_member_paths_are_flattened(tmp_path: Path) -> None:
    """CASP15 members are stored as ``./T1104-D1.pdb``."""
    tarball = make_tarball(tmp_path / "d.tgz", {"./T1104-D1.pdb": crambin_domain()})
    (chain,), _ = casp.casp_domain_chains(tarball, {"T1104": CRAMBIN}, "15")
    assert chain.name == "casp15_T1104-D1"


def test_canonical_target() -> None:
    assert casp.canonical_target("T0206") == "T1206"
    assert casp.canonical_target("T2201") == "T1201"
    assert casp.canonical_target("T1228v2") == "T1228v2"
    assert casp.canonical_target("T1104") == "T1104"


def test_read_fasta(tmp_path: Path) -> None:
    path = tmp_path / "s.fasta"
    path.write_text(">T1201 Q9GZX9, Human, 8 residues\nACDE\nFGHI\n>T1202\nKL\n")
    assert casp.read_fasta(path) == {"T1201": "ACDEFGHI", "T1202": "KL"}


def test_verify_casp_sources(casp_sources: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert len(casp.verify_casp_sources(casp_sources)) == 4
    name = next(iter(casp.CASP_SHA256))
    monkeypatch.setitem(casp.CASP_SHA256, name, "0" * 64)
    with pytest.raises(ValueError, match="sha256"):
        casp.verify_casp_sources(casp_sources)
    (casp_sources / name).unlink()
    with pytest.raises(FileNotFoundError, match="missing CASP source"):
        casp.verify_casp_sources(casp_sources)
