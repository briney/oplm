"""CAMEO and PDB40 chains from the preprocessed fixture dataset."""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import pyarrow.parquet as pq
import pytest

from oplm.fold.evalsets import pdb

if TYPE_CHECKING:
    from pathlib import Path

_RULES = {"min_length": 20, "max_length": 1024, "min_modeled_fraction": 0.7}


def _sequence(dataset: Path, entry: str, chain: str) -> str:
    rows = pq.read_table(dataset / "index.parquet", columns=["id", "chain_id", "sequence"])
    (row,) = [r for r in rows.to_pylist() if (r["id"], r["chain_id"]) == (entry, chain)]
    return row["sequence"]


def _clusters(dataset: Path) -> dict[str, str]:
    """Sequence -> cluster: the crambins together, everything else alone."""
    groups = {
        "crambin": [("1crn", "A"), ("1ejg", "A")],
        "zipper": [("1jb6", "A")],
        "2ds7": [("2ds7", "A")],
        "alpha": [("4hhb", "A")],
        "beta": [("4hhb", "B")],
    }
    return {
        _sequence(dataset, entry, chain): cluster
        for cluster, members in groups.items()
        for entry, chain in members
    }


@pytest.mark.parametrize(
    ("method", "resolution", "expected"),
    [
        ("X-RAY DIFFRACTION", 2.5, True),
        ("X-RAY DIFFRACTION", 2.51, False),
        ("ELECTRON MICROSCOPY", 3.0, True),
        ("ELECTRON MICROSCOPY", 3.2, False),
        ("SOLUTION NMR", None, False),
        ("X-RAY DIFFRACTION", None, False),
    ],
)
def test_passes_quality(method: str, resolution: float | None, expected: bool) -> None:
    assert pdb.passes_quality(method, resolution) is expected


def test_cameo_chains(fixture_dataset: Path) -> None:
    chains, stats = pdb.cameo_chains(fixture_dataset)
    assert [c.name for c in chains] == ["1crn_A"]
    assert stats == {"listed": 1, "missing": 0, "skipped": 0, "quality": 0}
    assert chains[0].meta["method"] == "X-RAY DIFFRACTION"
    assert chains[0].meta["resolution"] == pytest.approx(1.5)
    assert chains[0].meta["chain"] == "A"


def test_write_cluster_fasta(fixture_dataset: Path, tmp_path: Path) -> None:
    """Six distinct sequences of length >= 30 (trp-cage is too short), sorted, stable ids."""
    ids = pdb.write_cluster_fasta(fixture_dataset, tmp_path / "pdb40.fasta")
    assert list(ids) == [f"s{i:06d}" for i in range(6)]
    assert list(ids.values()) == sorted(ids.values())
    assert min(len(s) for s in ids.values()) >= pdb.CLUSTER_MIN_LENGTH
    text = (tmp_path / "pdb40.fasta").read_text()
    assert text.startswith(">s000000\n") and text.count(">") == 6


def test_read_clusters(tmp_path: Path) -> None:
    tsv = tmp_path / "c.tsv"
    tsv.write_text("s0\ts0\ns0\ts2\ns1\ts1\n")
    assert pdb.read_clusters(tsv) == {"s0": "s0", "s2": "s0", "s1": "s1"}


def test_pdb40_takes_the_best_passing_member(fixture_dataset: Path) -> None:
    """1ejg (0.54 Å) beats 1crn (1.5 Å); the zipper and 2ds7 clusters fail on contacts."""
    chains, stats = pdb.pdb40_chains(fixture_dataset, _clusters(fixture_dataset), **_RULES)
    assert [c.name for c in chains] == ["1ejg_A", "4hhb_A", "4hhb_B"]
    assert stats == {
        "clusters": 5,
        "no_candidate": 0,
        "contacts": 3,
        "exhausted": 2,
        "max_rounds": 0,
    }
    assert chains[0].meta["cluster"] == "crambin"


def test_pdb40_stops_after_max_rounds(fixture_dataset: Path) -> None:
    """With one round, the zipper's second chain is never tried."""
    _, stats = pdb.pdb40_chains(fixture_dataset, _clusters(fixture_dataset), **_RULES, max_rounds=1)
    assert stats["max_rounds"] == 1 and stats["exhausted"] == 1 and stats["contacts"] == 2


def test_run_mmseqs_with_a_stand_in(fake_mmseqs: Path, tmp_path: Path) -> None:
    fasta = tmp_path / "in.fasta"
    fasta.write_text(">s0\nTTCCPSIVARSNF\n>s1\nTTCCPSIVARQQQ\n>s2\nVLSPADKTNVKAA\n")
    tsv = pdb.run_mmseqs(fasta, tmp_path, mmseqs=str(fake_mmseqs))
    assert pdb.read_clusters(tsv) == {"s0": "s0", "s1": "s0", "s2": "s2"}
    assert pdb.mmseqs_version(str(fake_mmseqs)) == "fake-mmseqs"


@pytest.mark.skipif(shutil.which("mmseqs") is None, reason="needs MMseqs2 on PATH")
def test_run_mmseqs_reads_protein_when_the_fasta_opens_with_poly_a(tmp_path: Path) -> None:
    """write_cluster_fasta sorts sequences, so the real file opens with poly-A. MMseqs2 guessed
    nucleotide from it, found no hits and left 156k of 163k clusters as singletons."""
    lysozyme = (
        "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALL"
        "SSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"
    )
    poly_a = "".join(f">a{i}\n{'A' * (30 + i)}\n" for i in range(10))
    fasta = tmp_path / "in.fasta"
    fasta.write_text(f"{poly_a}>p1\n{lysozyme}\n>p2\n{lysozyme.replace('KVFGRC', 'KVFGRA')}\n")
    clusters = pdb.read_clusters(pdb.run_mmseqs(fasta, tmp_path))
    assert clusters["p1"] == clusters["p2"]


def test_run_mmseqs_reports_a_missing_binary(tmp_path: Path) -> None:
    fasta = tmp_path / "in.fasta"
    fasta.write_text(">s0\nACDE\n")
    with pytest.raises(RuntimeError, match="no-such-mmseqs"):
        pdb.run_mmseqs(fasta, tmp_path, mmseqs=str(tmp_path / "no-such-mmseqs"))


def test_mmseqs_version_reports_a_missing_binary(tmp_path: Path) -> None:
    """mmseqs_version raises RuntimeError with binary name when binary is missing."""
    with pytest.raises(RuntimeError, match="no-such-mmseqs"):
        pdb.mmseqs_version(str(tmp_path / "no-such-mmseqs"))


def test_run_mmseqs_reports_a_failing_binary(tmp_path: Path) -> None:
    """run_mmseqs raises RuntimeError with stderr when binary exits non-zero."""
    import sys

    # Create a failing binary that writes to stderr and exits 1
    bad_mmseqs = tmp_path / "bad-mmseqs"
    bad_mmseqs.write_text(
        f"""#!{sys.executable}
import sys
sys.stderr.write("boom")
sys.exit(1)
"""
    )
    bad_mmseqs.chmod(0o755)

    fasta = tmp_path / "in.fasta"
    fasta.write_text(">s0\nACDE\n")
    with pytest.raises(RuntimeError, match="boom"):
        pdb.run_mmseqs(fasta, tmp_path, mmseqs=str(bad_mmseqs))
