"""Shared fixtures for the eval-set builder tests: CASP sources built from real crambin."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

CRAMBIN_PDB = Path(__file__).parents[2] / "data" / "fixtures" / "structures" / "1CRN.pdb"
CRAMBIN = "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN"
GAP = frozenset(range(20, 25))  # residues 20-24 (1-based) left unmodeled


def crambin_domain(drop: frozenset[int] = frozenset(), renumber: int = 0) -> bytes:
    """Crambin's ATOM records as a CASP domain file, minus residues ``drop`` (1-based)."""
    out: list[str] = []
    for line in CRAMBIN_PDB.read_text().splitlines():
        if line.startswith("ATOM"):
            resseq = int(line[22:26])
            if resseq not in drop:
                out.append(f"{line[:22]}{resseq + renumber:>4}{line[26:]}")
    return ("\n".join(out) + "\nEND\n").encode()


def make_tarball(path: Path, files: dict[str, bytes]) -> Path:
    """A gzipped tarball holding ``files`` (member name -> bytes)."""
    with tarfile.open(path, "w:gz") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return path


@pytest.fixture
def casp_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The four CASP source files, made from crambin, with their hashes pinned for the test.

    CASP15 holds one full domain; CASP16 holds a gapped domain and its phase-2 copy.
    """
    from oplm.fold.evalsets import casp

    directory = tmp_path / "casp"
    directory.mkdir()
    (tar15, seq15), (tar16, seq16) = casp.CASP_EDITIONS["15"], casp.CASP_EDITIONS["16"]
    make_tarball(directory / tar15, {"./T1104-D1.pdb": crambin_domain()})
    (directory / seq15).write_text(f">T1104 crambin\n{CRAMBIN}\n")
    gapped = crambin_domain(GAP)
    make_tarball(directory / tar16, {"T0201-D1.pdb": gapped, "T2201-D1.pdb": gapped})
    (directory / seq16).write_text(f">T1201 crambin, 46 residues\n{CRAMBIN}\n")
    pinned = {path.name: casp.sha256sum(path) for path in directory.iterdir()}
    monkeypatch.setattr(casp, "CASP_SHA256", pinned)
    return directory
