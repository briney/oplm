"""CASP domain targets: the assessed, domain-trimmed coordinates on the full target sequence.

Sources are the Prediction Center's domain tarballs and target sequence files, pinned by sha256
(docs/superpowers/specs/2026-10-27-structure-eval-sets-design.md, "CASP").
"""

from __future__ import annotations

import hashlib
import re
import tarfile
from pathlib import Path

import numpy as np

from oplm.fold.data.ccd import PROTEIN_1TO3
from oplm.fold.evalsets.chain import EvalChain

__all__ = [
    "CASP_EDITIONS",
    "CASP_SHA256",
    "canonical_target",
    "casp_domain_chains",
    "read_fasta",
    "sha256sum",
    "verify_casp_sources",
]

# edition -> (domain tarball, target sequences); both under predictioncenter.org/download_area/
# CASP<edition>/targets/ and CASP<edition>/sequences/.
CASP_EDITIONS: dict[str, tuple[str, str]] = {
    "15": ("casp15.targets.TS-domains.public_12.20.2022.tar.gz", "casp15.seq.txt"),
    "16": ("casp16.targets_monomer_trimmed2domains.tgz", "casp16.T1.seq.txt"),
}
CASP_SHA256: dict[str, str] = {
    "casp15.targets.TS-domains.public_12.20.2022.tar.gz": (
        "ac159c284a1506b460143ef6e22832272c5da1b4078fdb5fec8292b373b6239b"
    ),
    "casp15.seq.txt": "84dd04c1fda2a550af73e77ad25be94da23f16bd7ff6632f2c22c1978b9ee0a5",
    "casp16.targets_monomer_trimmed2domains.tgz": (
        "30e6bf992baed08695ebebe481711adea867b483015d30e18d170a0eee66ceb9"
    ),
    "casp16.T1.seq.txt": "77d0b5c4ffda52ca1be7563a92eabae7ca59a3aab814e79a377ddd962783e286",
}

_DOMAIN_FILE = re.compile(r"^(?P<target>T\d{4}[^-]*)-D(?P<domain>\d+)\.pdb$")
_ONE_LETTER = {three: one for one, three in PROTEIN_1TO3.items()} | {"MSE": "M"}


def sha256sum(path: Path) -> str:
    """Hex sha256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_casp_sources(directory: Path) -> list[Path]:
    """The CASP source files in ``directory``, each checked against its pinned sha256.

    Raises:
        FileNotFoundError: A pinned file is missing.
        ValueError: A file's sha256 differs from the pinned value.
    """
    paths: list[Path] = []
    for name, expected in CASP_SHA256.items():
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"missing CASP source {path}")
        actual = sha256sum(path)
        if actual != expected:
            raise ValueError(f"{path}: sha256 {actual} != pinned {expected}")
        paths.append(path)
    return paths


def read_fasta(path: Path) -> dict[str, str]:
    """``{id: sequence}`` from a FASTA file; the id is the header's first word."""
    sequences: dict[str, str] = {}
    name: str | None = None
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            name = line[1:].split()[0]
            sequences[name] = ""
        elif name is not None:
            sequences[name] += line.strip()
    return sequences


def canonical_target(stem: str) -> str:
    """The ``T1xxx`` id behind a CASP16 file stem. Monomer files name target ``T1xxx`` as
    ``T0xxx`` and phase-2 copies as ``T2xxx``; other stems are returned unchanged."""
    return re.sub(r"^T[02]", "T1", stem)


def _domain_atoms(text: str, sequence: str) -> list[dict[str, np.ndarray]] | None:
    """Heavy atoms per target residue from a domain file, or ``None`` when a residue falls
    outside ``sequence`` or disagrees with it. Water and ligand HETATM records, hydrogens, and
    alternate locations other than the first are ignored."""
    atoms: list[dict[str, np.ndarray]] = [{} for _ in sequence]
    for line in text.splitlines():
        if not line.startswith(("ATOM", "HETATM")) or line[16] not in " A":
            continue
        resname = line[17:20].strip()
        if resname not in _ONE_LETTER:
            if line.startswith("HETATM"):
                continue
            return None
        name = line[12:16].strip()
        if (line[76:78].strip() or name[0]) in ("H", "D"):
            continue
        index = int(line[22:26]) - 1
        if (
            line[26].strip()
            or not 0 <= index < len(sequence)
            or _ONE_LETTER[resname] != sequence[index]
        ):
            return None
        if resname == "MSE" and name == "SE":
            name = "SD"
        xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        atoms[index].setdefault(name, np.array(xyz, dtype=np.float32))
    return atoms


def casp_domain_chains(
    tarball: Path, sequences: dict[str, str], edition: str
) -> tuple[list[EvalChain], dict[str, int]]:
    """One :class:`EvalChain` per distinct single-domain unit in a CASP domain tarball.

    A chain's sequence is the full target sequence; only the domain's residues have atoms,
    placed by target numbering (``resSeq - 1``). Files are skipped and counted under
    ``multi_domain`` (combined units such as ``D12``), ``duplicate`` (a ``T0``/``T2`` copy of a
    domain already taken, or a later ``vN`` version), ``no_sequence`` and
    ``numbering_mismatch``; ``files`` counts every domain file seen.
    """
    stats = dict.fromkeys(
        ("files", "multi_domain", "duplicate", "no_sequence", "numbering_mismatch"), 0
    )
    groups: dict[tuple[str, str], list[tuple[str, str, bytes]]] = {}
    with tarfile.open(tarball) as archive:
        for member in archive.getmembers():
            filename = Path(member.name).name
            match = _DOMAIN_FILE.match(filename)
            handle = (
                archive.extractfile(member)
                if (member.isfile() or member.islnk()) and match
                else None
            )
            if match is None or handle is None:
                continue
            stats["files"] += 1
            if len(match["domain"]) > 1:
                stats["multi_domain"] += 1
                continue
            target = canonical_target(match["target"])
            key = (re.sub(r"v\d+$", "", target), match["domain"])
            groups.setdefault(key, []).append((target, filename, handle.read()))
    chains: list[EvalChain] = []
    for (base, domain), candidates in sorted(groups.items()):
        stats["duplicate"] += len(candidates) - 1
        target, filename, data = min(candidates)
        sequence = sequences.get(target) or sequences.get(base)
        if sequence is None:
            stats["no_sequence"] += 1
            continue
        atoms = _domain_atoms(data.decode(), sequence)
        if atoms is None:
            stats["numbering_mismatch"] += 1
            continue
        meta = {
            "source": f"casp{edition}",
            "entry": target,
            "chain": f"D{domain}",
            "source_file": filename,
        }
        chains.append(
            EvalChain(
                name=f"casp{edition}_{target}-D{domain}",
                sequence=sequence,
                atoms=atoms,
                meta=meta,
            )
        )
    return chains, stats
