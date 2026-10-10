"""Residue vocabulary, atom tables and reference conformers for featurization.

Values transcribed from Biohub's ESMFold2 ``constants.py`` / ``protein_utils.py`` (Apache-2.0;
see THIRD_PARTY_NOTICES.md): the 33-class residue-type vocabulary (ids 2..21 are the twenty
amino acids by alphabetical three-letter code, 22 is UNK), the heavy-atom order per residue,
the three charged atoms, and the atom-name character encoding. Reference conformers come
from ``reference_conformers.json`` next to this file (see its ``source`` field); the fixture
generator can regenerate that file from the ``ccd.pkl`` shipped with the ESMFold2 weights.
Milestone 5 widens this module to ligands and nucleic acids via a gemmi-parsed CCD cache.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import torch

__all__ = [
    "ATOM_NAME_CHARS",
    "ATOM_NAME_VOCAB",
    "LM_UNKNOWN_ID",
    "MAX_ATOMIC_NUMBER",
    "MOL_TYPE_DNA",
    "MOL_TYPE_LIGAND",
    "MOL_TYPE_PROTEIN",
    "MOL_TYPE_RNA",
    "NUM_RES_TYPES",
    "PROTEIN_1TO3",
    "PROTEIN_RES_TYPES",
    "UNK_RES_TYPE",
    "ReferenceConformers",
    "ResidueTemplate",
    "encode_atom_name",
]

MOL_TYPE_PROTEIN = 0
MOL_TYPE_DNA = 1
MOL_TYPE_RNA = 2
MOL_TYPE_LIGAND = 3
NUM_RES_TYPES = 33  # 0 MSA pad, 1 MSA gap, 2..21 amino acids, 22 UNK, 23..27 RNA, 28..32 DNA
UNK_RES_TYPE = 22
LM_UNKNOWN_ID = 24  # the OPLM/ESMC token id of "X"; ESMFold2 gives it to every non-standard residue
MAX_ATOMIC_NUMBER = 128
ATOM_NAME_CHARS = 4
ATOM_NAME_VOCAB = 64

_THREE_LETTER = (
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
)
PROTEIN_RES_TYPES: dict[str, int] = {code: 2 + i for i, code in enumerate(_THREE_LETTER)}
PROTEIN_RES_TYPES["UNK"] = UNK_RES_TYPE
PROTEIN_1TO3: dict[str, str] = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
    "X": "UNK",
}


def encode_atom_name(name: str) -> tuple[int, int, int, int]:
    """Left-aligned four-character encoding, ``ord(c) - 32`` per character (space is 0)."""
    padded = name.ljust(ATOM_NAME_CHARS)[:ATOM_NAME_CHARS]
    codes = tuple(ord(c) - 32 for c in padded)
    if any(not 0 <= c < ATOM_NAME_VOCAB for c in codes):
        raise ValueError(f"atom name {name!r} has characters outside the 64-symbol vocabulary")
    return codes  # ty: ignore[invalid-return-type]  # tuple length fixed by ATOM_NAME_CHARS


@dataclass(frozen=True)
class ResidueTemplate:
    """Heavy atoms of one residue type in model order, with reference geometry."""

    name: str
    atoms: tuple[str, ...]
    elements: tuple[int, ...]
    charges: tuple[int, ...]
    positions: torch.Tensor  # (n_atoms, 3) float32, the reference conformer

    @property
    def representative_atom(self) -> int:
        """Index of CB, or CA when the residue has no CB (distogram/confidence representative)."""
        return self.atoms.index("CB") if "CB" in self.atoms else self.atoms.index("CA")


class ReferenceConformers:
    """Residue templates keyed by three-letter code."""

    def __init__(self, residues: dict[str, ResidueTemplate], source: str) -> None:
        self._residues = residues
        self.source = source

    @classmethod
    def load(cls, path: Path | None = None) -> ReferenceConformers:
        """Load the packaged table, or a JSON file with the same schema."""
        if path is None:
            text = (
                resources.files("oplm.fold.data").joinpath("reference_conformers.json").read_text()
            )
        else:
            text = Path(path).read_text()
        raw = json.loads(text)
        residues = {
            code: ResidueTemplate(
                name=code,
                atoms=tuple(entry["atoms"]),
                elements=tuple(entry["elements"]),
                charges=tuple(entry["charges"]),
                positions=torch.tensor(entry["positions"], dtype=torch.float32),
            )
            for code, entry in raw["residues"].items()
        }
        return cls(residues, raw["source"])

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(self._residues)

    def __getitem__(self, code: str) -> ResidueTemplate:
        return self._residues[code]
