"""Residue vocabulary, atom tables and reference conformers for featurization.

Values transcribed from Biohub's ESMFold2 ``constants.py`` / ``protein_utils.py`` (Apache-2.0;
see THIRD_PARTY_NOTICES.md): the 33-class residue-type vocabulary (ids 2..21 are the twenty
amino acids by alphabetical three-letter code, 22 is UNK), the heavy-atom order per residue,
the three charged atoms, and the atom-name character encoding. Reference conformers come
from ``reference_conformers.json`` next to this file: upstream's ``get_idealized_atom_pos``
dump of the CCD conformers in ``biohub/ESMFold2``'s ``ccd.pkl`` (see its ``source`` field),
written by ``oplm fold make-fixtures``; ESMFold2-Fast ships no ``ccd.pkl``.
A gemmi-parsed CCD table (``ChemCompTable``) now exists for the parent mapping; ligand and
nucleic-acid atom tables remain milestone 5.
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
    "ChemComp",
    "ChemCompTable",
    "LM_UNKNOWN_ID",
    "MAX_ATOMIC_NUMBER",
    "MOL_TYPE_DNA",
    "MOL_TYPE_LIGAND",
    "MOL_TYPE_PROTEIN",
    "MOL_TYPE_RNA",
    "NUM_RES_TYPES",
    "PROTEIN_1TO3",
    "PROTEIN_3TO1",
    "PROTEIN_RES_TYPES",
    "RES_TYPE_TO_CODE",
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


PROTEIN_3TO1: dict[str, str] = {code: letter for letter, code in PROTEIN_1TO3.items()}
RES_TYPE_TO_CODE: dict[int, str] = {res_type: code for code, res_type in PROTEIN_RES_TYPES.items()}


def _cif_value(raw: str | None) -> str | None:
    """A ``_chem_comp`` value with CIF quoting removed; ``None`` for absent, ``?`` and ``.``."""
    if raw is None or raw in ("?", "."):
        return None
    import gemmi

    return gemmi.cif.as_string(raw)


@dataclass(frozen=True)
class ChemComp:
    """One ``_chem_comp`` row of the wwPDB Chemical Component Dictionary."""

    code: str
    type: str  # ``_chem_comp.type`` upper-cased, e.g. "L-PEPTIDE LINKING"
    parent: str | None  # ``mon_nstd_parent_comp_id`` when it names exactly one component
    one_letter: str | None


class ChemCompTable:
    """``_chem_comp`` rows keyed by code, from ``components.cif.gz`` or a saved JSON cache."""

    def __init__(self, comps: dict[str, ChemComp], source: str) -> None:
        self._comps = comps
        self.source = source

    @classmethod
    def parse(cls, components_cif: Path) -> ChemCompTable:
        """Parse the dictionary (gzipped or plain mmCIF); about 3 s for the full 51k-block file."""
        import gemmi

        doc = gemmi.cif.read(str(components_cif))
        comps: dict[str, ChemComp] = {}
        for block in doc:
            parent = _cif_value(block.find_value("_chem_comp.mon_nstd_parent_comp_id"))
            if parent is not None and "," in parent:
                parent = None  # multi-parent (e.g. "ALA, SER, GLY") have no single parent
            comps[block.name] = ChemComp(
                code=block.name,
                type=(_cif_value(block.find_value("_chem_comp.type")) or "").upper(),
                parent=parent,
                one_letter=_cif_value(block.find_value("_chem_comp.one_letter_code")),
            )
        return cls(comps, f"{Path(components_cif).name} ({len(comps)} components)")

    def save(self, path: Path) -> None:
        """Write the table as JSON (``{"source": ..., "components": {code: {...}}}``)."""
        payload = {
            "source": self.source,
            "components": {
                code: {"type": c.type, "parent": c.parent, "one_letter": c.one_letter}
                for code, c in self._comps.items()
            },
        }
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ChemCompTable:
        """Read a table written by :meth:`save`."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        comps = {
            code: ChemComp(code, entry["type"], entry["parent"], entry["one_letter"])
            for code, entry in raw["components"].items()
        }
        return cls(comps, raw["source"])

    def __len__(self) -> int:
        return len(self._comps)

    def __contains__(self, code: str) -> bool:
        return code in self._comps

    def __getitem__(self, code: str) -> ChemComp:
        return self._comps[code]

    def is_peptide(self, code: str) -> bool:
        """Whether the component's type names a peptide (any chirality or linkage)."""
        comp = self._comps.get(code)
        return comp is not None and "PEPTIDE" in comp.type

    def protein_parent(self, code: str) -> str:
        """The standard amino acid a polymer residue collapses to (spec §4.1), else ``"UNK"``.

        A standard residue is its own parent. A modified L-peptide residue (type contains
        "PEPTIDE" and does not start with "D-") resolves to its single
        ``mon_nstd_parent_comp_id`` when that is a standard amino acid. D-peptides,
        NON-POLYMER, multi-parent, and unknown codes become ``UNK``.
        """
        if code in PROTEIN_RES_TYPES:
            return code
        comp = self._comps.get(code)
        if (
            comp is not None
            and self.is_peptide(code)
            and not comp.type.startswith("D-")
            and comp.parent in PROTEIN_RES_TYPES
            and comp.parent != "UNK"
        ):
            return comp.parent
        return "UNK"
