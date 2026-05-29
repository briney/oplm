"""Structure loading.

Parses PDB/mmCIF files into :class:`StructureData` (sequence plus backbone
N/CA/C coordinates) for the structure eval modality. Biopython is lazy-imported
inside the parsing functions and is only required under the ``train`` extra; the
core training install never hard-requires it (docs/DATA_TOOLING.md §5.5).

The structure modality is **eval-only**: sequences reuse the canonical tokenizer
via the pad primitive and no masking is applied. This module's job ends at
producing :class:`StructureData`; contact-map math lives in the eval harness.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

__all__ = ["StructureData", "load_structures"]

# Recognized structure-file suffixes (case-insensitive).
_STRUCTURE_SUFFIXES = (".pdb", ".cif", ".ent", ".mmcif")

# Suffixes routed to the mmCIF parser; everything else uses the PDB parser.
_MMCIF_SUFFIXES = frozenset({".cif", ".mmcif"})

# Backbone atoms extracted per residue, in fixed order (coords[..., i, :]).
_BACKBONE_ATOMS = ("N", "CA", "C")

# Common modified residues mapped to their canonical parent amino acid. Consulted
# before biopython's table so these HETATM residues are kept (not skipped) and
# resolve to the right one-letter code. Extend as needed.
_MODIFIED_RESIDUE_MAP: dict[str, str] = {
    "MSE": "M",  # selenomethionine
    "SEC": "C",  # selenocysteine
    "CSE": "C",  # selenocysteine (alternate)
    "SEP": "S",  # phosphoserine
    "TPO": "T",  # phosphothreonine
    "PTR": "Y",  # phosphotyrosine
}

_BIOPYTHON_IMPORT_ERROR = (
    "Structure parsing requires biopython, an optional dependency. "
    "Install it with: pip install oplm[train]"
)


@dataclass
class StructureData:
    """A parsed protein structure for the contact-prediction eval task.

    Attributes:
        name: PDB id / filename stem.
        sequence: One-letter amino-acid sequence (length ``L``).
        coords: Backbone coordinates, shape ``(L, 3, 3)`` for the N, CA, C atoms
            (in that order). Missing atoms are ``NaN`` rows so the first axis
            stays aligned with ``sequence``.
        chain_id: Identifier of the parsed chain, or ``None`` if unavailable.
    """

    name: str
    sequence: str
    coords: Tensor  # (L, 3, 3) backbone N, CA, C; NaN for missing atoms
    chain_id: str | None


def _residue_to_one_letter(residue_name: str) -> str:
    """Map a three-letter residue name to a one-letter amino-acid code.

    Consults the modified-residue map first, then biopython's
    ``protein_letters_3to1`` table, falling back to ``"X"`` for anything
    unrecognized.

    Args:
        residue_name: Three-letter residue name (e.g. ``"ALA"``, ``"MSE"``).

    Returns:
        The one-letter code, or ``"X"`` if unknown.

    Raises:
        ImportError: If biopython is not installed.
    """
    if residue_name in _MODIFIED_RESIDUE_MAP:
        return _MODIFIED_RESIDUE_MAP[residue_name]

    try:
        from Bio.Data.IUPACData import protein_letters_3to1
    except ImportError as exc:
        raise ImportError(_BIOPYTHON_IMPORT_ERROR) from exc

    # The table is keyed by capitalized names (e.g. "Ala").
    return str(protein_letters_3to1.get(residue_name.capitalize(), "X"))


def _parse_single_structure(path: Path) -> StructureData | None:
    """Parse one PDB/mmCIF file into :class:`StructureData`.

    Uses the **first model, first chain**. Heteroatoms are skipped unless they
    are known modified residues. Per residue, the N/CA/C backbone atoms are
    extracted (a ``NaN`` row substitutes for any missing atom).

    Args:
        path: Path to a single structure file.

    Returns:
        The parsed structure, or ``None`` (with a warning) on parse failure or an
        empty chain.

    Raises:
        ImportError: If biopython is not installed.
    """
    try:
        # biopython ships only partial type info, so these are untyped to mypy.
        from Bio.PDB import MMCIFParser, PDBParser  # type: ignore[attr-defined]
    except ImportError as exc:
        raise ImportError(_BIOPYTHON_IMPORT_ERROR) from exc

    parser = (
        MMCIFParser(QUIET=True)  # type: ignore[no-untyped-call]
        if path.suffix.lower() in _MMCIF_SUFFIXES
        else PDBParser(QUIET=True)  # type: ignore[no-untyped-call]
    )

    try:
        structure = parser.get_structure(path.stem, str(path))  # type: ignore[no-untyped-call]
    except Exception:
        # biopython raises a variety of parser-specific errors on malformed
        # files; the spec is to skip-and-warn rather than propagate.
        logger.warning("failed to parse structure file %s; skipping", path)
        return None

    model = next(iter(structure), None)  # first model
    chain = next(iter(model), None) if model is not None else None
    if chain is None:
        logger.warning("no model/chain found in %s; skipping", path)
        return None

    sequence_chars: list[str] = []
    coord_rows: list[list[list[float]]] = []
    for residue in chain.get_residues():
        # residue.id == (hetflag, seqid, icode); hetflag is " " for standard
        # residues, non-blank for heteroatoms (water, ligands, modified residues).
        hetflag = residue.id[0]
        resname = residue.resname.strip()
        if hetflag != " " and resname not in _MODIFIED_RESIDUE_MAP:
            continue  # skip water/ligands; keep known modified residues

        sequence_chars.append(_residue_to_one_letter(resname))
        coord_rows.append(
            [
                residue[atom].get_coord().tolist() if atom in residue else [float("nan")] * 3
                for atom in _BACKBONE_ATOMS
            ]
        )

    if not sequence_chars:
        logger.warning("no protein residues found in %s; skipping", path)
        return None

    return StructureData(
        name=path.stem,
        sequence="".join(sequence_chars),
        coords=torch.tensor(coord_rows, dtype=torch.float32),  # (L, 3, 3)
        chain_id=chain.id,
    )


def _discover_structure_files(directory: Path) -> list[Path]:
    """Return structure files in ``directory``, sorted by filename for determinism."""
    files: Iterable[Path] = (
        p for p in directory.iterdir() if p.suffix.lower() in _STRUCTURE_SUFFIXES
    )
    return sorted(files, key=lambda p: p.name)


def load_structures(
    directory: str | Path,
    max_structures: int | None = None,
) -> list[StructureData]:
    """Parse every PDB/mmCIF structure in a directory.

    Globs ``*.pdb``, ``*.cif``, ``*.ent``, and ``*.mmcif`` files, sorts them by
    filename for determinism, and parses each (first model, first chain). Files
    that fail to parse are skipped with a warning.

    Args:
        directory: Directory containing structure files.
        max_structures: Optional cap on the number of structures returned (applied
            after sorting). ``None`` loads all.

    Returns:
        A list of :class:`StructureData`, ordered by filename.

    Raises:
        FileNotFoundError: If ``directory`` is not an existing directory.
        ImportError: If biopython is not installed.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"structure directory not found: {directory}")

    files = _discover_structure_files(directory)
    if max_structures is not None:
        files = files[:max_structures]

    structures = [parsed for p in files if (parsed := _parse_single_structure(p)) is not None]
    logger.info("loaded %d structures from %s", len(structures), directory)
    return structures
