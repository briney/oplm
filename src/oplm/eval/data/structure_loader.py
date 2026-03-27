"""Load protein structures from PDB/CIF files for contact evaluation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

_STRUCTURE_EXTENSIONS = {"*.pdb", "*.cif", "*.ent", "*.mmcif"}

# Map common modified residues to their standard amino acid equivalents.
_MODIFIED_RESIDUE_MAP: dict[str, str] = {
    "MSE": "M",  # selenomethionine
    "SEC": "C",  # selenocysteine
    "CSE": "C",  # selenocysteine (alternate)
    "SEP": "S",  # phosphoserine
    "TPO": "T",  # phosphothreonine
    "PTR": "Y",  # phosphotyrosine
}


@dataclass
class StructureData:
    """Parsed protein structure for contact evaluation.

    Attributes:
        name: PDB ID or filename stem.
        sequence: One-letter amino acid sequence.
        coords: Backbone atom coordinates, shape ``(L, 3, 3)`` for N, CA, C.
            NaN for missing atoms.
        chain_id: Chain identifier (if applicable).
    """

    name: str
    sequence: str
    coords: Tensor  # (L, 3, 3) — N, CA, C
    chain_id: str | None = None


def _residue_to_one_letter(residue_name: str) -> str:
    """Map a three-letter residue name to a one-letter code.

    Handles standard residues, common modified residues, and unknowns.
    """
    # Check modified residues first
    if residue_name in _MODIFIED_RESIDUE_MAP:
        return _MODIFIED_RESIDUE_MAP[residue_name]

    try:
        from Bio.Data.IUPACData import protein_letters_3to1

        return protein_letters_3to1.get(residue_name.capitalize(), "X")
    except ImportError as e:
        raise ImportError(
            "Structure evaluation requires biopython. Install with: pip install oplm[eval]"
        ) from e


def _parse_single_structure(filepath: Path) -> StructureData | None:
    """Parse a single PDB or CIF file into a StructureData.

    Args:
        filepath: Path to the PDB or CIF file.

    Returns:
        Parsed structure, or None if parsing fails.
    """
    try:
        from Bio.PDB import MMCIFParser, PDBParser  # type: ignore[attr-defined]
    except ImportError as e:
        raise ImportError(
            "Structure evaluation requires biopython. Install with: pip install oplm[eval]"
        ) from e

    suffix = filepath.suffix.lower()
    parser = (
        MMCIFParser(QUIET=True)  # type: ignore[no-untyped-call]
        if suffix in (".cif", ".mmcif")
        else PDBParser(QUIET=True)  # type: ignore[no-untyped-call]
    )

    try:
        structure = parser.get_structure(filepath.stem, str(filepath))  # type: ignore[no-untyped-call]
    except Exception:
        logger.warning("Failed to parse structure file: %s", filepath)
        return None

    # Use first model (model 0 for X-ray, first NMR conformer)
    try:
        model = structure[0]
    except (KeyError, IndexError):
        logger.warning("No models found in structure file: %s", filepath)
        return None

    # Use first chain
    chains = list(model.get_chains())
    if not chains:
        logger.warning("No chains found in structure file: %s", filepath)
        return None
    chain = chains[0]

    sequence_chars: list[str] = []
    coord_list: list[list[list[float]]] = []

    for residue in chain.get_residues():
        # Skip heteroatoms (water, ligands) unless they are known modified residues
        hetflag = residue.id[0]
        resname = residue.resname.strip()
        if hetflag != " " and resname not in _MODIFIED_RESIDUE_MAP:
            continue

        # Map residue name to one-letter code
        one_letter = _residue_to_one_letter(resname)
        sequence_chars.append(one_letter)

        # Extract backbone atom coordinates: N, CA, C
        atom_coords: list[list[float]] = []
        for atom_name in ("N", "CA", "C"):
            if atom_name in residue:
                atom_coords.append(list(residue[atom_name].get_vector()))
            else:
                atom_coords.append([float("nan")] * 3)
        coord_list.append(atom_coords)

    if not sequence_chars:
        logger.warning("No residues found in structure file: %s", filepath)
        return None

    return StructureData(
        name=filepath.stem,
        sequence="".join(sequence_chars),
        coords=torch.tensor(coord_list, dtype=torch.float32),
        chain_id=chain.id,
    )


def load_structures(
    directory: str | Path,
    max_structures: int | None = None,
) -> list[StructureData]:
    """Parse all PDB/CIF files in a directory.

    Uses BioPython ``PDBParser`` and ``MMCIFParser``. Files that fail to
    parse are skipped with a warning.

    Args:
        directory: Path to directory containing structure files.
        max_structures: Maximum number of structures to load. None for all.

    Returns:
        List of :class:`StructureData` sorted alphabetically by filename.

    Raises:
        FileNotFoundError: If the directory does not exist.
        ImportError: If BioPython is not installed.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Structure directory not found: {directory}")

    # Collect all matching files, sorted for determinism
    files: list[Path] = []
    for ext in sorted(_STRUCTURE_EXTENSIONS):
        files.extend(directory.glob(ext))
    files.sort(key=lambda p: p.name)

    if max_structures is not None:
        files = files[:max_structures]

    structures: list[StructureData] = []
    for filepath in files:
        result = _parse_single_structure(filepath)
        if result is not None:
            structures.append(result)

    logger.info("Loaded %d structures from %s", len(structures), directory)
    return structures
