"""One eval structure: full sequence, modeled atoms, PDB output, and the curation rules.

The ``structure`` eval reads these files with :func:`oplm.data.load_structures`. SEQRES carries
the full sequence and residues are numbered by sequence position, so unmodeled residues load as
NaN rows and are left out of scoring
(docs/superpowers/specs/2026-10-27-structure-eval-sets-design.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from oplm.eval.metrics.contact import compute_contact_map, contact_positions_valid
from oplm.fold.data.ccd import PROTEIN_1TO3

if TYPE_CHECKING:
    from pathlib import Path

    from oplm.fold.eval.targets import EvalTarget

__all__ = [
    "STANDARD_AMINO_ACIDS",
    "EvalChain",
    "count_long_range_contacts",
    "from_eval_target",
    "rejection",
    "write_pdb",
]

STANDARD_AMINO_ACIDS = frozenset(PROTEIN_1TO3) - {"X"}
_BACKBONE = ("N", "CA", "C")
_SEQRES_PER_LINE = 13


@dataclass(frozen=True)
class EvalChain:
    """A curated chain: what the model reads (``sequence``) and what is scored (``atoms``).

    Attributes:
        name: File stem, e.g. ``casp16_T1201-D1`` or ``4hhb_A`` (entry and label chain id).
        sequence: Full input sequence, one letter per residue.
        atoms: Per residue, modeled atom name -> ``(3,)`` float32 coordinates; ``{}`` when the
            residue is unmodeled.
        meta: Manifest fields (``source``, ``entry``, ``chain``, ``method``, ...).
    """

    name: str
    sequence: str
    atoms: list[dict[str, np.ndarray]]
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.atoms) != len(self.sequence):
            raise ValueError(
                f"{self.name}: atoms for {len(self.atoms)} residues, "
                f"sequence of {len(self.sequence)}"
            )

    @property
    def modeled(self) -> int:
        """Residues with N, CA and C: the L of precision@L."""
        return sum(all(atom in residue for atom in _BACKBONE) for residue in self.atoms)

    def backbone(self) -> np.ndarray:
        """``(L, 3, 3)`` N/CA/C coordinates, NaN where an atom is unmodeled."""
        out = np.full((len(self.sequence), len(_BACKBONE), 3), np.nan, dtype=np.float32)
        for i, residue in enumerate(self.atoms):
            for k, atom in enumerate(_BACKBONE):
                if atom in residue:
                    out[i, k] = residue[atom]
        return out


def from_eval_target(target: EvalTarget, *, meta: dict[str, Any]) -> EvalChain:
    """An :class:`EvalChain` named ``target.key`` from a fold-dataset chain: every residue of
    the entity sequence, with the atoms that are resolved.
    """
    atoms: list[dict[str, np.ndarray]] = [{} for _ in target.sequence]
    for k, atom in enumerate(target.atom_names):
        if target.resolved[k]:
            atoms[int(target.residue_index[k])][atom] = target.coords[k].astype(np.float32)
    return EvalChain(name=target.key, sequence=target.sequence, atoms=atoms, meta=meta)


def count_long_range_contacts(
    backbone: np.ndarray, *, threshold: float = 8.0, min_sep: int = 24
) -> int:
    """Contacts (virtual Cβ closer than ``threshold`` Å) between modeled residues at least
    ``min_sep`` apart in sequence, counted with the ``structure`` eval's own contact code.
    """
    coords = torch.as_tensor(backbone, dtype=torch.float32)
    contacts = compute_contact_map(coords, threshold=threshold)
    valid = contact_positions_valid(coords)
    i, j = torch.triu_indices(len(coords), len(coords), offset=min_sep)
    keep = valid[i] & valid[j]
    return int(contacts[i[keep], j[keep]].sum().item())


def rejection(
    chain: EvalChain, *, min_length: int, max_length: int, min_modeled_fraction: float
) -> str | None:
    """Why ``chain`` fails the shared curation rules, or ``None`` when it passes.

    Checked in order: only standard amino acids (``"nonstandard"``); sequence length within
    ``[min_length, max_length]`` (``"length"``); at least one modeled residue and a modeled
    fraction of at least ``min_modeled_fraction`` (``"modeled"``); at least ``modeled / 5``
    long-range contacts (``"contacts"``).
    """
    if set(chain.sequence) - STANDARD_AMINO_ACIDS:
        return "nonstandard"
    if not min_length <= len(chain.sequence) <= max_length:
        return "length"
    modeled = chain.modeled
    if modeled == 0 or modeled < min_modeled_fraction * len(chain.sequence):
        return "modeled"
    if count_long_range_contacts(chain.backbone()) < modeled / 5:
        return "contacts"
    return None


def write_pdb(chain: EvalChain, path: Path) -> None:
    """Write ``chain`` as a one-chain PDB file (chain id ``A``).

    SEQRES holds the full sequence; ATOM records hold the modeled atoms, numbered by sequence
    position (1-based), with occupancy 1 and B-factor 0.

    Raises:
        ValueError: The sequence has a residue outside the twenty standard amino acids.
    """
    unknown = sorted(set(chain.sequence) - STANDARD_AMINO_ACIDS)
    if unknown:
        raise ValueError(f"{chain.name}: non-standard residues {unknown}")
    names = [PROTEIN_1TO3[aa] for aa in chain.sequence]
    lines = [
        f"SEQRES {n // _SEQRES_PER_LINE + 1:>3} A {len(names):>4}  "
        + " ".join(names[n : n + _SEQRES_PER_LINE])
        for n in range(0, len(names), _SEQRES_PER_LINE)
    ]
    serial = 0
    for i, residue in enumerate(chain.atoms):
        for atom, (x, y, z) in residue.items():
            serial += 1
            label = atom if len(atom) == 4 else f" {atom:<3}"
            lines.append(
                f"ATOM  {serial:>5} {label}{names[i]:>4} A{i + 1:>4}    "
                f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{1.0:>6.2f}{0.0:>6.2f}          {atom[0]:>2}"
            )
    lines += ["TER", "END"]
    path.write_text("\n".join(lines) + "\n")
