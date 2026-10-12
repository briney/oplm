"""Batch ``FoldExample``s to the crop and the atom budget (spec §4.7).

Tokens pad to ``crop_size``; atoms pad to ``atoms_per_token_budget × crop_size`` rounded up to
the attention block multiple. An example over either limit is rejected, never truncated. LM rows
of all examples are stacked and ``lm_rows`` is offset so each token still names its own row.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import torch

from oplm.fold.data.featurize import FoldFeatures, FoldTargets
from oplm.model.tokenization_oplm import VOCAB

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from torch import Tensor

    from oplm.fold.data.featurize import FoldExample

__all__ = ["FoldBatch", "atom_budget", "collate_examples", "collate_rows"]

_PAD = VOCAB["<pad>"]
_TOKEN_FIELDS = (
    "token_index",
    "residue_index",
    "asym_id",
    "entity_id",
    "sym_id",
    "mol_type",
    "res_type",
    "token_mask",
    "distogram_atom_idx",
    "lm_rows",
    "lm_positions",
)
_ATOM_FIELDS = (
    "ref_pos",
    "ref_element",
    "ref_charge",
    "ref_atom_name_chars",
    "ref_space_uid",
    "atom_mask",
    "atom_to_token",
)


@dataclass
class FoldBatch:
    """A collated batch: ids, features ``(B, ...)`` and targets ``(B, ...)``."""

    ids: list[str]
    features: FoldFeatures
    targets: FoldTargets


def atom_budget(crop_size: int, atoms_per_token_budget: int, atom_pad_multiple: int) -> int:
    """Padded atom-axis length for a crop: budget × crop rounded up to the block multiple."""
    return -(-atoms_per_token_budget * crop_size // atom_pad_multiple) * atom_pad_multiple


def _pad(t: Tensor, dim: int, size: int, value: float = 0) -> Tensor:
    if t.shape[dim] == size:
        return t
    shape = list(t.shape)
    shape[dim] = size - t.shape[dim]
    return torch.cat([t, torch.full(shape, value, dtype=t.dtype)], dim=dim)


def collate_examples(
    examples: Sequence[FoldExample],
    *,
    crop_size: int,
    atoms_per_token_budget: int = 24,
    atom_pad_multiple: int = 32,
) -> FoldBatch:
    """Stack examples into one :class:`FoldBatch`.

    Raises:
        ValueError: ``examples`` is empty, an example has more tokens than ``crop_size``, or more
            real atoms than the atom budget.
    """
    if not examples:
        raise ValueError("collate_examples needs at least one example")
    L = crop_size
    A = atom_budget(crop_size, atoms_per_token_budget, atom_pad_multiple)
    for ex in examples:
        n_tokens = int(ex.features.token_mask.sum())
        n_atoms = int(ex.features.atom_mask.sum())
        if n_tokens > L:
            raise ValueError(f"{ex.id}: {n_tokens} tokens exceed crop_size={crop_size}")
        if n_atoms > A:
            raise ValueError(
                f"{ex.id}: {n_atoms} atoms exceed the atom budget {A} "
                f"({atoms_per_token_budget} per token × {crop_size}); recrop instead of truncating"
            )
    T = max(ex.features.lm_input_ids.shape[1] for ex in examples)
    cols: dict[str, list[Tensor]] = {name: [] for name in (*_TOKEN_FIELDS, *_ATOM_FIELDS)}
    bonds, lm_ids, lm_mask, chain_ids = [], [], [], []
    row_offset = 0
    for ex in examples:
        f = ex.features
        for name in _TOKEN_FIELDS:
            value = getattr(f, name)
            if name == "lm_rows":
                value = value + row_offset * f.token_mask.long()
            cols[name].append(_pad(value, 1, L))
        for name in _ATOM_FIELDS:
            cols[name].append(_pad(getattr(f, name), 1, A))
        bonds.append(_pad(_pad(f.token_bonds, 1, L), 2, L))
        lm_ids.append(_pad(f.lm_input_ids, 1, T, _PAD))
        lm_mask.append(_pad(f.lm_attention_mask, 1, T))
        chain_ids.extend(f.chain_ids)
        row_offset += f.num_chains
    features = FoldFeatures(
        **{name: torch.cat(cols[name], dim=0) for name in (*_TOKEN_FIELDS, *_ATOM_FIELDS)},
        token_bonds=torch.cat(bonds, dim=0),
        lm_input_ids=torch.cat(lm_ids, dim=0),
        lm_attention_mask=torch.cat(lm_mask, dim=0),
        chain_ids=chain_ids,
    )
    targets = FoldTargets(
        **{
            fld.name: torch.cat(
                [
                    _pad(getattr(ex.targets, fld.name), 1, L if fld.name == "frame_atom_idx" else A)
                    for ex in examples
                ],
                dim=0,
            )
            for fld in fields(FoldTargets)
        }
    )
    return FoldBatch(ids=[ex.id for ex in examples], features=features, targets=targets)


def collate_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    crop_size: int,
    atoms_per_token_budget: int = 24,
    atom_pad_multiple: int = 32,
) -> FoldBatch:
    """``collate_examples`` over ``StructureDataset`` rows (``{"id", "example"}`` dicts)."""
    examples = [row["example"] for row in rows]
    return collate_examples(
        examples,  # ty: ignore[invalid-argument-type]  # rows are untyped dicts by the dataset contract
        crop_size=crop_size,
        atoms_per_token_budget=atoms_per_token_budget,
        atom_pad_multiple=atom_pad_multiple,
    )
