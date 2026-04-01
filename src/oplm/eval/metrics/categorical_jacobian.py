"""Categorical-Jacobian contact extraction for structure evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from oplm.eval.metrics.contact import apply_apc, compute_precision_at_l

if TYPE_CHECKING:
    from collections.abc import Callable

    from oplm.data.tokenizer import ProteinTokenizer

# Matches the canonical amino-acid token ordering used by ESM-family models.
CANONICAL_AMINO_ACIDS: tuple[str, ...] = (
    "L",
    "A",
    "G",
    "V",
    "S",
    "E",
    "R",
    "T",
    "I",
    "D",
    "P",
    "K",
    "Q",
    "N",
    "F",
    "Y",
    "M",
    "H",
    "W",
    "C",
)


@dataclass
class StructurePairScoreData:
    """Flat residue-pair scores and labels for one structure.

    Attributes:
        scores: Predicted score for each valid residue pair, shape ``(n_pairs,)``.
        labels: Binary contact labels for the same pairs, shape ``(n_pairs,)``.
        seq_len: Effective sequence length.
        pair_indices: ``(i_indices, j_indices)`` for reconstructing ``(L, L)``
            matrices from the flat arrays.
    """

    scores: Tensor
    labels: Tensor
    seq_len: int
    pair_indices: tuple[Tensor, Tensor]


def get_canonical_amino_acid_token_ids(tokenizer: ProteinTokenizer) -> Tensor:
    """Return tokenizer IDs for the 20 canonical amino acids.

    Args:
        tokenizer: Tokenizer used to encode protein sequences.

    Returns:
        Long tensor of shape ``(20,)`` in canonical ESM ordering.
    """
    token_ids = [
        tokenizer.encode(amino_acid, add_special_tokens=False)[0]
        for amino_acid in CANONICAL_AMINO_ACIDS
    ]
    return torch.tensor(token_ids, dtype=torch.long)


def compute_categorical_jacobian(
    wildtype_input_ids: Tensor,
    wildtype_logits: Tensor,
    canonical_token_ids: Tensor,
    logits_fn: Callable[[Tensor], Tensor],
    mutation_batch_size: int = 20,
) -> Tensor:
    """Compute the categorical Jacobian via batched finite differences.

    Args:
        wildtype_input_ids: Wild-type token IDs including special tokens,
            shape ``(T,)``.
        wildtype_logits: Wild-type logits restricted to canonical amino-acid
            output channels, shape ``(L, A)``.
        canonical_token_ids: Canonical amino-acid token IDs, shape ``(A,)``.
        logits_fn: Function mapping a batch of mutated token IDs ``(B, T)``
            to canonical-channel logits ``(B, L, A)``.
        mutation_batch_size: Number of mutant sequences per forward pass.

    Returns:
        Tensor of shape ``(L, A, L, A)`` containing ``mutant - wildtype``
        logit deltas for every input-position/amino-acid mutation.
    """
    if mutation_batch_size < 1:
        raise ValueError("mutation_batch_size must be >= 1")

    if wildtype_input_ids.ndim != 1:
        raise ValueError("wildtype_input_ids must have shape (T,)")
    if wildtype_logits.ndim != 2:
        raise ValueError("wildtype_logits must have shape (L, A)")
    if canonical_token_ids.ndim != 1:
        raise ValueError("canonical_token_ids must have shape (A,)")

    seq_len, alphabet_size = wildtype_logits.shape
    if canonical_token_ids.shape[0] != alphabet_size:
        raise ValueError("canonical_token_ids length must match wildtype_logits.shape[1]")
    if wildtype_input_ids.shape[0] < seq_len + 2:
        raise ValueError("wildtype_input_ids must include leading and trailing special tokens")

    input_ids_cpu = wildtype_input_ids.cpu()
    canonical_ids_cpu = canonical_token_ids.cpu()
    wildtype_logits_cpu = wildtype_logits.cpu().float()

    jacobian = torch.empty((seq_len, alphabet_size, seq_len, alphabet_size), dtype=torch.float32)
    n_mutants = seq_len * alphabet_size

    for start in range(0, n_mutants, mutation_batch_size):
        stop = min(start + mutation_batch_size, n_mutants)
        batch_size = stop - start
        flat_indices = torch.arange(start, stop, dtype=torch.long)
        position_indices = torch.div(flat_indices, alphabet_size, rounding_mode="floor")
        amino_acid_indices = torch.remainder(flat_indices, alphabet_size)

        batch_input_ids = input_ids_cpu.unsqueeze(0).repeat(batch_size, 1)
        batch_rows = torch.arange(batch_size, dtype=torch.long)
        batch_input_ids[batch_rows, position_indices + 1] = canonical_ids_cpu[amino_acid_indices]

        batch_logits = logits_fn(batch_input_ids).cpu().float()
        expected_shape = (batch_size, seq_len, alphabet_size)
        if tuple(batch_logits.shape) != expected_shape:
            raise ValueError(
                f"logits_fn returned shape {tuple(batch_logits.shape)}, expected {expected_shape}"
            )

        jacobian[position_indices, amino_acid_indices] = (
            batch_logits - wildtype_logits_cpu.unsqueeze(0)
        )

    return jacobian


def center_categorical_jacobian(categorical_jacobian: Tensor, copy: bool = True) -> Tensor:
    """Mean-center a categorical Jacobian across each of its four axes."""
    centered = categorical_jacobian.clone() if copy else categorical_jacobian
    centered = centered.float()
    for dim in range(4):
        centered -= centered.mean(dim=dim, keepdim=True)
    return centered


def symmetrize_categorical_jacobian(categorical_jacobian: Tensor) -> Tensor:
    """Symmetrize a categorical Jacobian over input/output residue pairs."""
    return (categorical_jacobian + categorical_jacobian.permute(2, 3, 0, 1)) / 2


def categorical_jacobian_to_contact_map(
    categorical_jacobian: Tensor,
    center: bool = True,
    symmetrize: bool = True,
    apc: bool = True,
    copy: bool = True,
) -> Tensor:
    """Reduce a categorical Jacobian to an ``(L, L)`` contact score map.

    Args:
        categorical_jacobian: Input tensor of shape ``(L, A, L, A)``.
        center: If True, mean-center across all four axes.
        symmetrize: If True, symmetrize both the 4D tensor and final contact map.
        apc: If True, apply APC to the reduced contact map.
        copy: If True, operate on a copy of the input tensor.

    Returns:
        Contact score matrix of shape ``(L, L)``.
    """
    jacobian = categorical_jacobian.clone() if copy else categorical_jacobian
    jacobian = jacobian.float()

    if center:
        jacobian = center_categorical_jacobian(jacobian, copy=False)
    if symmetrize:
        jacobian = symmetrize_categorical_jacobian(jacobian)

    contact_scores = torch.sqrt(torch.square(jacobian).sum(dim=(1, 3)))
    contact_scores.fill_diagonal_(0.0)

    if apc:
        contact_scores = apply_apc(contact_scores)
    if symmetrize:
        contact_scores = (contact_scores + contact_scores.T) / 2

    return contact_scores


def build_structure_pair_score_data(
    pred_contacts: Tensor,
    true_contacts: Tensor,
    seq_len: int,
    min_seq_sep: int = 6,
) -> StructurePairScoreData:
    """Flatten a contact score matrix into valid long-range residue pairs."""
    seq_len_actual = true_contacts.shape[0]
    i_indices, j_indices = torch.triu_indices(seq_len_actual, seq_len_actual, offset=min_seq_sep)

    scores = pred_contacts[i_indices, j_indices]
    labels = true_contacts[i_indices, j_indices]

    return StructurePairScoreData(
        scores=scores,
        labels=labels,
        seq_len=seq_len,
        pair_indices=(i_indices, j_indices),
    )


def compute_mean_pair_score_precision_at_l(
    structures: list[StructurePairScoreData],
    l_divisor: int = 1,
    min_seq_sep: int = 6,
) -> float:
    """Average precision@L over per-structure pair-score predictions."""
    precisions: list[float] = []
    for structure in structures:
        pred = torch.zeros(structure.seq_len, structure.seq_len)
        pred[structure.pair_indices[0], structure.pair_indices[1]] = structure.scores

        true = torch.zeros(structure.seq_len, structure.seq_len)
        true[structure.pair_indices[0], structure.pair_indices[1]] = structure.labels

        precision = compute_precision_at_l(pred, true, structure.seq_len, min_seq_sep, l_divisor)
        precisions.append(precision)

    if not precisions:
        return 0.0
    return sum(precisions) / len(precisions)
