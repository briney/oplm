"""Unit tests for the categorical-Jacobian contact metric.

These are fast, model-free checks on the pure-tensor pieces that now back the
sole structure precision@L metric: the batched finite-difference Jacobian, the
``(L, A, L, A)`` → ``(L, L)`` reduction, and pair-score precision@L.
"""

from __future__ import annotations

import pytest
import torch

from oplm.eval.metrics.categorical_jacobian import (
    build_structure_pair_score_data,
    categorical_jacobian_to_contact_map,
    compute_categorical_jacobian,
    compute_mean_pair_score_precision_at_l,
)


@pytest.mark.parametrize("mutation_batch_size", [1, 5, 12])
def test_compute_categorical_jacobian_matches_known_deltas(mutation_batch_size: int) -> None:
    """A linear, position-local ``logits_fn`` yields an analytically known Jacobian.

    With ``logit[b, j, c] = residue_token[b, j]`` (independent of ``c``), mutating
    input position ``p`` to amino acid ``a`` only changes the output at ``j == p``,
    by exactly ``canonical_token_ids[a] - wildtype_token``. The result is also
    invariant to the mutation batch size.
    """
    seq_len, alphabet_size = 4, 3
    canonical_token_ids = torch.tensor([5, 6, 7], dtype=torch.long)
    wildtype_token = int(canonical_token_ids[0])  # residues all set to the first AA

    # (T,) = (L + 2,) with leading/trailing special tokens (cls=0, eos=2).
    wildtype_input_ids = torch.tensor([0, *([wildtype_token] * seq_len), 2], dtype=torch.long)
    wildtype_logits = torch.full((seq_len, alphabet_size), float(wildtype_token))

    def logits_fn(batch_input_ids: torch.Tensor) -> torch.Tensor:
        residues = batch_input_ids[:, 1 : seq_len + 1].float()  # (B, L)
        return residues.unsqueeze(-1).expand(-1, -1, alphabet_size).clone()  # (B, L, A)

    jacobian = compute_categorical_jacobian(
        wildtype_input_ids=wildtype_input_ids,
        wildtype_logits=wildtype_logits,
        canonical_token_ids=canonical_token_ids,
        logits_fn=logits_fn,
        mutation_batch_size=mutation_batch_size,
    )

    expected = torch.zeros(seq_len, alphabet_size, seq_len, alphabet_size)
    for p in range(seq_len):
        for a in range(alphabet_size):
            expected[p, a, p, :] = float(canonical_token_ids[a]) - wildtype_token

    assert jacobian.shape == (seq_len, alphabet_size, seq_len, alphabet_size)
    assert torch.allclose(jacobian, expected)


def test_contact_map_reduction_is_symmetric() -> None:
    """The reduced contact map is ``(L, L)`` and symmetric (default APC on)."""
    torch.manual_seed(0)
    seq_len, alphabet_size = 8, 20
    jacobian = torch.randn(seq_len, alphabet_size, seq_len, alphabet_size)

    contact_map = categorical_jacobian_to_contact_map(jacobian)

    assert contact_map.shape == (seq_len, seq_len)
    assert torch.allclose(contact_map, contact_map.T, atol=1e-6)


def test_contact_map_reduction_zero_diagonal_without_apc() -> None:
    """Without APC the self-contact diagonal stays exactly zero."""
    torch.manual_seed(1)
    seq_len, alphabet_size = 8, 20
    jacobian = torch.randn(seq_len, alphabet_size, seq_len, alphabet_size)

    contact_map = categorical_jacobian_to_contact_map(jacobian, apc=False)

    assert torch.allclose(contact_map.diagonal(), torch.zeros(seq_len))


def test_mean_pair_score_precision_is_one_for_perfect_ranking() -> None:
    """A contact map equal to ground truth scores P@L = 1.0."""
    seq_len, min_seq_sep = 24, 6
    true_contacts = torch.zeros(seq_len, seq_len)
    # Populate several long-range diagonal bands so #contacts > L among valid pairs.
    for offset in (6, 7, 8):
        idx = torch.arange(seq_len - offset)
        true_contacts[idx, idx + offset] = 1.0

    pred_contacts = true_contacts.clone()  # perfect prediction
    data = build_structure_pair_score_data(pred_contacts, true_contacts, seq_len, min_seq_sep)

    precision = compute_mean_pair_score_precision_at_l([data], l_divisor=1, min_seq_sep=min_seq_sep)

    assert precision == pytest.approx(1.0)


def test_mean_pair_score_precision_empty_is_zero() -> None:
    """No structures yields 0.0 rather than dividing by zero."""
    assert compute_mean_pair_score_precision_at_l([]) == 0.0
