"""Tests for categorical-Jacobian contact extraction."""

from __future__ import annotations

import torch

from oplm.data.tokenizer import ID_TO_TOKEN, ProteinTokenizer
from oplm.eval.metrics.categorical_jacobian import (
    CANONICAL_AMINO_ACIDS,
    build_structure_pair_score_data,
    categorical_jacobian_to_contact_map,
    center_categorical_jacobian,
    compute_categorical_jacobian,
    compute_mean_pair_score_precision_at_l,
    get_canonical_amino_acid_token_ids,
    symmetrize_categorical_jacobian,
)


class TestCanonicalAminoAcids:
    """Test canonical amino-acid token resolution."""

    def test_get_canonical_amino_acid_token_ids(self) -> None:
        tokenizer = ProteinTokenizer()
        token_ids = get_canonical_amino_acid_token_ids(tokenizer)
        tokens = tuple(ID_TO_TOKEN[token_id.item()] for token_id in token_ids)
        assert tokens == CANONICAL_AMINO_ACIDS


class TestCategoricalJacobian:
    """Test categorical-Jacobian extraction and reduction."""

    def test_compute_categorical_jacobian_is_batch_size_invariant(self) -> None:
        tokenizer = ProteinTokenizer()
        canonical_token_ids = get_canonical_amino_acid_token_ids(tokenizer)
        wildtype_input_ids = torch.tensor(tokenizer.encode("LAGVS"), dtype=torch.long)
        seq_len = wildtype_input_ids.shape[0] - 2

        position_grid = torch.arange(seq_len, dtype=torch.float32).view(1, seq_len, 1)
        amino_acid_grid = torch.arange(len(canonical_token_ids), dtype=torch.float32).view(1, 1, -1)

        def logits_fn(batch_input_ids: torch.Tensor) -> torch.Tensor:
            residues = batch_input_ids[:, 1:-1].float()
            sequence_sum = residues.sum(dim=-1, keepdim=True).unsqueeze(-1)
            return (
                residues.unsqueeze(-1) * 0.1
                + sequence_sum * 0.01
                + position_grid
                + amino_acid_grid * 0.001
            )

        wildtype_logits = logits_fn(wildtype_input_ids.unsqueeze(0))[0]
        batched = compute_categorical_jacobian(
            wildtype_input_ids=wildtype_input_ids,
            wildtype_logits=wildtype_logits,
            canonical_token_ids=canonical_token_ids,
            logits_fn=logits_fn,
            mutation_batch_size=20,
        )
        chunked = compute_categorical_jacobian(
            wildtype_input_ids=wildtype_input_ids,
            wildtype_logits=wildtype_logits,
            canonical_token_ids=canonical_token_ids,
            logits_fn=logits_fn,
            mutation_batch_size=7,
        )

        assert torch.allclose(batched, chunked)

    def test_center_categorical_jacobian_zeroes_axis_means(self) -> None:
        torch.manual_seed(42)
        jacobian = torch.randn(4, 3, 4, 3)
        centered = center_categorical_jacobian(jacobian)

        for dim in range(4):
            zeros = torch.zeros_like(centered.mean(dim=dim))
            assert torch.allclose(centered.mean(dim=dim), zeros, atol=1e-6)

    def test_symmetrize_categorical_jacobian_matches_transpose(self) -> None:
        torch.manual_seed(42)
        jacobian = torch.randn(4, 3, 4, 3)
        symmetrized = symmetrize_categorical_jacobian(jacobian)
        assert torch.allclose(symmetrized, symmetrized.permute(2, 3, 0, 1))

    def test_categorical_jacobian_to_contact_map_is_symmetric(self) -> None:
        jacobian = torch.zeros(4, 20, 4, 20)
        jacobian[0, 0, 3, 1] = 2.0
        jacobian[3, 1, 0, 0] = 2.0

        contact_map = categorical_jacobian_to_contact_map(
            jacobian,
            center=False,
            symmetrize=True,
            apc=False,
        )

        assert contact_map.shape == (4, 4)
        assert torch.allclose(contact_map, contact_map.T)
        assert contact_map[0, 3] > 0.0
        assert contact_map.diag().eq(0.0).all()

    def test_pair_score_precision_is_valid(self) -> None:
        seq_len = 20
        pred = torch.zeros(seq_len, seq_len)
        true = torch.zeros(seq_len, seq_len)
        for i in range(seq_len):
            for j in range(i + 6, seq_len):
                if (i + j) % 2 == 0:
                    pred[i, j] = 1.0
                    true[i, j] = 1.0

        structure = build_structure_pair_score_data(pred, true, seq_len=seq_len, min_seq_sep=6)
        precision = compute_mean_pair_score_precision_at_l(
            [structure],
            l_divisor=5,
            min_seq_sep=6,
        )

        assert 0.0 <= precision <= 1.0
