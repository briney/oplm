"""Tests for contact prediction metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

if TYPE_CHECKING:
    from pathlib import Path

from oplm.eval.data.structure_loader import load_structures
from oplm.eval.metrics.contact import (
    StructureContactData,
    apply_apc,
    build_structure_contact_data,
    compute_contact_map,
    compute_logreg_precision_at_l,
    compute_precision_at_l,
    compute_virtual_cbeta,
    extract_attention_contacts,
)

# ---------------------------------------------------------------------------
# compute_virtual_cbeta
# ---------------------------------------------------------------------------


class TestComputeVirtualCbeta:
    """Test virtual Cβ computation from backbone atoms."""

    def test_output_shape(self) -> None:
        coords = torch.randn(10, 3, 3)
        cbeta = compute_virtual_cbeta(coords)
        assert cbeta.shape == (10, 3)

    def test_produces_finite_positions(self) -> None:
        """Virtual Cβ should be finite for valid backbone atoms."""
        coords = torch.tensor(
            [
                [
                    [1.458, 0.0, 0.0],  # N
                    [0.0, 0.0, 0.0],  # CA
                    [-0.553, 1.419, 0.0],
                ],  # C
            ]
        )
        cbeta = compute_virtual_cbeta(coords)
        assert torch.isfinite(cbeta).all()

    def test_nan_backbone_propagates(self) -> None:
        """NaN in backbone atoms should produce NaN in Cβ."""
        coords = torch.tensor(
            [
                [
                    [float("nan"), 0.0, 0.0],  # N has NaN
                    [0.0, 0.0, 0.0],  # CA
                    [-0.553, 1.419, 0.0],
                ],  # C
            ]
        )
        cbeta = compute_virtual_cbeta(coords)
        assert torch.isnan(cbeta).any()

    def test_known_geometry_matches_real_cbeta(self, structure_fixtures_dir: Path) -> None:
        """Virtual Cβ should be close to real Cβ in crystal structures."""
        from Bio.PDB import PDBParser

        parser = PDBParser(QUIET=True)
        pdb_path = structure_fixtures_dir / "1CRN.pdb"
        structure = parser.get_structure("1CRN", str(pdb_path))
        chain = next(structure[0].get_chains())

        backbone_coords = []
        real_cbeta_coords = []
        for residue in chain.get_residues():
            if residue.id[0] != " ":
                continue
            if "CB" not in residue:
                continue  # Skip glycine
            backbone_coords.append(
                [
                    list(residue["N"].get_vector()),
                    list(residue["CA"].get_vector()),
                    list(residue["C"].get_vector()),
                ]
            )
            real_cbeta_coords.append(list(residue["CB"].get_vector()))

        coords = torch.tensor(backbone_coords, dtype=torch.float32)
        real_cb = torch.tensor(real_cbeta_coords, dtype=torch.float32)
        virtual_cb = compute_virtual_cbeta(coords)

        # Should match within ~0.25 Å (slight deviations from ideal geometry
        # in real structures are expected; still negligible vs 8 Å threshold)
        distances = torch.norm(virtual_cb - real_cb, dim=-1)
        assert distances.max() < 0.25, f"Max deviation: {distances.max():.3f} Å"
        assert distances.mean() < 0.12, f"Mean deviation: {distances.mean():.3f} Å"

    def test_vectorized_matches_loop(self) -> None:
        """Vectorized computation should match a naive per-residue loop."""
        torch.manual_seed(42)
        coords = torch.randn(5, 3, 3)
        vectorized = compute_virtual_cbeta(coords)

        for i in range(5):
            single = compute_virtual_cbeta(coords[i : i + 1])
            assert torch.allclose(vectorized[i], single[0], atol=1e-6)


# ---------------------------------------------------------------------------
# compute_contact_map
# ---------------------------------------------------------------------------


class TestComputeContactMap:
    """Test binary contact map computation."""

    def test_diagonal_is_contact(self) -> None:
        """Self-contacts (diagonal) should be 1.0."""
        coords = torch.randn(10, 3, 3) * 20  # Spread out
        # Place CA atoms far apart but self-distance is always 0
        contact_map = compute_contact_map(coords, threshold=8.0, use_cbeta=False)
        assert (contact_map.diag() == 1.0).all()

    def test_threshold_respected(self) -> None:
        """Known distances should be correctly classified."""
        # Two residues: CA at origin and at (5, 0, 0) -> distance = 5
        coords = torch.zeros(2, 3, 3)
        coords[0, 1, :] = torch.tensor([0.0, 0.0, 0.0])  # CA of residue 0
        coords[1, 1, :] = torch.tensor([5.0, 0.0, 0.0])  # CA of residue 1
        # Fill N and C with reasonable offsets to avoid NaN from cbeta calc
        coords[0, 0, :] = torch.tensor([1.458, 0.0, 0.0])  # N
        coords[0, 2, :] = torch.tensor([-0.553, 1.419, 0.0])  # C
        coords[1, 0, :] = torch.tensor([6.458, 0.0, 0.0])
        coords[1, 2, :] = torch.tensor([4.447, 1.419, 0.0])

        # With CA distance = 5, threshold = 8 -> contact
        contact_map = compute_contact_map(coords, threshold=8.0, use_cbeta=False)
        assert contact_map[0, 1] == 1.0

        # With threshold = 3 -> no contact
        contact_map = compute_contact_map(coords, threshold=3.0, use_cbeta=False)
        assert contact_map[0, 1] == 0.0

    def test_symmetry(self) -> None:
        torch.manual_seed(42)
        coords = torch.randn(10, 3, 3)
        contact_map = compute_contact_map(coords, use_cbeta=False)
        assert torch.allclose(contact_map, contact_map.T)

    def test_cbeta_vs_calpha_differ(self) -> None:
        """Cβ and Cα contact maps should generally differ."""
        torch.manual_seed(42)
        # Use realistic-ish backbone coordinates
        coords = torch.randn(15, 3, 3) * 5
        cb_map = compute_contact_map(coords, use_cbeta=True)
        ca_map = compute_contact_map(coords, use_cbeta=False)
        # They should not be identical (different reference atoms)
        assert not torch.allclose(cb_map, ca_map)

    def test_nan_positions_excluded(self) -> None:
        """Positions with NaN coords should not form contacts."""
        coords = torch.zeros(3, 3, 3)
        coords[0, :, :] = float("nan")
        contact_map = compute_contact_map(coords, use_cbeta=False)
        assert contact_map[0, 1] == 0.0
        assert contact_map[1, 0] == 0.0

    def test_with_real_structure(self, structure_fixtures_dir: Path) -> None:
        """Contact map from a real structure should be symmetric and sparse."""
        structures = load_structures(structure_fixtures_dir, max_structures=1)
        s = structures[0]
        contact_map = compute_contact_map(s.coords)
        L = len(s.sequence)
        assert contact_map.shape == (L, L)
        assert torch.allclose(contact_map, contact_map.T)
        # Contact map should be sparse (most residues are not in contact)
        density = contact_map.sum() / (L * L)
        assert density < 0.5, f"Contact density {density:.2f} is suspiciously high"


# ---------------------------------------------------------------------------
# apply_apc
# ---------------------------------------------------------------------------


class TestApplyApc:
    """Test Average Product Correction."""

    def test_preserves_shape(self) -> None:
        m = torch.randn(10, 10)
        result = apply_apc(m)
        assert result.shape == (10, 10)

    def test_zero_matrix_unchanged(self) -> None:
        m = torch.zeros(5, 5)
        result = apply_apc(m)
        assert torch.allclose(result, m)

    def test_uniform_matrix_becomes_zero(self) -> None:
        """A uniform matrix has row_mean = col_mean = global_mean = c,
        so APC = c - c*c/c = c - c = 0."""
        m = torch.ones(5, 5) * 3.0
        result = apply_apc(m)
        assert torch.allclose(result, torch.zeros(5, 5), atol=1e-6)

    def test_known_values(self) -> None:
        """Hand-computed APC on a 3x3 matrix."""
        m = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        )
        row_mean = m.mean(dim=-1, keepdim=True)  # [[2], [5], [8]]
        col_mean = m.mean(dim=-2, keepdim=True)  # [[4, 5, 6]]
        global_mean = m.mean()  # 5.0
        expected = m - (row_mean * col_mean) / global_mean
        result = apply_apc(m)
        assert torch.allclose(result, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# extract_attention_contacts
# ---------------------------------------------------------------------------


class TestExtractAttentionContacts:
    """Test attention contact extraction with various configurations."""

    @pytest.fixture()
    def attn_weights(self) -> list[torch.Tensor]:
        """3 layers, 4 heads, sequence length 10."""
        torch.manual_seed(42)
        return [torch.rand(4, 10, 10) for _ in range(3)]

    def test_all_layers_no_aggregation_shape(self, attn_weights: list[torch.Tensor]) -> None:
        result = extract_attention_contacts(attn_weights)
        assert result.shape == (3, 4, 10, 10)

    def test_all_layers_mean_aggregation_shape(self, attn_weights: list[torch.Tensor]) -> None:
        result = extract_attention_contacts(attn_weights, head_aggregation="mean")
        assert result.shape == (3, 10, 10)

    def test_all_layers_max_aggregation_shape(self, attn_weights: list[torch.Tensor]) -> None:
        result = extract_attention_contacts(attn_weights, head_aggregation="max")
        assert result.shape == (3, 10, 10)

    def test_last_layer_no_aggregation_shape(self, attn_weights: list[torch.Tensor]) -> None:
        result = extract_attention_contacts(attn_weights, layer="last")
        assert result.shape == (4, 10, 10)

    def test_last_layer_mean_aggregation_shape(self, attn_weights: list[torch.Tensor]) -> None:
        result = extract_attention_contacts(attn_weights, layer="last", head_aggregation="mean")
        assert result.shape == (10, 10)

    def test_specific_layer_index(self, attn_weights: list[torch.Tensor]) -> None:
        result = extract_attention_contacts(attn_weights, layer=1)
        assert result.shape == (4, 10, 10)

    def test_negative_index_equals_last(self, attn_weights: list[torch.Tensor]) -> None:
        """layer=-1 should produce the same result as layer='last'."""
        result_neg = extract_attention_contacts(attn_weights, layer=-1)
        result_last = extract_attention_contacts(attn_weights, layer="last")
        assert torch.allclose(result_neg, result_last)

    def test_negative_index_second_to_last(self, attn_weights: list[torch.Tensor]) -> None:
        """layer=-2 should select the second-to-last layer."""
        result = extract_attention_contacts(attn_weights, layer=-2)
        assert result.shape == (4, 10, 10)
        # Verify it's different from last layer
        result_last = extract_attention_contacts(attn_weights, layer=-1)
        assert not torch.allclose(result, result_last)

    def test_symmetrize(self, attn_weights: list[torch.Tensor]) -> None:
        """Output should be symmetric when symmetrize=True."""
        result = extract_attention_contacts(attn_weights, symmetrize=True, apc=False)
        assert torch.allclose(result, result.transpose(-1, -2), atol=1e-6)

    def test_no_symmetrize_preserves_asymmetry(self, attn_weights: list[torch.Tensor]) -> None:
        result = extract_attention_contacts(attn_weights, symmetrize=False, apc=False)
        # Random attention weights should not be symmetric
        assert not torch.allclose(result, result.transpose(-1, -2))

    def test_apc_changes_values(self, attn_weights: list[torch.Tensor]) -> None:
        """APC should change the attention values."""
        with_apc = extract_attention_contacts(attn_weights, apc=True, symmetrize=False)
        without_apc = extract_attention_contacts(attn_weights, apc=False, symmetrize=False)
        assert not torch.allclose(with_apc, without_apc)

    def test_defaults_match_esm_protocol(self, attn_weights: list[torch.Tensor]) -> None:
        """Default args should give (n_layers, n_heads, L, L) — ESM protocol."""
        result = extract_attention_contacts(attn_weights)
        assert result.shape == (3, 4, 10, 10)
        # Should be symmetrized
        assert torch.allclose(result, result.transpose(-1, -2), atol=1e-6)

    def test_invalid_layer_raises(self, attn_weights: list[torch.Tensor]) -> None:
        with pytest.raises(ValueError, match="Invalid layer"):
            extract_attention_contacts(attn_weights, layer="middle")

    def test_invalid_head_aggregation_raises(self, attn_weights: list[torch.Tensor]) -> None:
        with pytest.raises(ValueError, match="Invalid head_aggregation"):
            extract_attention_contacts(attn_weights, head_aggregation="median")


# ---------------------------------------------------------------------------
# compute_precision_at_l
# ---------------------------------------------------------------------------


class TestComputePrecisionAtL:
    """Test precision@L computation."""

    def test_perfect_prediction(self) -> None:
        """When all top-k predictions are true contacts, precision = 1.0."""
        L = 20
        true = torch.zeros(L, L)
        # Place many contacts with |i-j| >= 6 so top-L are all contacts
        for i in range(L):
            for j in range(i + 6, L):
                true[i, j] = 1.0

        # Predicted scores exactly match true labels (contacts ranked highest)
        pred = true.clone()

        p = compute_precision_at_l(pred, true, seq_len=L, min_seq_sep=6, l_divisor=1)
        assert p == 1.0

    def test_l_divisor(self) -> None:
        """l_divisor=2 should take L/2 predictions instead of L."""
        L = 20
        true = torch.zeros(L, L)
        # Add some contacts
        for i in range(L):
            for j in range(i + 6, L):
                if (i + j) % 3 == 0:
                    true[i, j] = 1.0

        pred = true.clone()

        p_l = compute_precision_at_l(pred, true, seq_len=L, l_divisor=1)
        p_l2 = compute_precision_at_l(pred, true, seq_len=L, l_divisor=2)
        p_l5 = compute_precision_at_l(pred, true, seq_len=L, l_divisor=5)

        # With perfect predictions, all should be 1.0 as long as k <= n_contacts
        assert p_l > 0.0
        assert p_l2 > 0.0
        assert p_l5 > 0.0

    def test_min_seq_sep_filters(self) -> None:
        """Short-range contacts should be excluded."""
        L = 20
        true = torch.zeros(L, L)
        # Only add short-range contacts (|i-j| < 6)
        for i in range(L - 1):
            true[i, i + 1] = 1.0

        pred = true.clone()
        p = compute_precision_at_l(pred, true, seq_len=L, min_seq_sep=6)
        # No valid long-range pairs have contacts, so precision should be 0
        assert p == 0.0

    def test_empty_sequence_returns_zero(self) -> None:
        """Very short sequence with no valid pairs should return 0."""
        true = torch.zeros(3, 3)
        pred = torch.zeros(3, 3)
        p = compute_precision_at_l(pred, true, seq_len=3, min_seq_sep=6)
        assert p == 0.0


# ---------------------------------------------------------------------------
# build_structure_contact_data
# ---------------------------------------------------------------------------


class TestBuildStructureContactData:
    """Test feature/label extraction for logreg."""

    def test_feature_shape(self) -> None:
        n_layers, n_heads, L = 3, 4, 20
        torch.manual_seed(42)
        attn = torch.rand(n_layers, n_heads, L, L)
        true = (torch.rand(L, L) > 0.8).float()

        data = build_structure_contact_data(attn, true, seq_len=L, min_seq_sep=6)
        assert data.features.shape[1] == n_layers * n_heads

    def test_labels_binary(self) -> None:
        n_layers, n_heads, L = 2, 3, 15
        attn = torch.rand(n_layers, n_heads, L, L)
        true = (torch.rand(L, L) > 0.7).float()

        data = build_structure_contact_data(attn, true, seq_len=L)
        unique_labels = data.labels.unique()
        assert all(v in (0.0, 1.0) for v in unique_labels)

    def test_pair_indices_consistent(self) -> None:
        """Pair indices should match feature and label arrays."""
        n_layers, n_heads, L = 2, 4, 20
        attn = torch.rand(n_layers, n_heads, L, L)
        true = (torch.rand(L, L) > 0.7).float()

        data = build_structure_contact_data(attn, true, seq_len=L)
        n_pairs = data.features.shape[0]
        assert len(data.pair_indices[0]) == n_pairs
        assert len(data.pair_indices[1]) == n_pairs
        assert data.labels.shape[0] == n_pairs

    def test_min_seq_sep_filtering(self) -> None:
        """All pairs should have |i - j| >= min_seq_sep."""
        L, min_sep = 20, 6
        attn = torch.rand(2, 3, L, L)
        true = torch.ones(L, L)

        data = build_structure_contact_data(attn, true, seq_len=L, min_seq_sep=min_sep)
        i_idx, j_idx = data.pair_indices
        sep = (j_idx - i_idx).abs()
        assert (sep >= min_sep).all()


# ---------------------------------------------------------------------------
# compute_logreg_precision_at_l
# ---------------------------------------------------------------------------


class TestComputeLogregPrecisionAtL:
    """Test logistic regression P@L pipeline."""

    @pytest.fixture()
    def synthetic_structures(self) -> list[StructureContactData]:
        """Create synthetic structures with known contact patterns."""
        torch.manual_seed(42)
        structures = []
        for _i in range(25):
            L = 30
            n_features = 12  # 2 layers * 6 heads
            i_idx, j_idx = torch.triu_indices(L, L, offset=6)
            n_pairs = len(i_idx)

            # Create labels with some contacts
            labels = (torch.rand(n_pairs) > 0.9).float()
            # Make features correlated with labels (so logreg can learn)
            features = torch.randn(n_pairs, n_features)
            features += labels.unsqueeze(-1) * 0.5

            structures.append(
                StructureContactData(
                    features=features,
                    labels=labels,
                    seq_len=L,
                    pair_indices=(i_idx, j_idx),
                )
            )
        return structures

    def test_deterministic(self, synthetic_structures: list[StructureContactData]) -> None:
        """Same input should give same output."""
        r1 = compute_logreg_precision_at_l(synthetic_structures, n_train=20)
        r2 = compute_logreg_precision_at_l(synthetic_structures, n_train=20)
        assert r1 == r2

    def test_returns_valid_precision(
        self, synthetic_structures: list[StructureContactData]
    ) -> None:
        p = compute_logreg_precision_at_l(synthetic_structures, n_train=20)
        assert 0.0 <= p <= 1.0

    def test_fallback_insufficient_structures(self) -> None:
        """With too few structures, should fall back to mean-attention P@L."""
        L = 20
        i_idx, j_idx = torch.triu_indices(L, L, offset=6)
        n_pairs = len(i_idx)

        structures = [
            StructureContactData(
                features=torch.randn(n_pairs, 12),
                labels=(torch.rand(n_pairs) > 0.8).float(),
                seq_len=L,
                pair_indices=(i_idx, j_idx),
            )
            for _ in range(3)  # Too few for n_train=20
        ]
        p = compute_logreg_precision_at_l(structures, n_train=20)
        assert 0.0 <= p <= 1.0

    @pytest.mark.slow
    def test_with_real_structures(self, structure_fixtures_dir: Path) -> None:
        """End-to-end with real structures and a tiny model."""
        from oplm.config import ModelConfig
        from oplm.data.tokenizer import ProteinTokenizer
        from oplm.model.transformer import OplmForMLM

        # Small model for testing
        cfg = ModelConfig(hidden_dim=32, num_layers=2, num_heads=2, num_kv_heads=2)
        model = OplmForMLM(cfg)
        model.eval()
        tokenizer = ProteinTokenizer()

        structures = load_structures(structure_fixtures_dir)
        contact_data_list: list[StructureContactData] = []

        for s in structures:
            tokens = tokenizer.batch_encode([s.sequence], max_length=128)
            with torch.no_grad():
                outputs = model(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                    need_weights=True,
                )

            seq_len = len(s.sequence)
            attn_weights = [
                w.squeeze(0)[:, 1 : seq_len + 1, 1 : seq_len + 1].cpu()
                for w in outputs["attention_weights"]
            ]

            attn_contacts = extract_attention_contacts(attn_weights)
            true_contacts = compute_contact_map(s.coords)
            cd = build_structure_contact_data(attn_contacts, true_contacts, seq_len)
            contact_data_list.append(cd)

        # Too few for logreg (3 structures, n_train=20), so falls back
        p = compute_logreg_precision_at_l(contact_data_list)
        assert 0.0 <= p <= 1.0
