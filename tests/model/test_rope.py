"""Tests for rotary position embeddings."""

from __future__ import annotations

import torch

from oplm.model.rope import PartialRotaryEmbedding, RotaryEmbedding, rotate_half


class TestRotateHalf:
    def test_shape_preserved(self) -> None:
        x = torch.randn(2, 4, 8, 64)
        assert rotate_half(x).shape == x.shape

    def test_double_rotation_negates(self) -> None:
        """Rotating twice should negate the input."""
        x = torch.randn(2, 4, 8, 64)
        torch.testing.assert_close(rotate_half(rotate_half(x)), -x)


class TestRotaryEmbedding:
    def test_output_shapes(self) -> None:
        rope = RotaryEmbedding(dim=64, max_seq_len=128)
        q = torch.randn(2, 16, 8, 64)  # (B, T, H, D)
        k = torch.randn(2, 16, 4, 64)  # (B, T, KV_H, D)
        q_rot, k_rot = rope(q, k)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_preserves_norm(self) -> None:
        """RoPE is an orthogonal rotation and should preserve vector norms."""
        rope = RotaryEmbedding(dim=64, max_seq_len=128)
        q = torch.randn(2, 16, 8, 64)
        k = torch.randn(2, 16, 4, 64)
        q_rot, k_rot = rope(q, k)
        torch.testing.assert_close(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(k.norm(dim=-1), k_rot.norm(dim=-1), atol=1e-4, rtol=1e-4)

    def test_with_position_ids(self) -> None:
        rope = RotaryEmbedding(dim=64, max_seq_len=128)
        q = torch.randn(2, 10, 8, 64)
        k = torch.randn(2, 10, 4, 64)
        pos = torch.arange(10).unsqueeze(0).expand(2, -1)  # (2, 10)
        q_rot, k_rot = rope(q, k, position_ids=pos)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_position_ids_matches_implicit(self) -> None:
        """Explicit sequential position_ids should match implicit (no position_ids)."""
        rope = RotaryEmbedding(dim=64, max_seq_len=128)
        q = torch.randn(2, 10, 8, 64)
        k = torch.randn(2, 10, 4, 64)

        q_impl, k_impl = rope(q, k)
        pos = torch.arange(10).unsqueeze(0).expand(2, -1)
        q_expl, k_expl = rope(q, k, position_ids=pos)

        torch.testing.assert_close(q_impl, q_expl, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(k_impl, k_expl, atol=1e-5, rtol=1e-5)

    def test_cache_extends_for_longer_seq(self) -> None:
        rope = RotaryEmbedding(dim=64, max_seq_len=32)
        q = torch.randn(1, 64, 4, 64)  # seq_len > max_seq_len
        k = torch.randn(1, 64, 2, 64)
        q_rot, k_rot = rope(q, k)
        assert q_rot.shape == q.shape


class TestPartialRotaryEmbedding:
    def test_output_shapes(self) -> None:
        partial = PartialRotaryEmbedding(rope_dim=32, nope_dim=32, max_seq_len=128)
        q = torch.randn(2, 16, 8, 64)  # head_dim = nope_dim + rope_dim = 64
        k = torch.randn(2, 16, 4, 64)
        q_out, k_out = partial(q, k)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    def test_nope_portion_unchanged(self) -> None:
        """The NoPE portion should pass through without modification."""
        nope_dim = 32
        partial = PartialRotaryEmbedding(rope_dim=32, nope_dim=nope_dim, max_seq_len=128)
        q = torch.randn(2, 16, 8, 64)
        k = torch.randn(2, 16, 4, 64)
        q_out, k_out = partial(q, k)
        torch.testing.assert_close(q[..., :nope_dim], q_out[..., :nope_dim])
        torch.testing.assert_close(k[..., :nope_dim], k_out[..., :nope_dim])

    def test_rope_portion_rotated(self) -> None:
        """The RoPE portion should differ from the original."""
        nope_dim = 32
        partial = PartialRotaryEmbedding(rope_dim=32, nope_dim=nope_dim, max_seq_len=128)
        q = torch.randn(2, 16, 8, 64)
        k = torch.randn(2, 16, 4, 64)
        q_out, _ = partial(q, k)
        # The rope portion should be different (not identical to input)
        assert not torch.allclose(q[..., nope_dim:], q_out[..., nope_dim:])
