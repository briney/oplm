"""Tests for `oplm.model.rope` — RotaryEmbedding (full and partial RoPE)."""

from __future__ import annotations

import pytest
import torch

from oplm.model.rope import RotaryEmbedding


def _make_qk(
    batch: int = 2,
    heads: int = 4,
    seq_len: int = 8,
    head_dim: int = 16,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(batch, heads, seq_len, head_dim, generator=g, dtype=torch.float32).to(dtype)
    k = torch.randn(batch, heads, seq_len, head_dim, generator=g, dtype=torch.float32).to(dtype)
    return q, k


# ---------------------------------------------------------------------------
# Construction and buffer registration
# ---------------------------------------------------------------------------


def test_constructor_registers_nonpersistent_buffers():
    rope = RotaryEmbedding(head_dim=16, rope_dim=16, max_position_embeddings=32)
    # Non-persistent buffers do not appear in state_dict.
    sd = rope.state_dict()
    assert "inv_freq" not in sd
    assert "cos_cached" not in sd
    assert "sin_cached" not in sd
    # But they exist as buffers.
    assert rope.inv_freq.shape == (8,)
    assert rope.cos_cached.shape == (32, 8)
    assert rope.sin_cached.shape == (32, 8)


def test_constructor_rejects_odd_rope_dim():
    with pytest.raises(ValueError, match="even"):
        RotaryEmbedding(head_dim=16, rope_dim=3, max_position_embeddings=16)


def test_constructor_rejects_rope_dim_larger_than_head_dim():
    with pytest.raises(ValueError, match="rope_dim"):
        RotaryEmbedding(head_dim=16, rope_dim=32, max_position_embeddings=16)


def test_constructor_rejects_negative_rope_dim():
    with pytest.raises(ValueError, match="rope_dim"):
        RotaryEmbedding(head_dim=16, rope_dim=-2, max_position_embeddings=16)


def test_inv_freq_matches_formula():
    head_dim = 8
    rope_dim = 8
    base = 10000.0
    rope = RotaryEmbedding(
        head_dim=head_dim, rope_dim=rope_dim, max_position_embeddings=4, base=base
    )
    expected = base ** (-torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim)
    assert torch.allclose(rope.inv_freq, expected)


# ---------------------------------------------------------------------------
# Position-0 identity, norm preservation, dtype passthrough
# ---------------------------------------------------------------------------


def test_position_zero_is_identity():
    rope = RotaryEmbedding(head_dim=16, rope_dim=16, max_position_embeddings=4)
    q, k = _make_qk(seq_len=1, head_dim=16)
    q_out, k_out = rope.apply_rotary(q, k)
    assert torch.allclose(q_out, q, atol=1e-6)
    assert torch.allclose(k_out, k, atol=1e-6)


def test_norm_preservation_per_channel_pair():
    """Rotation is an isometry on each (x_even, x_odd) pair."""
    rope = RotaryEmbedding(head_dim=16, rope_dim=16, max_position_embeddings=32)
    q, k = _make_qk(seq_len=8, head_dim=16)
    q_out, k_out = rope.apply_rotary(q, k)
    # ‖(x_even, x_odd)‖ is preserved by the rotation.
    q_pair_sq_in = q[..., 0::2].pow(2) + q[..., 1::2].pow(2)
    q_pair_sq_out = q_out[..., 0::2].pow(2) + q_out[..., 1::2].pow(2)
    assert torch.allclose(q_pair_sq_in, q_pair_sq_out, atol=1e-5)
    k_pair_sq_in = k[..., 0::2].pow(2) + k[..., 1::2].pow(2)
    k_pair_sq_out = k_out[..., 0::2].pow(2) + k_out[..., 1::2].pow(2)
    assert torch.allclose(k_pair_sq_in, k_pair_sq_out, atol=1e-5)


def test_full_head_norm_preserved():
    rope = RotaryEmbedding(head_dim=16, rope_dim=16, max_position_embeddings=32)
    q, k = _make_qk(seq_len=8, head_dim=16)
    q_out, _ = rope.apply_rotary(q, k)
    assert torch.allclose(q.norm(dim=-1), q_out.norm(dim=-1), atol=1e-5)


def test_preserves_input_dtype_bf16():
    rope = RotaryEmbedding(head_dim=16, rope_dim=16, max_position_embeddings=32)
    q, k = _make_qk(seq_len=4, head_dim=16, dtype=torch.bfloat16)
    q_out, k_out = rope.apply_rotary(q, k)
    assert q_out.dtype == torch.bfloat16
    assert k_out.dtype == torch.bfloat16


def test_preserves_input_dtype_fp16():
    rope = RotaryEmbedding(head_dim=16, rope_dim=16, max_position_embeddings=32)
    q, k = _make_qk(seq_len=4, head_dim=16, dtype=torch.float16)
    q_out, k_out = rope.apply_rotary(q, k)
    assert q_out.dtype == torch.float16
    assert k_out.dtype == torch.float16


# ---------------------------------------------------------------------------
# Partial RoPE / NoPE split
# ---------------------------------------------------------------------------


def test_partial_rope_passes_nope_channels_unchanged():
    rope = RotaryEmbedding(head_dim=64, rope_dim=32, max_position_embeddings=32)
    q, k = _make_qk(seq_len=8, head_dim=64)
    q_out, k_out = rope.apply_rotary(q, k)
    # Trailing nope_dim=32 channels are bit-identical to the input.
    assert torch.equal(q_out[..., 32:], q[..., 32:])
    assert torch.equal(k_out[..., 32:], k[..., 32:])


def test_partial_rope_rotates_only_rope_channels():
    rope = RotaryEmbedding(head_dim=64, rope_dim=32, max_position_embeddings=32)
    q, k = _make_qk(seq_len=8, head_dim=64)
    q_out, _ = rope.apply_rotary(q, k)
    # Leading rope_dim=32 channels differ from input at positions > 0.
    assert not torch.allclose(q_out[..., 1:, :32], q[..., 1:, :32], atol=1e-3)


def test_rope_dim_zero_is_full_nope_passthrough():
    rope = RotaryEmbedding(head_dim=16, rope_dim=0, max_position_embeddings=32)
    q, k = _make_qk(seq_len=8, head_dim=16)
    q_out, k_out = rope.apply_rotary(q, k)
    assert q_out is q
    assert k_out is k


# ---------------------------------------------------------------------------
# Buffer extension
# ---------------------------------------------------------------------------


def test_buffer_extends_on_longer_input():
    rope = RotaryEmbedding(head_dim=16, rope_dim=16, max_position_embeddings=8)
    assert rope.cos_cached.shape[0] == 8
    q, k = _make_qk(seq_len=20, head_dim=16)
    q_out, _ = rope.apply_rotary(q, k)
    assert q_out.shape == q.shape
    assert rope.cos_cached.shape[0] >= 20
    assert rope.sin_cached.shape[0] >= 20
    assert rope.max_position_embeddings >= 20


def test_buffer_extension_preserves_existing_positions():
    """After extension, the cos/sin values for old positions are unchanged."""
    rope = RotaryEmbedding(head_dim=16, rope_dim=16, max_position_embeddings=8)
    old_cos = rope.cos_cached.clone()
    old_sin = rope.sin_cached.clone()
    q, k = _make_qk(seq_len=20, head_dim=16)
    rope.apply_rotary(q, k)
    assert torch.allclose(rope.cos_cached[:8], old_cos, atol=1e-6)
    assert torch.allclose(rope.sin_cached[:8], old_sin, atol=1e-6)


def test_no_extension_when_within_cache():
    rope = RotaryEmbedding(head_dim=16, rope_dim=16, max_position_embeddings=64)
    cos_before = rope.cos_cached
    q, k = _make_qk(seq_len=16, head_dim=16)
    rope.apply_rotary(q, k)
    # Same tensor object — no recomputation.
    assert rope.cos_cached is cos_before


# ---------------------------------------------------------------------------
# Per-position phase sanity
# ---------------------------------------------------------------------------


def test_rotation_phase_matches_inv_freq():
    """At position p, channel-pair i rotates by p * inv_freq[i]."""
    rope = RotaryEmbedding(head_dim=8, rope_dim=8, max_position_embeddings=16)
    # Place a unit vector in a single channel-pair at every position; compare
    # to the closed-form (cos, sin) rotation.
    q = torch.zeros(1, 1, 16, 8)
    q[..., 0] = 1.0  # x_even of pair 0
    k = torch.zeros_like(q)
    q_out, _ = rope.apply_rotary(q, k)
    positions = torch.arange(16, dtype=torch.float32)
    expected_even = torch.cos(positions * rope.inv_freq[0])
    expected_odd = torch.sin(positions * rope.inv_freq[0])
    assert torch.allclose(q_out[0, 0, :, 0], expected_even, atol=1e-5)
    assert torch.allclose(q_out[0, 0, :, 1], expected_odd, atol=1e-5)
