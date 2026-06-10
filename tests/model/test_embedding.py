"""Tests for `oplm.model.embedding` — OplmEmbedding, mean_pool, cls_pool."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from oplm.model.embedding import OplmEmbedding, cls_pool, mean_pool
from oplm.model.norm import OplmLayerNorm, OplmRMSNorm


def _config(
    *,
    vocab_size: int = 33,
    hidden_size: int = 16,
    post_embed_norm: bool = False,
    norm_type: str = "layernorm",
    norm_eps: float = 1e-6,
    mask_dropout: bool = False,
    mask_dropout_reference_ratio: float = 0.12,
    mask_token_id: int = 32,
) -> SimpleNamespace:
    return SimpleNamespace(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        post_embed_norm=post_embed_norm,
        norm_type=norm_type,
        norm_eps=norm_eps,
        mask_dropout=mask_dropout,
        mask_dropout_reference_ratio=mask_dropout_reference_ratio,
        mask_token_id=mask_token_id,
    )


# ---------------------------------------------------------------------------
# OplmEmbedding
# ---------------------------------------------------------------------------


def test_embedding_output_shape():
    emb = OplmEmbedding(_config(vocab_size=33, hidden_size=16))
    input_ids = torch.randint(0, 33, (2, 5))
    out = emb(input_ids)
    assert out.shape == (2, 5, 16)


def test_embedding_dtype_matches_table():
    emb = OplmEmbedding(_config())
    input_ids = torch.randint(0, 33, (2, 5))
    out = emb(input_ids)
    assert out.dtype == emb.embed_tokens.weight.dtype


def test_embedding_lookup_matches_table_rows():
    emb = OplmEmbedding(_config(vocab_size=33, hidden_size=8))
    input_ids = torch.tensor([[0, 2, 7]])
    out = emb(input_ids)
    expected = emb.embed_tokens.weight[input_ids]
    assert torch.allclose(out, expected)


def test_embedding_no_post_norm_by_default():
    emb = OplmEmbedding(_config(post_embed_norm=False))
    assert isinstance(emb.post_norm, torch.nn.Identity)


def test_embedding_post_norm_layernorm():
    emb = OplmEmbedding(_config(post_embed_norm=True, norm_type="layernorm"))
    assert isinstance(emb.post_norm, OplmLayerNorm)
    input_ids = torch.randint(0, 33, (2, 4))
    out = emb(input_ids)
    # LayerNorm zeros per-row mean.
    assert torch.allclose(out.mean(-1), torch.zeros(2, 4), atol=1e-5)


def test_embedding_post_norm_rmsnorm():
    emb = OplmEmbedding(_config(post_embed_norm=True, norm_type="rmsnorm"))
    assert isinstance(emb.post_norm, OplmRMSNorm)


def test_embedding_grad_flows_to_table():
    emb = OplmEmbedding(_config())
    input_ids = torch.randint(0, 33, (2, 5))
    out = emb(input_ids)
    out.sum().backward()
    assert emb.embed_tokens.weight.grad is not None
    assert emb.embed_tokens.weight.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# OplmEmbedding — mask dropout
# ---------------------------------------------------------------------------

_MASK_ID = 32


def test_mask_dropout_disabled_matches_plain_lookup():
    """With mask_dropout off, a `<mask>` row is embedded like any other token."""
    emb = OplmEmbedding(_config(mask_dropout=False))
    input_ids = torch.tensor([[5, _MASK_ID, 7, 9]])
    out = emb(input_ids)
    expected = emb.embed_tokens.weight[input_ids]
    assert torch.allclose(out, expected)
    # The <mask> row is not zeroed when the feature is disabled.
    assert out[0, 1].abs().sum() > 0


def test_mask_dropout_zeros_mask_rows():
    """Enabled mask dropout zeroes exactly the `<mask>` positions."""
    emb = OplmEmbedding(_config(mask_dropout=True))
    input_ids = torch.tensor([[5, _MASK_ID, 7, _MASK_ID]])
    out = emb(input_ids)
    assert torch.allclose(out[0, 1], torch.zeros(emb.hidden_size))
    assert torch.allclose(out[0, 3], torch.zeros(emb.hidden_size))
    assert out[0, 0].abs().sum() > 0
    assert out[0, 2].abs().sum() > 0


def test_mask_dropout_per_row_scaling_matches_formula():
    """Surviving rows are scaled by `(1 - ref) / (1 - observed)` per row."""
    ref = 0.12
    emb = OplmEmbedding(_config(mask_dropout=True, mask_dropout_reference_ratio=ref))
    # Row 0: 1/4 tokens masked; row 1: 2/4 tokens masked.
    input_ids = torch.tensor([[5, _MASK_ID, 7, 9], [5, _MASK_ID, _MASK_ID, 9]])
    out = emb(input_ids)

    scale0 = (1 - ref) / (1 - 1 / 4)
    scale1 = (1 - ref) / (1 - 2 / 4)
    table = emb.embed_tokens.weight
    assert torch.allclose(out[0, 0], table[5] * scale0, atol=1e-6)
    assert torch.allclose(out[0, 2], table[7] * scale0, atol=1e-6)
    assert torch.allclose(out[1, 0], table[5] * scale1, atol=1e-6)
    assert torch.allclose(out[1, 3], table[9] * scale1, atol=1e-6)


def test_mask_dropout_counts_only_real_tokens():
    """Pad positions (attention_mask 0) are excluded from the real-token count."""
    ref = 0.12
    emb = OplmEmbedding(_config(mask_dropout=True, mask_dropout_reference_ratio=ref))
    input_ids = torch.tensor([[5, _MASK_ID, 1, 1]])
    # Only the first two positions are real -> observed ratio 1/2, not 1/4.
    attention_mask = torch.tensor([[1, 1, 0, 0]])
    out = emb(input_ids, attention_mask)

    scale = (1 - ref) / (1 - 1 / 2)
    assert torch.allclose(out[0, 0], emb.embed_tokens.weight[5] * scale, atol=1e-6)


def test_mask_dropout_no_attention_mask_assumes_all_real():
    """Omitting the mask treats every position as a real token."""
    ref = 0.12
    emb = OplmEmbedding(_config(mask_dropout=True, mask_dropout_reference_ratio=ref))
    input_ids = torch.tensor([[5, _MASK_ID, 7, 9]])
    out_no_mask = emb(input_ids)
    out_ones = emb(input_ids, torch.ones_like(input_ids))
    assert torch.allclose(out_no_mask, out_ones)


def test_mask_dropout_all_mask_row_is_finite_and_zero():
    """An all-`<mask>` row stays finite (no inf/NaN) and resolves to zeros."""
    emb = OplmEmbedding(_config(mask_dropout=True))
    input_ids = torch.full((1, 4), _MASK_ID)
    out = emb(input_ids)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, torch.zeros_like(out))


def test_mask_dropout_all_pad_row_is_finite():
    """An all-pad row (no real tokens) produces finite output."""
    emb = OplmEmbedding(_config(mask_dropout=True))
    input_ids = torch.tensor([[1, 1, 1]])
    attention_mask = torch.zeros(1, 3, dtype=torch.long)
    out = emb(input_ids, attention_mask)
    assert torch.isfinite(out).all()


def test_mask_dropout_grad_flows_to_surviving_rows():
    """Gradients reach the table through the scaled, surviving rows."""
    emb = OplmEmbedding(_config(mask_dropout=True))
    input_ids = torch.tensor([[5, _MASK_ID, 7, 9]])
    out = emb(input_ids)
    out.sum().backward()
    grad = emb.embed_tokens.weight.grad
    assert grad is not None
    # Surviving tokens accumulate gradient; the zeroed <mask> row does not.
    assert grad[5].abs().sum() > 0
    assert torch.allclose(grad[_MASK_ID], torch.zeros(emb.hidden_size))


# ---------------------------------------------------------------------------
# mean_pool
# ---------------------------------------------------------------------------


def test_mean_pool_output_shape():
    hidden = torch.randn(3, 7, 16)
    mask = torch.ones(3, 7, dtype=torch.long)
    out = mean_pool(hidden, mask)
    assert out.shape == (3, 16)


def test_mean_pool_all_ones_matches_plain_mean():
    hidden = torch.randn(2, 5, 8)
    mask = torch.ones(2, 5, dtype=torch.long)
    assert torch.allclose(mean_pool(hidden, mask), hidden.mean(dim=1), atol=1e-6)


def test_mean_pool_ignores_pad_positions():
    hidden = torch.randn(1, 4, 3)
    # Mask out the last two positions.
    mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)
    out = mean_pool(hidden, mask)
    expected = hidden[0, :2, :].mean(dim=0, keepdim=True)
    assert torch.allclose(out, expected, atol=1e-6)


def test_mean_pool_pad_values_dont_leak():
    hidden = torch.zeros(1, 3, 4)
    hidden[0, 2, :] = 100.0  # pad position carries garbage
    mask = torch.tensor([[1, 1, 0]], dtype=torch.long)
    out = mean_pool(hidden, mask)
    assert torch.allclose(out, torch.zeros(1, 4))


def test_mean_pool_empty_mask_returns_zeros():
    hidden = torch.randn(1, 3, 4)
    mask = torch.zeros(1, 3, dtype=torch.long)
    out = mean_pool(hidden, mask)
    assert torch.allclose(out, torch.zeros(1, 4))


def test_mean_pool_preserves_dtype():
    hidden = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    mask = torch.ones(2, 4, dtype=torch.long)
    out = mean_pool(hidden, mask)
    assert out.dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# cls_pool
# ---------------------------------------------------------------------------


def test_cls_pool_returns_position_zero():
    hidden = torch.randn(3, 7, 16)
    out = cls_pool(hidden)
    assert out.shape == (3, 16)
    assert torch.equal(out, hidden[:, 0, :])


def test_cls_pool_preserves_dtype():
    hidden = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    out = cls_pool(hidden)
    assert out.dtype == torch.bfloat16
