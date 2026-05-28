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
) -> SimpleNamespace:
    return SimpleNamespace(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        post_embed_norm=post_embed_norm,
        norm_type=norm_type,
        norm_eps=norm_eps,
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
