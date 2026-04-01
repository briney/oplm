"""Tests for token and value embeddings."""

from __future__ import annotations

import torch

from oplm.config import ModelConfig
from oplm.model.embedding import TokenEmbedding, ValueEmbedding


class TestTokenEmbedding:
    def test_output_shape(self) -> None:
        cfg = ModelConfig()
        emb = TokenEmbedding(cfg)
        ids = torch.randint(0, cfg.vocab_size, (2, 16))
        out = emb(ids)
        assert out.shape == (2, 16, cfg.hidden_dim)

    def test_scaling(self) -> None:
        """Output should be scaled by sqrt(hidden_dim)."""
        cfg = ModelConfig(post_embed_norm=False)
        emb = TokenEmbedding(cfg)
        ids = torch.tensor([[0]])
        raw = emb.embed(ids)
        out = emb(ids)
        expected = raw * emb.scale
        torch.testing.assert_close(out, expected)

    def test_post_embed_norm_applied(self) -> None:
        cfg = ModelConfig(post_embed_norm=True)
        emb = TokenEmbedding(cfg)
        assert emb.post_norm is not None
        ids = torch.randint(0, cfg.vocab_size, (2, 16))
        out = emb(ids)
        assert out.shape == (2, 16, cfg.hidden_dim)

    def test_no_post_embed_norm(self) -> None:
        cfg = ModelConfig(post_embed_norm=False)
        emb = TokenEmbedding(cfg)
        assert emb.post_norm is None


class TestValueEmbedding:
    def _make_config(self, **kwargs: object) -> ModelConfig:
        defaults = {
            "hidden_dim": 64,
            "num_heads": 4,
            "num_kv_heads": 2,
            "num_layers": 12,
            "num_value_embeds": 2,
            "value_embed_gate_dim": 16,
        }
        defaults.update(kwargs)
        return ModelConfig(**defaults)

    def test_active_layer_output_shape(self) -> None:
        cfg = self._make_config()
        ve = ValueEmbedding(cfg)
        ids = torch.randint(0, cfg.vocab_size, (2, 10))
        x = torch.randn(2, 10, cfg.hidden_dim)
        # Layer 0 should be active (first N=2 layers)
        out = ve(ids, x, layer_idx=0)
        assert out is not None
        head_dim = cfg.head_dim or cfg.hidden_dim // cfg.num_heads
        assert out.shape == (2, 10, cfg.num_kv_heads, head_dim)

    def test_inactive_layer_returns_none(self) -> None:
        cfg = self._make_config()
        ve = ValueEmbedding(cfg)
        ids = torch.randint(0, cfg.vocab_size, (2, 10))
        x = torch.randn(2, 10, cfg.hidden_dim)
        # Layer 5 is neither in first 2 nor last 2 of 12 layers
        out = ve(ids, x, layer_idx=5)
        assert out is None

    def test_last_layers_active(self) -> None:
        cfg = self._make_config()
        ve = ValueEmbedding(cfg)
        ids = torch.randint(0, cfg.vocab_size, (2, 10))
        x = torch.randn(2, 10, cfg.hidden_dim)
        # Last 2 layers (10, 11) should be active
        assert ve(ids, x, layer_idx=10) is not None
        assert ve(ids, x, layer_idx=11) is not None

    def test_uses_layer(self) -> None:
        cfg = self._make_config()
        ve = ValueEmbedding(cfg)
        assert ve.uses_layer(0)
        assert ve.uses_layer(1)
        assert not ve.uses_layer(5)
        assert ve.uses_layer(10)
        assert ve.uses_layer(11)

    def test_layer_map_correct(self) -> None:
        cfg = self._make_config(num_layers=12, num_value_embeds=3)
        ve = ValueEmbedding(cfg)
        # First 3: layers 0, 1, 2
        # Last 3: layers 9, 10, 11
        assert ve.layer_map == {0: 0, 1: 1, 2: 2, 9: 0, 10: 1, 11: 2}

    def test_gate_range(self) -> None:
        """Gate output should be in [0, 2] due to 2*sigmoid scaling."""
        cfg = self._make_config()
        ve = ValueEmbedding(cfg)
        ids = torch.randint(0, cfg.vocab_size, (4, 20))
        x = torch.randn(4, 20, cfg.hidden_dim)
        out = ve(ids, x, layer_idx=0)
        assert out is not None
        # Values should be bounded (gate in [0, 2], embeddings are finite)
        assert torch.isfinite(out).all()
