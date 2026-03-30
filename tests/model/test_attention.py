"""Tests for the Attention module."""

from __future__ import annotations

import torch
import torch.testing

from oplm.config import ModelConfig
from oplm.model.attention import Attention


def _make_config(**kwargs: object) -> ModelConfig:
    defaults: dict[str, object] = {
        "hidden_dim": 64,
        "num_heads": 4,
        "num_kv_heads": 2,
        "num_layers": 4,
        "max_seq_len": 32,
    }
    defaults.update(kwargs)
    return ModelConfig(**defaults)


# ---------------------------------------------------------------------------
# Basic forward tests
# ---------------------------------------------------------------------------


class TestAttentionForward:
    """Basic forward-pass tests for the Attention module."""

    def test_output_shape(self) -> None:
        cfg = _make_config()
        attn = Attention(cfg, layer_idx=0)
        x = torch.randn(2, 8, cfg.hidden_dim)
        output, v_first, weights = attn(x)
        assert output.shape == (2, 8, cfg.hidden_dim)

    def test_default_weights_none(self) -> None:
        cfg = _make_config()
        attn = Attention(cfg, layer_idx=0)
        x = torch.randn(2, 8, cfg.hidden_dim)
        _, _, weights = attn(x)
        assert weights is None

    def test_gqa_output_shape(self) -> None:
        cfg = _make_config(num_heads=8, num_kv_heads=2)
        attn = Attention(cfg, layer_idx=0)
        x = torch.randn(2, 8, cfg.hidden_dim)
        output, _, _ = attn(x)
        assert output.shape == (2, 8, cfg.hidden_dim)

    def test_shared_kv_output_shape(self) -> None:
        cfg = _make_config(shared_kv=True)
        attn = Attention(cfg, layer_idx=0)
        x = torch.randn(2, 8, cfg.hidden_dim)
        output, _, _ = attn(x)
        assert output.shape == (2, 8, cfg.hidden_dim)

    def test_value_residual_layer0(self) -> None:
        cfg = _make_config(value_residual=True)
        attn = Attention(cfg, layer_idx=0)
        x = torch.randn(2, 8, cfg.hidden_dim)
        _, v_first, _ = attn(x)
        assert v_first is not None

    def test_value_residual_layer1(self) -> None:
        cfg = _make_config(value_residual=True)
        attn0 = Attention(cfg, layer_idx=0)
        attn1 = Attention(cfg, layer_idx=1)
        x = torch.randn(2, 8, cfg.hidden_dim)
        _, v_first, _ = attn0(x)
        output, v_first_out, _ = attn1(x, v_first=v_first)
        assert output.shape == (2, 8, cfg.hidden_dim)
        assert v_first_out is None  # only layer 0 returns v_first


# ---------------------------------------------------------------------------
# need_weights tests
# ---------------------------------------------------------------------------


class TestAttentionNeedWeights:
    """Tests for the need_weights feature."""

    def test_returns_weights(self) -> None:
        cfg = _make_config()
        attn = Attention(cfg, layer_idx=0)
        x = torch.randn(2, 8, cfg.hidden_dim)
        _, _, weights = attn(x, need_weights=True)
        assert weights is not None
        assert weights.shape == (2, cfg.num_heads, 8, 8)  # (B, H, T, T)

    def test_weights_sum_to_one(self) -> None:
        cfg = _make_config()
        attn = Attention(cfg, layer_idx=0)
        x = torch.randn(2, 8, cfg.hidden_dim)
        _, _, weights = attn(x, need_weights=True)
        row_sums = weights.sum(dim=-1)  # (B, H, T)
        torch.testing.assert_close(row_sums, torch.ones_like(row_sums), atol=1e-5, rtol=1e-5)

    def test_weights_nonnegative(self) -> None:
        cfg = _make_config()
        attn = Attention(cfg, layer_idx=0)
        x = torch.randn(2, 8, cfg.hidden_dim)
        _, _, weights = attn(x, need_weights=True)
        assert (weights >= 0).all()

    def test_output_matches_default_path(self) -> None:
        """Manual and SDPA paths should produce approximately equal outputs."""
        cfg = _make_config()
        attn = Attention(cfg, layer_idx=0)
        attn.eval()
        x = torch.randn(2, 8, cfg.hidden_dim)
        with torch.no_grad():
            out_default, _, _ = attn(x, need_weights=False)
            out_manual, _, _ = attn(x, need_weights=True)
        torch.testing.assert_close(out_default, out_manual, atol=1e-4, rtol=1e-4)

    def test_with_attention_mask(self) -> None:
        """Masked positions should receive ~0 attention weight."""
        cfg = _make_config()
        attn = Attention(cfg, layer_idx=0)
        x = torch.randn(2, 8, cfg.hidden_dim)
        # Mask out positions 4-7
        mask = torch.zeros(2, 1, 8, 8)
        mask[:, :, :, 4:] = float("-inf")
        _, _, weights = attn(x, attention_mask=mask, need_weights=True)
        assert weights[:, :, :, 4:].abs().max() < 1e-6

    def test_bool_mask_matches_default_path(self) -> None:
        """Boolean keep-masks should behave the same in SDPA and manual paths."""
        cfg = _make_config()
        attn = Attention(cfg, layer_idx=0)
        attn.eval()
        x = torch.randn(2, 8, cfg.hidden_dim)
        mask = torch.ones(2, 1, 1, 8, dtype=torch.bool)
        mask[:, :, :, 4:] = False

        with torch.no_grad():
            out_default, _, _ = attn(x, attention_mask=mask, need_weights=False)
            out_manual, _, weights = attn(x, attention_mask=mask, need_weights=True)

        torch.testing.assert_close(out_default, out_manual, atol=1e-4, rtol=1e-4)
        assert weights[:, :, :, 4:].abs().max() < 1e-6

    def test_with_shared_kv(self) -> None:
        cfg = _make_config(shared_kv=True)
        attn = Attention(cfg, layer_idx=0)
        x = torch.randn(2, 8, cfg.hidden_dim)
        output, _, weights = attn(x, need_weights=True)
        assert output.shape == (2, 8, cfg.hidden_dim)
        assert weights.shape == (2, cfg.num_heads, 8, 8)

    def test_with_output_gate(self) -> None:
        cfg = _make_config(output_gate=True)
        attn = Attention(cfg, layer_idx=0)
        x = torch.randn(2, 8, cfg.hidden_dim)
        output, _, weights = attn(x, need_weights=True)
        assert output.shape == (2, 8, cfg.hidden_dim)
        assert weights is not None

    def test_with_post_sdpa_norm(self) -> None:
        cfg = _make_config(post_sdpa_norm=True)
        attn = Attention(cfg, layer_idx=0)
        x = torch.randn(2, 8, cfg.hidden_dim)
        output, _, weights = attn(x, need_weights=True)
        assert output.shape == (2, 8, cfg.hidden_dim)
        assert weights is not None
