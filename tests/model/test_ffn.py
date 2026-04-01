"""Tests for the FFN module."""

from __future__ import annotations

import pytest
import torch

from oplm.config import ModelConfig
from oplm.model.ffn import FFN


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


B, T = 2, 8


# ---------------------------------------------------------------------------
# Output shape tests
# ---------------------------------------------------------------------------


class TestFFNOutputShape:
    """Verify output shapes for all activation variants."""

    def test_swiglu_shape(self) -> None:
        cfg = _make_config(ffn_activation="swiglu")
        ffn = FFN(cfg)
        x = torch.randn(B, T, cfg.hidden_dim)
        assert ffn(x).shape == (B, T, cfg.hidden_dim)

    def test_relu_squared_shape(self) -> None:
        cfg = _make_config(ffn_activation="relu_squared")
        ffn = FFN(cfg)
        x = torch.randn(B, T, cfg.hidden_dim)
        assert ffn(x).shape == (B, T, cfg.hidden_dim)

    def test_gelu_shape(self) -> None:
        cfg = _make_config(ffn_activation="gelu")
        ffn = FFN(cfg)
        x = torch.randn(B, T, cfg.hidden_dim)
        assert ffn(x).shape == (B, T, cfg.hidden_dim)


# ---------------------------------------------------------------------------
# Activation variant tests
# ---------------------------------------------------------------------------


class TestFFNActivations:
    """Verify activation-specific behavior."""

    def test_swiglu_has_gate_proj(self) -> None:
        cfg = _make_config(ffn_activation="swiglu")
        ffn = FFN(cfg)
        assert hasattr(ffn, "gate_proj")

    def test_relu_squared_no_gate_proj(self) -> None:
        cfg = _make_config(ffn_activation="relu_squared")
        ffn = FFN(cfg)
        assert not hasattr(ffn, "gate_proj")

    def test_gelu_no_gate_proj(self) -> None:
        cfg = _make_config(ffn_activation="gelu")
        ffn = FFN(cfg)
        assert not hasattr(ffn, "gate_proj")

    def test_swiglu_ffn_dim_is_8_3_rounded(self) -> None:
        """SwiGLU should use ceil(8/3 * D / 256) * 256 for ffn_dim."""
        cfg = _make_config(ffn_activation="swiglu")
        # 8/3 * 64 = 170.67 -> ceil(170.67 / 256) * 256 = 256
        assert cfg.ffn_dim == 256

    def test_relu_squared_ffn_dim_is_4x(self) -> None:
        cfg = _make_config(ffn_activation="relu_squared")
        assert cfg.ffn_dim == 4 * cfg.hidden_dim

    def test_gelu_ffn_dim_is_4x(self) -> None:
        cfg = _make_config(ffn_activation="gelu")
        assert cfg.ffn_dim == 4 * cfg.hidden_dim

    def test_different_activations_produce_different_outputs(self) -> None:
        """Different activations should generally produce different outputs."""
        torch.manual_seed(42)
        x = torch.randn(B, T, 64)
        outputs = {}
        for act in ("swiglu", "relu_squared", "gelu"):
            torch.manual_seed(0)
            cfg = _make_config(ffn_activation=act)
            ffn = FFN(cfg)
            outputs[act] = ffn(x)
        # At least one pair should differ
        assert not torch.allclose(outputs["swiglu"], outputs["gelu"], atol=1e-5)


# ---------------------------------------------------------------------------
# Canon-D convolution tests
# ---------------------------------------------------------------------------


class TestFFNConvD:
    """Test optional Canon-D depthwise convolution in expanded space."""

    def test_no_conv_d_by_default(self) -> None:
        cfg = _make_config()
        ffn = FFN(cfg)
        assert ffn.conv_d is None

    def test_conv_d_present_when_enabled(self) -> None:
        cfg = _make_config(conv_positions="D")
        ffn = FFN(cfg)
        assert ffn.conv_d is not None
        assert ffn.conv_d.conv.kernel_size == (cfg.conv_kernel_size,)

    def test_conv_d_output_shape(self) -> None:
        cfg = _make_config(conv_positions="D")
        ffn = FFN(cfg)
        x = torch.randn(B, T, cfg.hidden_dim)
        assert ffn(x).shape == (B, T, cfg.hidden_dim)

    def test_conv_d_uses_resolved_kernel_size(self) -> None:
        cfg = _make_config(conv_positions="D", conv_kernel_size=3)
        ffn = FFN(cfg, conv_kernel_size=9)
        assert ffn.conv_d is not None
        assert ffn.conv_d.conv.kernel_size == (9,)

    def test_conv_d_with_all_activations(self) -> None:
        """Canon-D should work with every activation variant."""
        x = torch.randn(B, T, 64)
        for act in ("swiglu", "relu_squared", "gelu"):
            cfg = _make_config(ffn_activation=act, conv_positions="D")
            ffn = FFN(cfg)
            out = ffn(x)
            assert out.shape == (B, T, 64), f"Failed for {act}"


# ---------------------------------------------------------------------------
# Parameter count / bias tests
# ---------------------------------------------------------------------------


class TestFFNParameters:
    """Verify parameter properties."""

    def test_no_bias(self) -> None:
        """All linear layers should be bias-free."""
        cfg = _make_config()
        ffn = FFN(cfg)
        for name, param in ffn.named_parameters():
            assert "bias" not in name or param is None, f"Unexpected bias in {name}"

    def test_swiglu_has_more_params_than_gelu(self) -> None:
        """SwiGLU has an extra gate_proj, so more params (at same ffn_dim)."""
        # Force same ffn_dim to isolate gate_proj effect
        cfg_swiglu = _make_config(ffn_activation="swiglu", ffn_dim=256)
        cfg_gelu = _make_config(ffn_activation="gelu", ffn_dim=256)
        n_swiglu = sum(p.numel() for p in FFN(cfg_swiglu).parameters())
        n_gelu = sum(p.numel() for p in FFN(cfg_gelu).parameters())
        assert n_swiglu > n_gelu


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


class TestFFNGradient:
    """Verify gradients flow through all parameters."""

    @pytest.mark.parametrize("activation", ["swiglu", "relu_squared", "gelu"])
    def test_gradient_flow(self, activation: str) -> None:
        cfg = _make_config(ffn_activation=activation)
        ffn = FFN(cfg)
        x = torch.randn(B, T, cfg.hidden_dim, requires_grad=True)
        loss = ffn(x).sum()
        loss.backward()
        for name, param in ffn.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"
