"""Tests for `oplm.model.ffn` — SwiGLU, round_up_to, make_ffn."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from oplm.model.ffn import SwiGLU, make_ffn, round_up_to


def _config(
    *,
    hidden_size: int = 16,
    intermediate_size: int = 32,
    ffn_activation: str = "swiglu",
    ffn_bias: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        ffn_activation=ffn_activation,
        ffn_bias=ffn_bias,
    )


# ---------------------------------------------------------------------------
# SwiGLU
# ---------------------------------------------------------------------------


def test_swiglu_output_shape_matches_input():
    ffn = SwiGLU(hidden_size=16, intermediate_size=32)
    x = torch.randn(2, 5, 16)
    out = ffn(x)
    assert out.shape == x.shape


def test_swiglu_no_bias_by_default():
    ffn = SwiGLU(hidden_size=8, intermediate_size=16)
    assert ffn.gate_proj.bias is None
    assert ffn.up_proj.bias is None
    assert ffn.down_proj.bias is None


def test_swiglu_with_bias():
    ffn = SwiGLU(hidden_size=8, intermediate_size=16, bias=True)
    assert ffn.gate_proj.bias is not None
    assert ffn.up_proj.bias is not None
    assert ffn.down_proj.bias is not None


def test_swiglu_linear_shapes():
    ffn = SwiGLU(hidden_size=12, intermediate_size=24)
    assert ffn.gate_proj.weight.shape == (24, 12)
    assert ffn.up_proj.weight.shape == (24, 12)
    assert ffn.down_proj.weight.shape == (12, 24)


def test_swiglu_matches_reference_formula():
    """Forward must equal `down(silu(gate(x)) * up(x))` exactly."""
    ffn = SwiGLU(hidden_size=8, intermediate_size=16)
    x = torch.randn(3, 4, 8)
    expected = ffn.down_proj(F.silu(ffn.gate_proj(x)) * ffn.up_proj(x))
    assert torch.allclose(ffn(x), expected, atol=1e-6)


def test_swiglu_grad_flows_through_all_three_linears():
    ffn = SwiGLU(hidden_size=8, intermediate_size=16)
    x = torch.randn(2, 3, 8, requires_grad=True)
    ffn(x).sum().backward()
    for proj in (ffn.gate_proj, ffn.up_proj, ffn.down_proj):
        assert proj.weight.grad is not None
        assert proj.weight.grad.abs().sum() > 0


def test_swiglu_grad_flows_to_input():
    ffn = SwiGLU(hidden_size=8, intermediate_size=16)
    x = torch.randn(2, 3, 8, requires_grad=True)
    ffn(x).sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# round_up_to
# ---------------------------------------------------------------------------


def test_round_up_to_exact_multiple_is_unchanged():
    assert round_up_to(256, 256) == 256
    assert round_up_to(512, 256) == 512


def test_round_up_to_rounds_up():
    assert round_up_to(257, 256) == 512
    assert round_up_to(1, 256) == 256


def test_round_up_to_with_typical_8_over_3_swiglu_sizing():
    # 8/3 * 768 = 2048 — already aligned to 256.
    assert round_up_to(int(8 * 768 / 3), 256) == 2048
    # 8/3 * 1024 = 2730.67 -> int -> 2730 -> rounds up to 2816.
    assert round_up_to(int(8 * 1024 / 3), 256) == 2816


def test_round_up_to_zero():
    assert round_up_to(0, 256) == 0


# ---------------------------------------------------------------------------
# make_ffn factory
# ---------------------------------------------------------------------------


def test_make_ffn_swiglu():
    ffn = make_ffn(_config(ffn_activation="swiglu"))
    assert isinstance(ffn, SwiGLU)


def test_make_ffn_forwards_hidden_and_intermediate_size():
    ffn = make_ffn(_config(hidden_size=12, intermediate_size=24))
    assert isinstance(ffn, SwiGLU)
    assert ffn.hidden_size == 12
    assert ffn.intermediate_size == 24
    assert ffn.gate_proj.weight.shape == (24, 12)


def test_make_ffn_forwards_bias_setting():
    ffn = make_ffn(_config(ffn_bias=True))
    assert ffn.gate_proj.bias is not None
    ffn = make_ffn(_config(ffn_bias=False))
    assert ffn.gate_proj.bias is None


def test_make_ffn_geglu_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="geglu"):
        make_ffn(_config(ffn_activation="geglu"))


def test_make_ffn_unknown_activation_raises_value_error():
    with pytest.raises(ValueError, match="Unknown ffn_activation"):
        make_ffn(_config(ffn_activation="relu"))
