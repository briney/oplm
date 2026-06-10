"""Tests for `oplm.model.attention` — OplmAttention (SDPA + manual-softmax paths)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from oplm.model.attention import OplmAttention
from oplm.model.norm import OplmLayerNorm, OplmRMSNorm


def _config(
    *,
    hidden_size: int = 32,
    num_attention_heads: int = 4,
    head_dim: int | None = None,
    max_position_embeddings: int = 64,
    rope_theta: float = 10000.0,
    rope_dim: int | None = None,
    norm_type: str = "layernorm",
    norm_eps: float = 1e-6,
    qk_norm: bool = True,
    qk_norm_mode: str = "channel",
    qk_norm_l2_scale_init: float | None = None,
    attention_dropout: float = 0.0,
    hidden_dropout: float = 0.0,
    norm_strategy: str = "pre",
    attn_output_gate: str = "none",
    value_residual: str = "none",
    value_residual_lambda_init: float = 0.5,
) -> SimpleNamespace:
    head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
    rope_dim = rope_dim if rope_dim is not None else head_dim
    return SimpleNamespace(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        head_dim=head_dim,
        max_position_embeddings=max_position_embeddings,
        rope_theta=rope_theta,
        rope_dim=rope_dim,
        norm_type=norm_type,
        norm_eps=norm_eps,
        qk_norm=qk_norm,
        qk_norm_mode=qk_norm_mode,
        qk_norm_l2_scale_init=qk_norm_l2_scale_init,
        attention_dropout=attention_dropout,
        hidden_dropout=hidden_dropout,
        norm_strategy=norm_strategy,
        attn_output_gate=attn_output_gate,
        value_residual=value_residual,
        value_residual_lambda_init=value_residual_lambda_init,
    )


def _ones_mask(batch: int, seq: int) -> torch.Tensor:
    return torch.ones(batch, seq, dtype=torch.long)


# ---------------------------------------------------------------------------
# Construction / parameter wiring
# ---------------------------------------------------------------------------


def test_constructor_rejects_head_dim_mismatch():
    cfg = _config(hidden_size=32, num_attention_heads=4, head_dim=9)
    with pytest.raises(ValueError, match="head_dim"):
        OplmAttention(cfg)


def test_o_proj_is_marked_residual_writer():
    attn = OplmAttention(_config())
    assert getattr(attn.o_proj, "_is_residual_writer", False) is True


def test_q_proj_k_proj_v_proj_have_no_bias():
    attn = OplmAttention(_config())
    assert attn.q_proj.bias is None
    assert attn.k_proj.bias is None
    assert attn.v_proj.bias is None
    assert attn.o_proj.bias is None


def test_qk_norm_modules_built_when_enabled():
    attn = OplmAttention(_config(qk_norm=True, norm_type="layernorm"))
    assert isinstance(attn.q_norm, OplmLayerNorm)
    assert isinstance(attn.k_norm, OplmLayerNorm)


def test_qk_norm_disabled_uses_identity():
    attn = OplmAttention(_config(qk_norm=False))
    assert isinstance(attn.q_norm, nn.Identity)
    assert isinstance(attn.k_norm, nn.Identity)


def test_qk_norm_respects_norm_type():
    attn = OplmAttention(_config(qk_norm=True, norm_type="rmsnorm"))
    assert isinstance(attn.q_norm, OplmRMSNorm)
    assert isinstance(attn.k_norm, OplmRMSNorm)


# ---------------------------------------------------------------------------
# V-norm wiring per norm_strategy
# ---------------------------------------------------------------------------


def test_v_norm_is_identity_under_pre_strategy():
    attn = OplmAttention(_config(norm_strategy="pre"))
    assert isinstance(attn.v_norm, nn.Identity)


@pytest.mark.parametrize("strategy", ["pre", "sandwich", "post_sdpa"])
def test_v_norm_is_identity_under_non_hybrid_strategies(strategy: str):
    attn = OplmAttention(_config(norm_strategy=strategy))
    assert isinstance(attn.v_norm, nn.Identity)


def test_v_norm_is_real_norm_under_hybrid_and_parameters_appear():
    attn = OplmAttention(_config(norm_strategy="hybrid", norm_type="layernorm"))
    assert isinstance(attn.v_norm, OplmLayerNorm)
    # The v_norm weight (and bias) should appear in the module's parameter list.
    param_ids = {id(p) for p in attn.parameters()}
    assert id(attn.v_norm.weight) in param_ids
    assert id(attn.v_norm.bias) in param_ids


def test_v_norm_is_real_norm_under_hybrid_with_rmsnorm():
    attn = OplmAttention(_config(norm_strategy="hybrid", norm_type="rmsnorm"))
    assert isinstance(attn.v_norm, OplmRMSNorm)


# ---------------------------------------------------------------------------
# Forward shape + dtypes
# ---------------------------------------------------------------------------


def test_forward_output_shape_matches_input():
    torch.manual_seed(0)
    attn = OplmAttention(_config(hidden_size=32, num_attention_heads=4))
    x = torch.randn(2, 7, 32)
    out, attn_weights = attn(x, _ones_mask(2, 7))
    assert out.shape == (2, 7, 32)
    # output_attentions defaults to False -> the SDPA path runs and returns no
    # weights (the manual softmax path is used only when they are requested).
    assert attn_weights is None


def test_output_attentions_returns_fp32_weights_summing_to_one():
    torch.manual_seed(1)
    attn = OplmAttention(_config(hidden_size=32, num_attention_heads=4))
    x = torch.randn(2, 5, 32)
    mask = _ones_mask(2, 5)
    _, w = attn(x, mask, output_attentions=True)
    assert w is not None
    assert w.shape == (2, 4, 5, 5)
    assert w.dtype == torch.float32
    # Row sums == 1 on real positions (every position is real here).
    sums = w.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


# ---------------------------------------------------------------------------
# Pad masking
# ---------------------------------------------------------------------------


def test_padded_inputs_match_unpadded_at_real_positions():
    """Doubling the seq with `<pad>` tokens must not change the output at real positions."""
    torch.manual_seed(2)
    attn = OplmAttention(_config(hidden_size=16, num_attention_heads=4))
    x_real = torch.randn(1, 4, 16)
    mask_real = torch.ones(1, 4, dtype=torch.long)

    # Pad with garbage to length 8; mask marks the last 4 positions as pads.
    x_pad = torch.cat([x_real, torch.randn(1, 4, 16) * 1000.0], dim=1)
    mask_pad = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.long)

    out_real, _ = attn(x_real, mask_real)
    out_pad, _ = attn(x_pad, mask_pad)
    assert torch.allclose(out_real, out_pad[:, :4, :], atol=1e-5)


def test_softmax_row_masks_pads_to_zero():
    torch.manual_seed(3)
    attn = OplmAttention(_config(hidden_size=16, num_attention_heads=4))
    x = torch.randn(1, 6, 16)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.long)
    _, w = attn(x, mask, output_attentions=True)
    assert w is not None
    # Attention to pad positions (last 2 cols) is zero for every real query row.
    assert torch.allclose(w[..., :4, 4:], torch.zeros_like(w[..., :4, 4:]), atol=1e-6)
    # Real-position queries sum to 1 across the real-key columns.
    real_sums = w[..., :4, :4].sum(dim=-1)
    assert torch.allclose(real_sums, torch.ones_like(real_sums), atol=1e-5)


# ---------------------------------------------------------------------------
# Path selection: SDPA by default, manual softmax for output_attentions
# ---------------------------------------------------------------------------


def test_default_path_returns_no_attention_weights():
    """output_attentions=False takes the SDPA path, which yields no weights."""
    attn = OplmAttention(_config()).eval()
    out, w = attn(torch.randn(2, 6, 32), _ones_mask(2, 6), output_attentions=False)
    assert out.shape == (2, 6, 32)
    assert w is None


def test_output_attentions_path_returns_weights():
    """output_attentions=True takes the manual softmax path, which yields weights."""
    attn = OplmAttention(_config()).eval()
    _, w = attn(torch.randn(2, 6, 32), _ones_mask(2, 6), output_attentions=True)
    assert w is not None
    assert w.shape == (2, 4, 6, 6)


# ---------------------------------------------------------------------------
# RoPE / NoPE wiring
# ---------------------------------------------------------------------------


def test_partial_rope_runs():
    """rope_dim < head_dim still produces correct shapes."""
    cfg = _config(hidden_size=32, num_attention_heads=4, head_dim=8, rope_dim=4)
    attn = OplmAttention(cfg)
    x = torch.randn(2, 6, 32)
    out, _ = attn(x, _ones_mask(2, 6))
    assert out.shape == (2, 6, 32)


def test_pure_nope_runs():
    """rope_dim == 0 (pure NoPE) takes the fast no-op rotary path."""
    cfg = _config(hidden_size=32, num_attention_heads=4, head_dim=8, rope_dim=0)
    attn = OplmAttention(cfg)
    x = torch.randn(2, 6, 32)
    out, _ = attn(x, _ones_mask(2, 6))
    assert out.shape == (2, 6, 32)


# ---------------------------------------------------------------------------
# Hybrid strategy forward
# ---------------------------------------------------------------------------


def test_hybrid_strategy_forward_runs_and_v_norm_is_active():
    """Under hybrid, v_norm is a real norm and the forward still produces (B,T,D)."""
    cfg = _config(norm_strategy="hybrid")
    attn = OplmAttention(cfg)
    x = torch.randn(2, 5, 32)
    out, _ = attn(x, _ones_mask(2, 5))
    assert out.shape == (2, 5, 32)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_grad_flows_through_all_projections():
    attn = OplmAttention(_config())
    x = torch.randn(2, 5, 32, requires_grad=True)
    out, _ = attn(x, _ones_mask(2, 5))
    out.sum().backward()
    for proj in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj):
        assert proj.weight.grad is not None
        assert proj.weight.grad.abs().sum() > 0
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# SDPA / manual-softmax equivalence
# ---------------------------------------------------------------------------


def test_sdpa_and_manual_paths_agree():
    """The SDPA and manual-softmax paths produce the same `(B, T, D)` output.

    Both share the scale (`1/sqrt(d_head)`) and key-padding semantics, so in
    fp32 they agree to tight tolerance. The SDPA path returns no weights; we
    compare the projected outputs only. Runs on CPU via SDPA's math backend, so
    it is not gated on CUDA.
    """
    torch.manual_seed(0)
    attn = OplmAttention(_config(hidden_size=32, num_attention_heads=4)).eval()
    x = torch.randn(2, 8, 32)
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.long)

    with torch.no_grad():
        out_sdpa, w_sdpa = attn(x, mask, output_attentions=False)
        out_manual, w_manual = attn(x, mask, output_attentions=True)

    assert w_sdpa is None
    assert w_manual is not None
    assert torch.allclose(out_sdpa, out_manual, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# L2 QK-norm mode
# ---------------------------------------------------------------------------


def test_channel_mode_uses_canonical_kernel_scale():
    """Default (channel) QK norm keeps the 1/sqrt(head_dim) attention scale."""
    attn = OplmAttention(_config(hidden_size=32, num_attention_heads=4))  # head_dim 8
    assert attn.qk_l2 is False
    assert attn.attn_scale == pytest.approx(1.0 / (8**0.5))
    assert not hasattr(attn, "qk_l2_scale")


def test_disabled_qk_norm_uses_canonical_kernel_scale():
    """qk_norm=False (even with mode='l2') keeps Identity norms and 1/sqrt(d) scale."""
    attn = OplmAttention(_config(qk_norm=False, qk_norm_mode="l2", hidden_size=32))
    assert attn.qk_l2 is False
    assert attn.attn_scale == pytest.approx(1.0 / (8**0.5))
    assert isinstance(attn.q_norm, nn.Identity)
    assert isinstance(attn.k_norm, nn.Identity)
    assert not hasattr(attn, "qk_l2_scale")


def test_l2_mode_builds_per_head_scale_and_identity_norms():
    """L2 mode replaces the channel norms with a learned per-head scale param."""
    attn = OplmAttention(_config(hidden_size=32, num_attention_heads=4, qk_norm_mode="l2"))
    assert attn.qk_l2 is True
    assert attn.attn_scale == 1.0
    assert isinstance(attn.q_norm, nn.Identity)
    assert isinstance(attn.k_norm, nn.Identity)
    assert isinstance(attn.qk_l2_scale, nn.Parameter)
    assert attn.qk_l2_scale.shape == (4,)


def test_l2_scale_defaults_to_sqrt_head_dim():
    """With qk_norm_l2_scale_init unset, qk_l2_scale initializes to sqrt(head_dim)."""
    attn = OplmAttention(_config(hidden_size=32, num_attention_heads=4, qk_norm_mode="l2"))
    # head_dim = 8
    assert torch.allclose(attn.qk_l2_scale, torch.full((4,), 8.0**0.5))


def test_l2_scale_respects_explicit_init():
    """An explicit qk_norm_l2_scale_init sets every per-head entry."""
    attn = OplmAttention(
        _config(num_attention_heads=4, qk_norm_mode="l2", qk_norm_l2_scale_init=3.5)
    )
    assert torch.allclose(attn.qk_l2_scale, torch.full((4,), 3.5))


def test_l2_norm_unit_normalizes_k_and_scales_q():
    """_qk_l2_norm yields unit-norm K rows and Q rows scaled by the per-head value."""
    torch.manual_seed(0)
    attn = OplmAttention(
        _config(hidden_size=32, num_attention_heads=4, qk_norm_mode="l2", qk_norm_l2_scale_init=2.0)
    )
    q = torch.randn(2, 4, 6, 8)  # (B, H, T, d_head)
    k = torch.randn(2, 4, 6, 8)
    q_hat, k_hat = attn._qk_l2_norm(q, k)
    k_norms = k_hat.float().norm(dim=-1)
    q_norms = q_hat.float().norm(dim=-1)
    assert torch.allclose(k_norms, torch.ones_like(k_norms), atol=1e-5)
    assert torch.allclose(q_norms, torch.full_like(q_norms, 2.0), atol=1e-5)


def test_l2_mode_forward_shape_and_finite():
    torch.manual_seed(0)
    attn = OplmAttention(_config(hidden_size=32, num_attention_heads=4, qk_norm_mode="l2"))
    x = torch.randn(2, 7, 32)
    out, _ = attn(x, _ones_mask(2, 7))
    assert out.shape == (2, 7, 32)
    assert torch.isfinite(out).all()


def test_l2_mode_sdpa_and_manual_paths_agree():
    """Under L2 mode both kernels use scale 1.0 and agree in fp32."""
    torch.manual_seed(0)
    attn = OplmAttention(_config(hidden_size=32, num_attention_heads=4, qk_norm_mode="l2")).eval()
    x = torch.randn(2, 8, 32)
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.long)
    with torch.no_grad():
        out_sdpa, _ = attn(x, mask, output_attentions=False)
        out_manual, _ = attn(x, mask, output_attentions=True)
    assert torch.allclose(out_sdpa, out_manual, rtol=1e-4, atol=1e-5)


def test_l2_scale_receives_gradient():
    """Gradients flow into qk_l2_scale through the attention logits."""
    torch.manual_seed(0)
    attn = OplmAttention(_config(hidden_size=32, num_attention_heads=4, qk_norm_mode="l2"))
    x = torch.randn(2, 5, 32)
    out, _ = attn(x, _ones_mask(2, 5))
    out.sum().backward()
    assert attn.qk_l2_scale.grad is not None
    assert attn.qk_l2_scale.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Attention output gate (gated attention, arXiv:2505.06708 G1)
# ---------------------------------------------------------------------------


def test_output_gate_disabled_by_default_adds_no_params():
    attn = OplmAttention(_config())
    assert attn.attn_output_gate == "none"
    assert not hasattr(attn, "gate_proj")


@pytest.mark.parametrize("activation", ["sigmoid", "silu"])
def test_output_gate_builds_elementwise_projection(activation: str):
    """Enabling the gate adds a bias-free D x D projection that is not a residual writer."""
    attn = OplmAttention(_config(hidden_size=32, attn_output_gate=activation))
    assert attn.gate_proj.weight.shape == (32, 32)
    assert attn.gate_proj.bias is None
    assert getattr(attn.gate_proj, "_is_residual_writer", False) is False


def test_output_gate_constructor_rejects_unknown_activation():
    with pytest.raises(ValueError, match="attn_output_gate"):
        OplmAttention(_config(attn_output_gate="tanh"))


@pytest.mark.parametrize("activation", ["sigmoid", "silu"])
def test_output_gate_forward_shape_and_finite(activation: str):
    torch.manual_seed(0)
    attn = OplmAttention(
        _config(hidden_size=32, num_attention_heads=4, attn_output_gate=activation)
    )
    x = torch.randn(2, 7, 32)
    out, _ = attn(x, _ones_mask(2, 7))
    assert out.shape == (2, 7, 32)
    assert torch.isfinite(out).all()


def test_output_gate_changes_output():
    """With identical Q/K/V/O weights, enabling the gate changes the output."""
    torch.manual_seed(0)
    base = OplmAttention(_config(hidden_size=32, num_attention_heads=4)).eval()
    # Same seed: q/k/v/o draw the same init values; gate_proj draws afterwards.
    torch.manual_seed(0)
    gated = OplmAttention(
        _config(hidden_size=32, num_attention_heads=4, attn_output_gate="sigmoid")
    ).eval()
    assert torch.equal(base.o_proj.weight, gated.o_proj.weight)

    torch.manual_seed(1)
    x = torch.randn(2, 6, 32)
    with torch.no_grad():
        out_base, _ = base(x, _ones_mask(2, 6))
        out_gated, _ = gated(x, _ones_mask(2, 6))
    assert not torch.allclose(out_base, out_gated)


def test_output_gate_sigmoid_and_silu_differ():
    """Identical weights, different gate activation: outputs must differ."""
    torch.manual_seed(0)
    sig = OplmAttention(
        _config(hidden_size=32, num_attention_heads=4, attn_output_gate="sigmoid")
    ).eval()
    torch.manual_seed(0)
    silu = OplmAttention(
        _config(hidden_size=32, num_attention_heads=4, attn_output_gate="silu")
    ).eval()
    assert torch.equal(sig.gate_proj.weight, silu.gate_proj.weight)

    torch.manual_seed(1)
    x = torch.randn(2, 6, 32)
    with torch.no_grad():
        out_sig, _ = sig(x, _ones_mask(2, 6))
        out_silu, _ = silu(x, _ones_mask(2, 6))
    assert not torch.allclose(out_sig, out_silu)


def test_output_gate_sdpa_and_manual_paths_agree():
    """The gate sits after the kernel, so both compute paths still agree."""
    torch.manual_seed(0)
    attn = OplmAttention(
        _config(hidden_size=32, num_attention_heads=4, attn_output_gate="sigmoid")
    ).eval()
    x = torch.randn(2, 8, 32)
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.long)
    with torch.no_grad():
        out_sdpa, _ = attn(x, mask, output_attentions=False)
        out_manual, _ = attn(x, mask, output_attentions=True)
    assert torch.allclose(out_sdpa, out_manual, rtol=1e-4, atol=1e-5)


def test_output_gate_receives_gradient():
    """Gradients flow into gate_proj alongside the standard projections."""
    torch.manual_seed(0)
    attn = OplmAttention(_config(hidden_size=32, num_attention_heads=4, attn_output_gate="silu"))
    x = torch.randn(2, 5, 32, requires_grad=True)
    out, _ = attn(x, _ones_mask(2, 5))
    out.sum().backward()
    for proj in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj, attn.gate_proj):
        assert proj.weight.grad is not None
        assert proj.weight.grad.abs().sum() > 0
    assert x.grad is not None


# ---------------------------------------------------------------------------
# Value residual (ResFormer, arXiv:2410.17897)
# ---------------------------------------------------------------------------


def _random_v1(batch: int, seq: int, heads: int = 4, head_dim: int = 8) -> torch.Tensor:
    return torch.randn(batch, heads, seq, head_dim)


def test_value_residual_disabled_by_default_adds_no_params():
    attn = OplmAttention(_config(), layer_idx=1)
    assert attn.value_residual == "none"
    assert not hasattr(attn, "value_residual_lambda")
    out = attn(torch.randn(2, 5, 32), _ones_mask(2, 5))
    assert len(out) == 2


def test_value_residual_learnable_creates_param_only_on_later_layers():
    layer1 = OplmAttention(_config(value_residual="learnable"), layer_idx=1)
    assert isinstance(layer1.value_residual_lambda, nn.Parameter)
    assert layer1.value_residual_lambda.shape == (1,)
    assert layer1.value_residual_lambda.item() == pytest.approx(0.5)

    layer0 = OplmAttention(_config(value_residual="learnable"), layer_idx=0)
    assert not hasattr(layer0, "value_residual_lambda")


def test_value_residual_fixed_uses_buffer_not_param():
    layer1 = OplmAttention(
        _config(value_residual="fixed", value_residual_lambda_init=0.7), layer_idx=1
    )
    assert layer1.value_residual_lambda.item() == pytest.approx(0.7)
    assert "value_residual_lambda" not in dict(layer1.named_parameters())
    assert "value_residual_lambda" in dict(layer1.named_buffers())

    layer0 = OplmAttention(_config(value_residual="fixed"), layer_idx=0)
    assert not hasattr(layer0, "value_residual_lambda")


def test_value_residual_constructor_rejects_unknown():
    with pytest.raises(ValueError, match="value_residual"):
        OplmAttention(_config(value_residual="blend"))


def test_value_residual_layer0_returns_v_others_return_none():
    torch.manual_seed(0)
    x = torch.randn(2, 6, 32)

    layer0 = OplmAttention(_config(value_residual="learnable"), layer_idx=0)
    out0, _, v1 = layer0(x, _ones_mask(2, 6))
    assert out0.shape == (2, 6, 32)
    assert v1 is not None
    assert v1.shape == (2, 4, 6, 8)  # (B, H, T, d_head)

    layer1 = OplmAttention(_config(value_residual="learnable"), layer_idx=1)
    _, _, v_later = layer1(x, _ones_mask(2, 6), value_residual=v1)
    assert v_later is None


def test_value_residual_lambda_one_is_identity():
    """Fixed lambda = 1.0: blending toward v1 is a no-op."""
    torch.manual_seed(0)
    attn = OplmAttention(
        _config(value_residual="fixed", value_residual_lambda_init=1.0), layer_idx=1
    ).eval()
    x = torch.randn(2, 6, 32)
    with torch.no_grad():
        out_blend, _, _ = attn(x, _ones_mask(2, 6), value_residual=_random_v1(2, 6))
        out_plain, _, _ = attn(x, _ones_mask(2, 6))
    assert torch.allclose(out_blend, out_plain, rtol=1e-5, atol=1e-6)


def test_value_residual_lambda_zero_replaces_v():
    """Fixed lambda = 0.0: this layer's values are fully replaced by v1."""
    torch.manual_seed(0)
    attn = OplmAttention(
        _config(value_residual="fixed", value_residual_lambda_init=0.0), layer_idx=1
    ).eval()
    x = torch.randn(2, 6, 32)
    v1 = _random_v1(2, 6)
    with torch.no_grad():
        out_a, _, _ = attn(x, _ones_mask(2, 6), value_residual=v1)
        # Zeroing v_proj must not matter: V is fully substituted by v1.
        attn.v_proj.weight.zero_()
        out_b, _, _ = attn(x, _ones_mask(2, 6), value_residual=v1)
    assert torch.allclose(out_a, out_b, rtol=1e-5, atol=1e-6)


def test_value_residual_blend_changes_output():
    torch.manual_seed(0)
    attn = OplmAttention(_config(value_residual="learnable"), layer_idx=1).eval()
    x = torch.randn(2, 6, 32)
    with torch.no_grad():
        out_blend, _, _ = attn(x, _ones_mask(2, 6), value_residual=_random_v1(2, 6))
        out_plain, _, _ = attn(x, _ones_mask(2, 6))
    assert not torch.allclose(out_blend, out_plain)


def test_value_residual_sdpa_and_manual_paths_agree():
    """The blend sits before the kernel, so both compute paths still agree."""
    torch.manual_seed(0)
    attn = OplmAttention(_config(value_residual="learnable"), layer_idx=1).eval()
    x = torch.randn(2, 8, 32)
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.long)
    v1 = _random_v1(2, 8)
    with torch.no_grad():
        out_sdpa, _, _ = attn(x, mask, output_attentions=False, value_residual=v1)
        out_manual, _, _ = attn(x, mask, output_attentions=True, value_residual=v1)
    assert torch.allclose(out_sdpa, out_manual, rtol=1e-4, atol=1e-5)


def test_value_residual_learnable_receives_gradient():
    torch.manual_seed(0)
    attn = OplmAttention(_config(value_residual="learnable"), layer_idx=1)
    x = torch.randn(2, 5, 32, requires_grad=True)
    out, _, _ = attn(x, _ones_mask(2, 5), value_residual=_random_v1(2, 5))
    out.sum().backward()
    assert attn.value_residual_lambda.grad is not None
    assert attn.value_residual_lambda.grad.abs().sum() > 0
    assert x.grad is not None
