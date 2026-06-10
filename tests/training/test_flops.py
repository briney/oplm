"""Tests for the FLOPs-per-token estimate (Phase 2 HF-field rewrite)."""

from __future__ import annotations

from oplm.model import OplmConfig as OplmModelConfig
from oplm.training.flops import estimate_flops_per_token


def _config(**overrides: int) -> OplmModelConfig:
    """Build a small HF config, overriding only the fields a test varies."""
    base = {
        "hidden_size": 256,
        "num_attention_heads": 4,
        "num_hidden_layers": 6,
        "max_position_embeddings": 128,
    }
    base.update(overrides)
    return OplmModelConfig(**base)


def test_estimate_is_positive_finite_int() -> None:
    """The estimate is a positive, finite integer."""
    flops = estimate_flops_per_token(_config())
    assert isinstance(flops, int)
    assert flops > 0


def test_more_layers_increases_estimate() -> None:
    """Doubling ``num_hidden_layers`` strictly increases the estimate."""
    assert estimate_flops_per_token(_config(num_hidden_layers=12)) > estimate_flops_per_token(
        _config(num_hidden_layers=6)
    )


def test_wider_hidden_increases_estimate() -> None:
    """Increasing ``hidden_size`` strictly increases the estimate."""
    assert estimate_flops_per_token(_config(hidden_size=512)) > estimate_flops_per_token(
        _config(hidden_size=256)
    )


def test_relu2_counts_two_ffn_projections() -> None:
    """At a pinned ``intermediate_size``, relu2 (2 proj) saves exactly one FFN matmul vs gated."""
    gated = estimate_flops_per_token(_config(intermediate_size=1024))
    relu2 = estimate_flops_per_token(_config(intermediate_size=1024, ffn_activation="relu2"))
    # One fewer h x intermediate projection per layer, fwd+bwd (3x) included.
    per_layer_proj = 2 * 256 * 1024
    assert gated - relu2 == 3 * 6 * per_layer_proj
