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


def test_partial_loop_counts_repeated_blocks_and_one_head() -> None:
    cfg = _config(num_hidden_layers=6, intermediate_size=1024)
    per_block_forward = 2 * 256 * (4 * 256) + 3 * 2 * 256 * 1024
    head_forward = 2 * 256 * 256 + 2 * 256 * cfg.vocab_size
    baseline = estimate_flops_per_token(cfg)
    assert baseline == 3 * (6 * per_block_forward + head_forward)
    cfg.num_loops = 3
    cfg.loop_start = 1
    cfg.loop_end = 4
    for strategy in ("stack", "interleave"):
        cfg.loop_strategy = strategy
        assert estimate_flops_per_token(cfg) - baseline == 3 * 6 * per_block_forward
        assert estimate_flops_per_token(cfg) == 3 * (12 * per_block_forward + head_forward)
    cfg.num_loops = 1
    assert estimate_flops_per_token(cfg) == baseline
