"""Tests for `oplm.model.configuration_oplm` — OplmConfig validation and derivation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from oplm.model import OplmConfig
from oplm.model.ffn import round_up_to

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults and derived fields
# ---------------------------------------------------------------------------


def test_defaults_match_architecture_spec():
    cfg = OplmConfig()
    assert cfg.model_type == "oplm"
    assert cfg.vocab_size == 33
    assert cfg.hidden_size == 768
    assert cfg.num_hidden_layers == 12
    assert cfg.num_attention_heads == 12
    assert cfg.max_position_embeddings == 1024
    assert cfg.rope_theta == 10000.0
    assert cfg.norm_type == "layernorm"
    assert cfg.norm_strategy == "pre"
    assert cfg.qk_norm is True
    assert cfg.residual_scaling == "sqrt_num_layers"
    assert cfg.init_scale_output_projections is True
    assert cfg.ffn_activation == "swiglu"
    assert cfg.ffn_bias is False
    assert cfg.attention_dropout == 0.0
    assert cfg.hidden_dropout == 0.0
    assert cfg.tie_word_embeddings is False
    assert cfg.mlm_head_activation == "gelu"
    assert cfg.canon_enabled is False
    assert cfg.canon_positions == []
    assert cfg.canon_activation == "none"
    assert cfg.classifier_pool == "mean"
    assert cfg.classifier_dropout == 0.0
    assert cfg.pre_head_norm is False
    assert cfg.gradient_checkpointing is False
    assert cfg.pad_token_id == 1
    assert cfg.bos_token_id == 0
    assert cfg.eos_token_id == 2
    assert cfg.unk_token_id == 3
    assert cfg.mask_token_id == 32


def test_head_dim_derived_from_hidden_and_heads():
    cfg = OplmConfig(hidden_size=512, num_attention_heads=8)
    assert cfg.head_dim == 64


def test_intermediate_size_derived_from_swiglu_convention():
    cfg = OplmConfig(hidden_size=768)
    # 8/3 * 768 = 2048; already a multiple of 256.
    assert cfg.intermediate_size == round_up_to(int(8 * 768 / 3), 256)
    assert cfg.intermediate_size == 2048


def test_intermediate_size_rounds_up_to_256():
    cfg = OplmConfig(hidden_size=512, num_attention_heads=8)
    # 8/3 * 512 ≈ 1365 -> round up to nearest 256 -> 1536.
    assert cfg.intermediate_size == 1536


def test_rope_dim_defaults_to_head_dim_with_zero_nope():
    cfg = OplmConfig(hidden_size=512, num_attention_heads=8)
    assert cfg.rope_dim == cfg.head_dim
    assert cfg.nope_dim == 0


def test_explicit_overrides_take_precedence_over_derivations():
    cfg = OplmConfig(
        hidden_size=512,
        num_attention_heads=8,
        head_dim=64,
        intermediate_size=2048,
        rope_dim=32,
        nope_dim=32,
    )
    assert cfg.head_dim == 64
    assert cfg.intermediate_size == 2048
    assert cfg.rope_dim == 32
    assert cfg.nope_dim == 32


# ---------------------------------------------------------------------------
# Validation rules
# ---------------------------------------------------------------------------


def test_rejects_hidden_size_not_divisible_by_heads():
    with pytest.raises(ValueError, match="divisible by num_attention_heads"):
        OplmConfig(hidden_size=100, num_attention_heads=8)


def test_rejects_head_dim_mismatch():
    with pytest.raises(ValueError, match="must equal hidden_size"):
        OplmConfig(hidden_size=512, num_attention_heads=8, head_dim=128)


def test_rejects_rope_plus_nope_mismatch():
    with pytest.raises(ValueError, match="must equal"):
        OplmConfig(
            hidden_size=512,
            num_attention_heads=8,
            head_dim=64,
            rope_dim=32,
            nope_dim=16,
        )


def test_rejects_odd_rope_dim():
    with pytest.raises(ValueError, match="rope_dim must be even"):
        OplmConfig(
            hidden_size=512,
            num_attention_heads=8,
            head_dim=64,
            rope_dim=33,
            nope_dim=31,
        )


def test_rejects_negative_rope_dim():
    with pytest.raises(ValueError, match="rope_dim must be >= 0"):
        OplmConfig(
            hidden_size=512,
            num_attention_heads=8,
            head_dim=64,
            rope_dim=-2,
            nope_dim=66,
        )


@pytest.mark.parametrize(
    "field,bad_value,expected_match",
    [
        ("norm_type", "zorm", "norm_type must be one of"),
        ("norm_strategy", "preprost", "norm_strategy must be one of"),
        ("residual_scaling", "linear", "residual_scaling must be one of"),
        ("ffn_activation", "relu", "ffn_activation must be one of"),
        ("mlm_head_activation", "swiglu", "mlm_head_activation must be one of"),
        ("canon_activation", "tanh", "canon_activation must be one of"),
        ("classifier_pool", "max", "classifier_pool must be one of"),
    ],
)
def test_rejects_unknown_categorical_values(field, bad_value, expected_match):
    with pytest.raises(ValueError, match=expected_match):
        OplmConfig(**{field: bad_value})


def test_canon_enabled_requires_non_empty_positions():
    with pytest.raises(ValueError, match="canon_positions must be non-empty"):
        OplmConfig(canon_enabled=True, canon_positions=[])


def test_canon_rejects_unknown_position():
    with pytest.raises(ValueError, match="canon_positions entries must be a subset"):
        OplmConfig(canon_enabled=True, canon_positions=["A", "Z"], canon_kernel_sizes=3)


def test_canon_rejects_duplicate_positions():
    with pytest.raises(ValueError, match="must not contain duplicates"):
        OplmConfig(canon_enabled=True, canon_positions=["A", "A"], canon_kernel_sizes=3)


def test_canon_resolved_kernel_sizes_cached_back_onto_field():
    cfg = OplmConfig(
        num_hidden_layers=4,
        canon_enabled=True,
        canon_positions=["A"],
        canon_kernel_sizes=3,
    )
    assert cfg.canon_kernel_sizes == [3, 3, 3, 3]


def test_canon_linear_schedule_resolves_per_layer():
    cfg = OplmConfig(
        num_hidden_layers=5,
        canon_enabled=True,
        canon_positions=["A", "D"],
        canon_kernel_sizes={"schedule": "linear", "min": 3, "max": 11},
    )
    assert cfg.canon_kernel_sizes == [3, 5, 7, 9, 11]


def test_canon_constant_schedule_resolves_per_layer():
    cfg = OplmConfig(
        num_hidden_layers=3,
        canon_enabled=True,
        canon_positions=["C"],
        canon_kernel_sizes={"schedule": "constant", "value": 5},
    )
    assert cfg.canon_kernel_sizes == [5, 5, 5]


def test_canon_kernel_size_list_length_mismatch_raises():
    with pytest.raises(ValueError, match="canon_kernel_sizes list has length"):
        OplmConfig(
            num_hidden_layers=4,
            canon_enabled=True,
            canon_positions=["A"],
            canon_kernel_sizes=[3, 3, 3],
        )


def test_canon_disabled_does_not_resolve_kernel_sizes():
    """When canon is off, the raw value is left untouched (no resolution required)."""
    cfg = OplmConfig(
        num_hidden_layers=4,
        canon_enabled=False,
        canon_kernel_sizes=4,
    )
    assert cfg.canon_kernel_sizes == 4


def test_non_default_vocab_emits_warning():
    with pytest.warns(UserWarning, match="custom vocabularies are not yet supported"):
        OplmConfig(vocab_size=64)


# ---------------------------------------------------------------------------
# Pass-through / forward-compat with PretrainedConfig kwargs
# ---------------------------------------------------------------------------


def test_pretrained_config_kwargs_forwarded():
    cfg = OplmConfig(
        architectures=["OplmForMaskedLM"],
        _name_or_path="brineylab/oplm-base",
    )
    # Both attrs come from PretrainedConfig.__init__'s kwargs handling.
    assert cfg.architectures == ["OplmForMaskedLM"]
    assert cfg._name_or_path == "brineylab/oplm-base"


def test_tie_word_embeddings_forwarded_to_base_class():
    cfg = OplmConfig(tie_word_embeddings=True)
    assert cfg.tie_word_embeddings is True


# ---------------------------------------------------------------------------
# save_pretrained / from_pretrained round-trip
# ---------------------------------------------------------------------------


def test_save_and_from_pretrained_roundtrip_preserves_fields(tmp_path: Path):
    cfg = OplmConfig(
        hidden_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        norm_type="rmsnorm",
        norm_strategy="hybrid",
        canon_enabled=True,
        canon_positions=["A", "D"],
        canon_kernel_sizes={"schedule": "linear", "min": 3, "max": 9},
        post_embed_norm=True,
        tie_word_embeddings=True,
    )
    cfg.save_pretrained(tmp_path)

    on_disk = json.loads((tmp_path / "config.json").read_text())
    assert on_disk["model_type"] == "oplm"
    # The resolved per-layer list is serialised — not the original dict spec.
    assert on_disk["canon_kernel_sizes"] == [3, 5, 7, 9]

    restored = OplmConfig.from_pretrained(tmp_path)
    assert restored.hidden_size == 512
    assert restored.num_hidden_layers == 4
    assert restored.num_attention_heads == 8
    assert restored.head_dim == 64
    assert restored.norm_type == "rmsnorm"
    assert restored.norm_strategy == "hybrid"
    assert restored.canon_enabled is True
    assert restored.canon_positions == ["A", "D"]
    assert restored.canon_kernel_sizes == [3, 5, 7, 9]
    assert restored.post_embed_norm is True
    assert restored.tie_word_embeddings is True


def test_save_pretrained_writes_token_ids(tmp_path: Path):
    cfg = OplmConfig()
    cfg.save_pretrained(tmp_path)
    on_disk = json.loads((tmp_path / "config.json").read_text())
    assert on_disk["pad_token_id"] == 1
    assert on_disk["bos_token_id"] == 0
    assert on_disk["eos_token_id"] == 2
