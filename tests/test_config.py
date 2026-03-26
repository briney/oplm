"""Tests for the configuration system."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from oplm.config import ModelConfig, OplmConfig, TrainConfig, load_config, round_multiple


class TestRoundMultiple:
    def test_exact_multiple(self) -> None:
        assert round_multiple(256, 256) == 256

    def test_rounds_up(self) -> None:
        assert round_multiple(257, 256) == 512

    def test_fractional(self) -> None:
        # 8/3 * 768 = 2048.0, which is already a multiple of 256
        assert round_multiple(8 / 3 * 768, 256) == 2048


class TestModelConfigDefaults:
    def test_default_creation(self) -> None:
        cfg = ModelConfig()
        assert cfg.hidden_dim == 768
        assert cfg.num_layers == 12
        assert cfg.num_heads == 12

    def test_head_dim_derived(self) -> None:
        cfg = ModelConfig()
        assert cfg.head_dim == 64  # 768 // 12

    def test_ffn_dim_swiglu(self) -> None:
        cfg = ModelConfig(ffn_activation="swiglu")
        assert cfg.ffn_dim == round_multiple(8 / 3 * 768, 256)

    def test_ffn_dim_relu_squared(self) -> None:
        cfg = ModelConfig(ffn_activation="relu_squared")
        assert cfg.ffn_dim == 4 * 768

    def test_ffn_dim_gelu(self) -> None:
        cfg = ModelConfig(ffn_activation="gelu")
        assert cfg.ffn_dim == 4 * 768

    def test_rope_dim_full(self) -> None:
        cfg = ModelConfig(partial_rope=False)
        assert cfg.rope_dim == cfg.head_dim
        assert cfg.nope_dim == 0

    def test_rope_dim_partial(self) -> None:
        cfg = ModelConfig(partial_rope=True)
        assert cfg.rope_dim == 32
        assert cfg.nope_dim == cfg.head_dim - 32

    def test_explicit_head_dim_preserved(self) -> None:
        cfg = ModelConfig(head_dim=128)
        assert cfg.head_dim == 128

    def test_explicit_ffn_dim_preserved(self) -> None:
        cfg = ModelConfig(ffn_dim=4096)
        assert cfg.ffn_dim == 4096


class TestModelConfigValidation:
    def test_hidden_dim_not_divisible_by_heads(self) -> None:
        with pytest.raises(ValueError, match="hidden_dim.*must be divisible"):
            ModelConfig(hidden_dim=100, num_heads=12)

    def test_heads_not_divisible_by_kv_heads(self) -> None:
        with pytest.raises(ValueError, match="num_heads.*must be divisible"):
            ModelConfig(num_heads=12, num_kv_heads=5)

    def test_even_kernel_size(self) -> None:
        with pytest.raises(ValueError, match="conv_kernel_size.*must be odd"):
            ModelConfig(conv_kernel_size=6)

    def test_invalid_conv_positions(self) -> None:
        with pytest.raises(ValueError, match="conv_positions"):
            ModelConfig(conv_positions="X")

    def test_valid_conv_positions(self) -> None:
        cfg = ModelConfig(conv_positions="ACD")
        assert cfg.conv_positions == "ACD"

    def test_attn_residual_block_size_divides(self) -> None:
        with pytest.raises(ValueError, match="attn_residual_block_size"):
            ModelConfig(attn_residual=True, num_layers=12, attn_residual_block_size=5)

    def test_attn_residual_valid(self) -> None:
        cfg = ModelConfig(attn_residual=True, num_layers=12, attn_residual_block_size=4)
        assert cfg.attn_residual_block_size == 4

    def test_invalid_ffn_activation(self) -> None:
        with pytest.raises(ValueError, match="ffn_activation"):
            ModelConfig(ffn_activation="tanh")

    def test_partial_rope_dimension_mismatch(self) -> None:
        with pytest.raises(ValueError, match="nope_dim.*rope_dim.*must equal head_dim"):
            ModelConfig(partial_rope=True, rope_dim=16, nope_dim=16, head_dim=64)


class TestOplmConfig:
    def test_default_composition(self) -> None:
        cfg = OplmConfig()
        assert isinstance(cfg.model, ModelConfig)
        assert isinstance(cfg.train, TrainConfig)
        assert cfg.train.lr == 1e-4


class TestLoadConfig:
    def test_defaults(self) -> None:
        cfg = load_config([])
        assert isinstance(cfg, OplmConfig)
        assert cfg.model.hidden_dim == 768

    def test_dotlist_override(self) -> None:
        cfg = load_config(["model.hidden_dim=256", "model.num_heads=4", "model.num_kv_heads=2"])
        assert cfg.model.hidden_dim == 256
        assert cfg.model.num_heads == 4

    def test_yaml_file(self, tmp_path: Path) -> None:
        yaml_content = "model:\n  hidden_dim: 512\n  num_heads: 8\n  num_kv_heads: 4\n"
        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml_content)

        cfg = load_config(["--config", str(config_file)])
        assert cfg.model.hidden_dim == 512
        assert cfg.model.num_heads == 8

    def test_yaml_plus_overrides(self, tmp_path: Path) -> None:
        yaml_content = "model:\n  hidden_dim: 512\n  num_heads: 8\n  num_kv_heads: 4\n"
        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml_content)

        cfg = load_config(["--config", str(config_file), "model.num_layers=24"])
        assert cfg.model.hidden_dim == 512
        assert cfg.model.num_layers == 24

    def test_derived_fields_computed(self) -> None:
        cfg = load_config(["model.hidden_dim=256", "model.num_heads=4", "model.num_kv_heads=2"])
        assert cfg.model.head_dim == 64  # 256 // 4
