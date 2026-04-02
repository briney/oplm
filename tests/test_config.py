"""Tests for the configuration system."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from oplm.config import (
    ModelConfig,
    OplmConfig,
    TrainConfig,
    load_config,
    parse_train_configs,
    round_multiple,
)


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
        assert cfg.conv_kernel_schedule == "static"
        assert cfg.conv_kernel_increment == 2
        assert cfg.conv_kernel_block_size == 1
        assert cfg.conv_kernel_max_size is None

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

    def test_invalid_conv_kernel_schedule(self) -> None:
        with pytest.raises(ValueError, match="conv_kernel_schedule"):
            ModelConfig(conv_kernel_schedule="per_layer")

    def test_conv_kernel_increment_must_be_non_negative_even(self) -> None:
        with pytest.raises(ValueError, match="conv_kernel_increment"):
            ModelConfig(conv_kernel_schedule="block_step", conv_kernel_increment=3)

        with pytest.raises(ValueError, match="conv_kernel_increment"):
            ModelConfig(conv_kernel_schedule="block_step", conv_kernel_increment=-2)

    def test_conv_kernel_block_size_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="conv_kernel_block_size"):
            ModelConfig(conv_kernel_schedule="block_step", conv_kernel_block_size=0)

    def test_conv_kernel_max_size_must_be_odd(self) -> None:
        with pytest.raises(ValueError, match="conv_kernel_max_size.*must be odd"):
            ModelConfig(conv_kernel_schedule="block_step", conv_kernel_max_size=8)

    def test_conv_kernel_max_size_must_not_be_smaller_than_base(self) -> None:
        with pytest.raises(ValueError, match="conv_kernel_max_size"):
            ModelConfig(
                conv_kernel_schedule="block_step",
                conv_kernel_size=7,
                conv_kernel_max_size=5,
            )

    def test_static_schedule_ignores_schedule_only_fields(self) -> None:
        cfg = ModelConfig(
            conv_kernel_schedule="static",
            conv_kernel_increment=-2,
            conv_kernel_block_size=0,
            conv_kernel_max_size=8,
        )
        assert cfg.conv_kernel_size_for_layer(3) == 7

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


class TestConvKernelSchedule:
    def test_static_returns_same_kernel_for_all_layers(self) -> None:
        cfg = ModelConfig(conv_kernel_size=9)
        assert [cfg.conv_kernel_size_for_layer(i) for i in range(4)] == [9, 9, 9, 9]

    def test_block_step_increments_every_layer(self) -> None:
        cfg = ModelConfig(
            conv_kernel_size=3,
            conv_kernel_schedule="block_step",
            conv_kernel_increment=2,
            conv_kernel_block_size=1,
        )
        assert [cfg.conv_kernel_size_for_layer(i) for i in range(4)] == [3, 5, 7, 9]

    def test_block_step_increments_by_block(self) -> None:
        cfg = ModelConfig(
            conv_kernel_size=3,
            conv_kernel_schedule="block_step",
            conv_kernel_increment=2,
            conv_kernel_block_size=2,
        )
        assert [cfg.conv_kernel_size_for_layer(i) for i in range(6)] == [3, 3, 5, 5, 7, 7]

    def test_block_step_clamps_to_max_size(self) -> None:
        cfg = ModelConfig(
            conv_kernel_size=3,
            conv_kernel_schedule="block_step",
            conv_kernel_increment=2,
            conv_kernel_block_size=2,
            conv_kernel_max_size=7,
        )
        assert [cfg.conv_kernel_size_for_layer(i) for i in range(8)] == [3, 3, 5, 5, 7, 7, 7, 7]


class TestParseTrainConfigs:
    def test_none_returns_empty(self) -> None:
        assert parse_train_configs(None) == []

    def test_empty_string_returns_empty(self) -> None:
        assert parse_train_configs("") == []

    def test_empty_dict_returns_empty(self) -> None:
        assert parse_train_configs({}) == []

    def test_string_path(self) -> None:
        entries = parse_train_configs("/path/to/dataset")
        assert len(entries) == 1
        assert entries[0].name == "default"
        assert entries[0].path == "/path/to/dataset"
        assert entries[0].fraction == 1.0

    def test_single_dict_entry(self) -> None:
        raw = {"uniref": {"path": "/data/uniref", "fraction": 0.7}}
        entries = parse_train_configs(raw)
        assert len(entries) == 1
        assert entries[0].name == "uniref"
        assert entries[0].path == "/data/uniref"
        assert entries[0].fraction == 1.0  # single entry always 1.0

    def test_multiple_with_fractions(self) -> None:
        raw = {
            "ds_a": {"path": "/a", "fraction": 0.6},
            "ds_b": {"path": "/b", "fraction": 0.4},
        }
        entries = parse_train_configs(raw)
        assert len(entries) == 2
        assert entries[0].fraction == pytest.approx(0.6)
        assert entries[1].fraction == pytest.approx(0.4)

    def test_fractions_normalized(self) -> None:
        raw = {
            "ds_a": {"path": "/a", "fraction": 3.0},
            "ds_b": {"path": "/b", "fraction": 1.0},
        }
        entries = parse_train_configs(raw)
        assert entries[0].fraction == pytest.approx(0.75)
        assert entries[1].fraction == pytest.approx(0.25)

    def test_omitted_fractions_shared(self) -> None:
        raw = {
            "ds_a": {"path": "/a", "fraction": 0.5},
            "ds_b": {"path": "/b"},
            "ds_c": {"path": "/c"},
        }
        entries = parse_train_configs(raw)
        total = sum(e.fraction for e in entries)
        assert total == pytest.approx(1.0)
        # ds_a gets 0.5 / total, ds_b and ds_c split remaining 0.5
        assert entries[1].fraction == pytest.approx(entries[2].fraction)

    def test_all_fractions_omitted(self) -> None:
        raw = {
            "ds_a": {"path": "/a"},
            "ds_b": {"path": "/b"},
            "ds_c": {"path": "/c"},
        }
        entries = parse_train_configs(raw)
        for e in entries:
            assert e.fraction == pytest.approx(1.0 / 3.0)

    def test_string_values_in_dict(self) -> None:
        raw = {"ds_a": "/path/a", "ds_b": "/path/b"}
        entries = parse_train_configs(raw)
        assert len(entries) == 2
        assert entries[0].path == "/path/a"
        assert entries[1].path == "/path/b"

    def test_negative_fraction_raises(self) -> None:
        raw = {
            "ds_a": {"path": "/a", "fraction": -0.5},
            "ds_b": {"path": "/b", "fraction": 0.5},
        }
        with pytest.raises(ValueError, match="fraction must be >= 0"):
            parse_train_configs(raw)

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid data.train config type"):
            parse_train_configs(42)  # type: ignore[arg-type]

    def test_invalid_entry_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid train config"):
            parse_train_configs({"ds_a": 42})  # type: ignore[dict-item]

    def test_none_values_skipped(self) -> None:
        raw = {"ds_a": {"path": "/a"}, "ds_b": None}
        entries = parse_train_configs(raw)
        assert len(entries) == 1
        assert entries[0].fraction == 1.0

    def test_missing_path_skipped(self) -> None:
        raw = {"ds_a": {"path": "/a"}, "ds_b": {"fraction": 0.5}}
        entries = parse_train_configs(raw)
        assert len(entries) == 1


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

    def test_model_max_seq_len_override_is_canonical(self) -> None:
        cfg = load_config(["model.max_seq_len=256"])
        assert cfg.model.max_seq_len == 256
        assert cfg.data.max_length == 256

    def test_legacy_data_max_length_override_maps_with_warning(self) -> None:
        with pytest.warns(DeprecationWarning, match="data.max_length"):
            cfg = load_config(["data.max_length=256"])

        assert cfg.model.max_seq_len == 256
        assert cfg.data.max_length == 256

    def test_equal_dual_sequence_length_settings_warn_and_pass(self) -> None:
        with pytest.warns(DeprecationWarning, match="data.max_length"):
            cfg = load_config(["model.max_seq_len=256", "data.max_length=256"])

        assert cfg.model.max_seq_len == 256
        assert cfg.data.max_length == 256

    def test_mismatched_dual_sequence_length_settings_raise(self) -> None:
        with pytest.raises(ValueError, match="Conflicting sequence length settings"):
            load_config(["model.max_seq_len=256", "data.max_length=128"])

    def test_yaml_file(self, tmp_path: Path) -> None:
        yaml_content = "model:\n  hidden_dim: 512\n  num_heads: 8\n  num_kv_heads: 4\n"
        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml_content)

        cfg = load_config(["--config", str(config_file)])
        assert cfg.model.hidden_dim == 512
        assert cfg.model.num_heads == 8
        assert cfg.train.config_path == str(config_file.resolve())

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

    def test_data_train_single_path(self) -> None:
        cfg = load_config(["data.train=/path/to/dataset"])
        assert cfg.data.train == "/path/to/dataset"

    def test_data_train_multi_dataset(self) -> None:
        cfg = load_config(
            [
                "data.train.ds_a.path=/path/a",
                "data.train.ds_a.fraction=0.6",
                "data.train.ds_b.path=/path/b",
                "data.train.ds_b.fraction=0.4",
            ]
        )
        assert isinstance(cfg.data.train, dict)
        entries = parse_train_configs(cfg.data.train)
        assert len(entries) == 2
        paths = {e.path for e in entries}
        assert paths == {"/path/a", "/path/b"}

    def test_data_train_from_yaml(self, tmp_path: Path) -> None:
        yaml_content = (
            "data:\n"
            "  train:\n"
            "    uniref:\n"
            "      path: /data/uniref\n"
            "      fraction: 0.7\n"
            "    bfd:\n"
            "      path: /data/bfd\n"
            "      fraction: 0.3\n"
        )
        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml_content)

        cfg = load_config(["--config", str(config_file)])
        entries = parse_train_configs(cfg.data.train)
        assert len(entries) == 2
        assert entries[0].fraction == pytest.approx(0.7)
        assert entries[1].fraction == pytest.approx(0.3)

    def test_data_config_defaults(self) -> None:
        cfg = load_config([])
        assert cfg.model.max_seq_len == 512
        assert cfg.data.max_length == 512
        assert cfg.data.mask_prob == 0.15
        assert cfg.data.num_workers == 4
        assert cfg.data.train is None
