"""Training-config tests for load_config (model + TrainConfig surface)."""

from __future__ import annotations

from dataclasses import asdict
from importlib.resources import files
from typing import TYPE_CHECKING, Any

import pytest
from omegaconf import OmegaConf

from oplm.config import AVAILABLE_PRESETS, DataConfig, TrainConfig, load_config, serialize_config
from oplm.model import OplmConfig as OplmModelConfig

if TYPE_CHECKING:
    from pathlib import Path


def _write_yaml(tmp_path: Path, body: str) -> str:
    """Write a YAML config file and return its path."""
    path = tmp_path / "config.yaml"
    path.write_text(body)
    return str(path)


def _packaged_section(package: str, filename: str, section: str) -> dict[str, Any]:
    """Load one top-level section from a packaged ``base.yaml`` as a plain dict."""
    text = files(package).joinpath(filename).read_text()
    container = OmegaConf.to_container(OmegaConf.create(text), resolve=True)
    assert isinstance(container, dict)
    return container[section]  # type: ignore[return-value]


# --- --name → train.wandb_run_name propagation ---------------------------------


def test_name_flag_sets_run_name() -> None:
    """``--name`` populates ``train.wandb_run_name`` when nothing else sets it."""
    cfg = load_config(["--name", "from_flag"])
    assert cfg.train.wandb_run_name == "from_flag"


def test_run_name_defaults_to_none() -> None:
    """With neither ``--name`` nor an explicit value, the run name stays None."""
    cfg = load_config([])
    assert cfg.train.wandb_run_name is None


def test_cli_override_beats_name_flag() -> None:
    """An explicit ``train.wandb_run_name`` CLI override wins over ``--name``."""
    cfg = load_config(["--name", "from_flag", "train.wandb_run_name=from_cli"])
    assert cfg.train.wandb_run_name == "from_cli"


def test_yaml_value_beats_name_flag(tmp_path: Path) -> None:
    """An explicit ``train.wandb_run_name`` in YAML wins over ``--name``."""
    config_path = _write_yaml(tmp_path, "train:\n  wandb_run_name: from_yaml\n")
    cfg = load_config(["--name", "from_flag", "--config", config_path])
    assert cfg.train.wandb_run_name == "from_yaml"


def test_explicit_null_override_beats_name_flag() -> None:
    """An explicit ``wandb_run_name=null`` is honored (random W&B name), not ``--name``."""
    cfg = load_config(["--name", "from_flag", "train.wandb_run_name=null"])
    assert cfg.train.wandb_run_name is None


def test_run_name_without_flag() -> None:
    """An explicit run name with no ``--name`` flag resolves to that value."""
    cfg = load_config(["train.wandb_run_name=from_cli"])
    assert cfg.train.wandb_run_name == "from_cli"


# --- --name → train.output_dir propagation -------------------------------------


def test_name_flag_sets_output_dir() -> None:
    """``--name`` seeds ``train.output_dir`` to ``./<name>`` when nothing else sets it."""
    cfg = load_config(["--name", "myrun"])
    assert cfg.train.output_dir == "myrun"


def test_output_dir_defaults_without_name() -> None:
    """With neither ``--name`` nor an explicit value, output_dir keeps the default."""
    cfg = load_config([])
    assert cfg.train.output_dir == "outputs"


def test_cli_output_dir_beats_name_flag() -> None:
    """An explicit ``train.output_dir`` CLI override wins over ``--name``."""
    cfg = load_config(["--name", "myrun", "train.output_dir=explicit"])
    assert cfg.train.output_dir == "explicit"


def test_yaml_output_dir_beats_name_flag(tmp_path: Path) -> None:
    """An explicit ``train.output_dir`` in YAML wins over ``--name``."""
    config_path = _write_yaml(tmp_path, "train:\n  output_dir: from_yaml\n")
    cfg = load_config(["--name", "myrun", "--config", config_path])
    assert cfg.train.output_dir == "from_yaml"


# --- model config: type, presets, derived fields, unknown keys -----------------


def test_default_model_is_hf_config() -> None:
    """``load_config([])`` resolves ``cfg.model`` to a validated HF ``OplmConfig``."""
    cfg = load_config([])
    assert isinstance(cfg.model, OplmModelConfig)
    # Construction runs OplmConfig._validate(); reaching here means it passed. The
    # derived fields are concrete (never None) post-__init__.
    assert cfg.model.head_dim is not None
    assert cfg.model.intermediate_size is not None


def test_preset_and_cli_override_apply() -> None:
    """A size preset sets dims; a CLI dotlist override beats the preset/base layer."""
    cfg = load_config(["--preset", "50M", "model.num_hidden_layers=4"])
    assert cfg.model.num_hidden_layers == 4  # CLI override wins
    assert cfg.model.hidden_size == 512  # from the `50M` preset


def test_ablation_toggle_cli_overrides_apply() -> None:
    """CLI dotlist overrides reach the new architecture-ablation knobs."""
    cfg = load_config(
        [
            "model.qk_norm_mode=l2",
            "model.qk_norm_l2_scale_init=8.0",
            "model.mask_dropout=true",
            "model.residual_gate=channel",
            "model.residual_gate_init=0.5",
            "model.ffn_activation=relu2",
        ]
    )
    assert cfg.model.qk_norm_mode == "l2"
    assert cfg.model.qk_norm_l2_scale_init == 8.0
    assert cfg.model.mask_dropout is True
    assert cfg.model.residual_gate == "channel"
    assert cfg.model.residual_gate_init == 0.5
    assert cfg.model.ffn_activation == "relu2"


def test_ablation_toggle_defaults_preserve_current_behavior() -> None:
    """With no overrides the new knobs resolve to behavior-preserving defaults."""
    cfg = load_config([])
    assert cfg.model.qk_norm_mode == "channel"
    assert cfg.model.qk_norm_l2_scale_init is None
    assert cfg.model.mask_dropout is False
    assert cfg.model.mask_dropout_reference_ratio == 0.12
    assert cfg.model.residual_gate == "none"
    assert cfg.model.residual_gate_init == 1.0
    assert cfg.model.canon_residual is True


def test_mup_cli_overrides_apply() -> None:
    """CLI dotlist overrides reach the μP knobs and the derived multiplier updates."""
    cfg = load_config(
        [
            "--preset",
            "400M",  # hidden_size = 1024
            "model.mup_enable=true",
            "model.mup_base_width=512",
            "model.mup_output_mult=2.0",
        ]
    )
    assert cfg.model.mup_enable is True
    assert cfg.model.mup_base_width == 512
    assert cfg.model.mup_output_mult == 2.0
    assert cfg.model.mup_width_mult() == 2.0  # 1024 / 512


def test_default_run_uses_mup_muon_recipe() -> None:
    """With no overrides the production default is μP + Muon at base LR 0.01."""
    cfg = load_config([])
    assert cfg.train.optimizer == "muon"
    assert cfg.train.muon_adjust_lr_fn == "original"
    assert cfg.train.lr == pytest.approx(0.01)
    assert cfg.model.mup_enable is True
    assert cfg.model.mup_base_width == 512
    assert cfg.model.mup_output_mult == 1.0
    # Default hidden_size 1024 over base 512 ⇒ m = 2 (μP actively scales here).
    assert cfg.model.mup_width_mult() == 2.0


def test_baseline_adamw_overlay_disables_mup() -> None:
    """The AdamW baseline overlay turns μP off and switches the optimizer to AdamW."""
    overlay = str(files("oplm.configs.train").joinpath("baseline_adamw.yaml"))
    cfg = load_config(["--config", overlay])
    assert cfg.train.optimizer == "adamw"
    assert cfg.train.muon_adjust_lr_fn == "match_rms_adamw"
    assert cfg.model.mup_enable is False
    assert cfg.model.mup_width_mult() == 1.0


def test_dataclass_fallbacks_are_conservative() -> None:
    """Bare dataclass construction stays μP-off / AdamW (library fallback, backward-compatible)."""
    assert OplmModelConfig().mup_enable is False
    assert TrainConfig().optimizer == "adamw"
    assert TrainConfig().lr == pytest.approx(1e-4)


def test_canon_residual_cli_override_applies() -> None:
    """The residual Canon toggle remains a simple boolean model field."""
    cfg = load_config(["model.canon_residual=false"])
    assert cfg.model.canon_enabled is False
    assert cfg.model.canon_residual is False


def test_derived_fields_resolve_when_omitted() -> None:
    """Omitted ``head_dim`` / ``intermediate_size`` resolve from the source dims."""
    cfg = load_config(["model.hidden_size=256", "model.num_attention_heads=4"])
    assert cfg.model.head_dim == 64  # 256 / 4
    assert cfg.model.intermediate_size > 0
    assert cfg.model.intermediate_size % 256 == 0  # rounded to a tensor-core multiple


def test_relu2_intermediate_size_derives_iso_param() -> None:
    """relu2 (2 projections) derives ~4*D so FFN params match the gated ~8/3*D variants."""
    gated = load_config(["model.hidden_size=768"])
    assert gated.model.intermediate_size == 2048  # round_up(8/3 * 768, 256)
    relu2 = load_config(["model.hidden_size=768", "model.ffn_activation=relu2"])
    assert relu2.model.intermediate_size == 3072  # round_up(4 * 768, 256)
    # Iso-param: 2 projections * 3072 == 3 projections * 2048.
    assert 2 * relu2.model.intermediate_size == 3 * gated.model.intermediate_size


def test_unknown_model_key_raises() -> None:
    """Unknown ``model.*`` keys are rejected, not silently absorbed into kwargs."""
    with pytest.raises(ValueError, match="Unknown model config key"):
        load_config(["model.bogus_key=1"])


def test_misspelled_canon_key_raises() -> None:
    """A typo'd ablation key (``cannon_enabled``) raises instead of silently no-op'ing."""
    with pytest.raises(ValueError, match="cannon_enabled"):
        load_config(["model.cannon_enabled=true"])


def test_misspelled_dimension_key_raises() -> None:
    """A typo'd dimension key (``hidden_dimm``) is rejected at load time."""
    with pytest.raises(ValueError, match="Unknown model config key"):
        load_config(["model.hidden_dimm=1024"])


@pytest.mark.parametrize("preset", AVAILABLE_PRESETS)
def test_all_presets_pass_strict_validation(preset: str) -> None:
    """Every packaged size preset loads cleanly under strict key validation."""
    cfg = load_config(["--preset", preset])
    assert isinstance(cfg.model, OplmModelConfig)


@pytest.mark.parametrize("preset", AVAILABLE_PRESETS)
def test_all_presets_resolve_golden_canon_defaults(preset: str) -> None:
    """Every preset resolves the paper-exact Canon defaults (Canon off, residual on)."""
    cfg = load_config(["--preset", preset])
    assert cfg.model.canon_enabled is False
    assert cfg.model.canon_residual is True
    assert cfg.model.canon_positions == []
    assert cfg.model.canon_activation == "none"


def test_canon_enabled_run_resolves_paper_exact_defaults() -> None:
    """A Canon-enabled run resolves to the paper-exact encoder configuration.

    Golden assertions for the ablation surface: residual Canon, no conv
    activation, pre-norm strategy, and the base k=4 kernel broadcast across
    every layer.
    """
    cfg = load_config(["model.canon_enabled=true", "model.canon_positions=[A,B,C,D]"])
    assert cfg.model.canon_enabled is True
    assert cfg.model.canon_residual is True
    assert cfg.model.canon_positions == ["A", "B", "C", "D"]
    assert cfg.model.canon_activation == "none"
    assert cfg.model.norm_strategy == "pre"
    # base.yaml's k=4 resolves to a per-layer list of the right length.
    assert cfg.model.canon_kernel_sizes == [4] * cfg.model.num_hidden_layers


def test_serialized_config_roundtrips_through_load_config(tmp_path: Path) -> None:
    """A serialized run config (HF ``to_dict`` metadata included) reloads via ``--config``.

    Strict validation must allow the HF metadata keys (``model_type``,
    ``transformers_version``, ``architectures``, ...) that ``serialize_config``
    emits, or every saved config would fail to reload.
    """
    cfg = load_config(["model.canon_enabled=true", "model.canon_positions=[A,D]"])
    path = tmp_path / "run.yaml"
    path.write_text(serialize_config(cfg))
    restored = load_config(["--config", str(path)])
    assert restored.model.canon_enabled is True
    assert restored.model.canon_positions == ["A", "D"]


def test_removed_max_length_alias_raises() -> None:
    """The removed ``data.max_length`` alias points users at the HF field name."""
    with pytest.raises(ValueError, match="model.max_position_embeddings"):
        load_config(["data.max_length=5"])


# --- base.yaml layer is actually loaded (not just documentation) ----------------


def test_base_layer_is_loaded() -> None:
    """The packaged ``train/base.yaml`` value is what ``load_config`` returns."""
    yaml_train = _packaged_section("oplm.configs.train", "base.yaml", "train")
    assert yaml_train  # the section is non-empty
    assert load_config([]).train.seed == yaml_train["seed"]


# --- YAML↔dataclass drift guards -----------------------------------------------

_TRAIN_YAML = _packaged_section("oplm.configs.train", "base.yaml", "train")
_DATA_YAML = _packaged_section("oplm.configs.data", "base.yaml", "data")
_MODEL_YAML = _packaged_section("oplm.configs.model", "base.yaml", "model")


# train/base.yaml carries OPLM's opinionated production defaults (μP + Muon), which
# intentionally diverge from the conservative TrainConfig dataclass fallbacks for
# these fields. Every other key must still mirror the dataclass (drift guard).
_TRAIN_PRODUCTION_OVERRIDES = {"optimizer", "lr", "muon_adjust_lr_fn"}


@pytest.mark.parametrize("key", sorted(_TRAIN_YAML))
def test_train_base_yaml_matches_dataclass(key: str) -> None:
    """Every ``train/base.yaml`` key is a ``TrainConfig`` field; non-overrides equal the default."""
    defaults = asdict(TrainConfig())
    assert key in defaults, f"{key!r} is not a TrainConfig field"
    if key not in _TRAIN_PRODUCTION_OVERRIDES:
        assert _TRAIN_YAML[key] == defaults[key]


@pytest.mark.parametrize("key", sorted(_DATA_YAML))
def test_data_base_yaml_matches_dataclass(key: str) -> None:
    """Every ``data/base.yaml`` key equals its ``DataConfig`` default (train/eval are None)."""
    defaults = asdict(DataConfig())
    assert key in defaults, f"{key!r} is not a DataConfig field"
    assert _DATA_YAML[key] == defaults[key]


@pytest.mark.parametrize("key", sorted(_MODEL_YAML))
def test_model_base_yaml_has_no_typos(key: str) -> None:
    """Every ``model/base.yaml`` key is a recognized HF ``OplmConfig`` field.

    ``load_config`` now rejects unknown ``model.*`` keys at runtime, but this
    guards the packaged default itself (which is merged in before any override).
    """
    assert key in set(OplmModelConfig().to_dict())


# --- torch.compile config fields -----------------------------------------------


def test_compile_defaults() -> None:
    """``compile`` is False and ``compile_mode`` is 'default' out of the box."""
    cfg = TrainConfig(wandb_enabled=False)
    assert cfg.compile is False
    assert cfg.compile_mode == "default"


@pytest.mark.parametrize("mode", ["default", "reduce-overhead", "max-autotune"])
def test_compile_mode_valid(mode: str) -> None:
    """Each recognized compile mode is accepted without error."""
    cfg = TrainConfig(wandb_enabled=False, compile_mode=mode)
    assert cfg.compile_mode == mode


def test_compile_mode_invalid() -> None:
    """An unrecognized compile_mode raises ValueError."""
    with pytest.raises(ValueError, match="compile_mode"):
        TrainConfig(wandb_enabled=False, compile_mode="turbo")
