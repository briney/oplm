"""Training-config tests for load_config (model + TrainConfig surface)."""

from __future__ import annotations

from dataclasses import asdict
from importlib.resources import files
from typing import TYPE_CHECKING, Any

import pytest
from omegaconf import OmegaConf

from oplm.config import DataConfig, TrainConfig, load_config
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
    cfg = load_config(["--preset", "small", "model.num_hidden_layers=4"])
    assert cfg.model.num_hidden_layers == 4  # CLI override wins
    assert cfg.model.hidden_size == 256  # from the `small` preset


def test_derived_fields_resolve_when_omitted() -> None:
    """Omitted ``head_dim`` / ``intermediate_size`` resolve from the source dims."""
    cfg = load_config(["model.hidden_size=256", "model.num_attention_heads=4"])
    assert cfg.model.head_dim == 64  # 256 / 4
    assert cfg.model.intermediate_size > 0
    assert cfg.model.intermediate_size % 256 == 0  # rounded to a tensor-core multiple


def test_unknown_model_key_is_absorbed_not_raised() -> None:
    """Unknown ``model.*`` keys flow into PretrainedConfig kwargs (documented caveat)."""
    cfg = load_config(["model.bogus_key=1"])
    assert cfg.model.bogus_key == 1


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


@pytest.mark.parametrize("key", sorted(_TRAIN_YAML))
def test_train_base_yaml_matches_dataclass(key: str) -> None:
    """Every ``train/base.yaml`` key equals its ``TrainConfig`` default."""
    defaults = asdict(TrainConfig())
    assert key in defaults, f"{key!r} is not a TrainConfig field"
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

    Unknown keys are silently absorbed into PretrainedConfig kwargs at runtime, so
    this is the only guard against a misspelled model default.
    """
    assert key in set(OplmModelConfig().to_dict())
