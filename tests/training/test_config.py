"""Training-config tests for load_config (TrainConfig surface)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from oplm.config import load_config

if TYPE_CHECKING:
    from pathlib import Path


def _write_yaml(tmp_path: Path, body: str) -> str:
    """Write a YAML config file and return its path."""
    path = tmp_path / "config.yaml"
    path.write_text(body)
    return str(path)


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
