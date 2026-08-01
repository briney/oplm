"""Shared fixtures for `oplm.sweep.phases` job-emission tests.

Each fixture writes a temp base config carrying the `slurm:` block from
`tests/slurm/test_config.py::RAW` plus the muP requirements every generated cell validates, then
drives the matching phase generator and returns the resulting `phase.json` path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from omegaconf import OmegaConf
from typer.testing import CliRunner

from oplm.sweep import phases
from oplm.sweep.common import PhaseManifest, write_phase
from tests.slurm.test_config import RAW as SLURM_RAW

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()

_BASE_CONFIG_BODY = """
model:
  norm_strategy: sandwich
  canon_enabled: true
  canon_positions: [A, B, C, D]
  residual_gate: channel
train:
  batch_size: 32
  optimizer: muon
  weight_decay: 0.01
  muon_adjust_lr_fn: original
  wandb_enabled: false
  save_final: false
data:
  train: /data/train
  eval:
    heldout:
      path: /data/heldout.parquet
      type: sequence
      every: {steps: 500}
""".lstrip()


def _base_config_text() -> str:
    """Model/train/data YAML plus the shared `slurm:` block from `tests/slurm/test_config.py`."""
    return _BASE_CONFIG_BODY + OmegaConf.to_yaml(OmegaConf.create({"slurm": dict(SLURM_RAW)}))


def _write_base_config(tmp_path: Path) -> Path:
    path = tmp_path / "base.yaml"
    path.write_text(_base_config_text())
    return path


def _write_selected(path: Path, phase: str, selected: list[dict[str, float]]) -> Path:
    write_phase(path, PhaseManifest(1, phase, "eval/heldout/loss", None, [], [], selected))
    return path


@pytest.fixture
def coarse_phase(tmp_path: Path) -> Path:
    """A generated `coarse` phase: single preset (170M), one array job."""
    config = _write_base_config(tmp_path)
    out = tmp_path / "coarse"
    result = runner.invoke(phases.app, ["coarse", "--config", str(config), "--out", str(out)])
    assert result.exit_code == 0, result.output
    return out / "phase.json"


@pytest.fixture
def transfer_phase(tmp_path: Path) -> Path:
    """A generated `transfer` phase spanning the default 400M/800M/1B presets."""
    config = _write_base_config(tmp_path)
    source = _write_selected(
        tmp_path / "replicate" / "phase.json",
        "replicate",
        [{"lr": 0.01, "output_mult": 1.0}],
    )
    out = tmp_path / "transfer"
    result = runner.invoke(
        phases.app,
        ["transfer", "--config", str(config), "--from", str(source), "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    return out / "phase.json"


@pytest.fixture
def confirm_phase(tmp_path: Path) -> Path:
    """A generated `confirm` phase (800M), sourced from a bridge winner."""
    config = _write_base_config(tmp_path)
    source = _write_selected(
        tmp_path / "bridge" / "phase.json",
        "bridge",
        [{"lr": 0.01, "output_mult": 1.0, "depth_exponent": 0.25, "batch_mult": 1.0}],
    )
    out = tmp_path / "confirm"
    result = runner.invoke(
        phases.app,
        ["confirm", "--config", str(config), "--from", str(source), "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    return out / "phase.json"
