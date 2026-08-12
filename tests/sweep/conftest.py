"""Shared fixtures for `oplm.sweep.phases` job-emission tests.

Each fixture writes a temp base config carrying the `slurm:` block from
`tests/slurm/test_config.py::RAW` plus the muP requirements every generated cell validates, then
drives the matching phase generator and returns the resulting `phase.json` path.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from omegaconf import OmegaConf
from typer.testing import CliRunner

from oplm.sweep import phases
from oplm.sweep.common import PhaseManifest, load_phase, write_phase
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


def _fabricate_finite_result(phase_path: Path, *, loss: float = 1.5) -> None:
    """Write a finite `result.json` for a freshly generated phase's first (only) run.

    Generation alone always leaves `.selected` empty -- only `analyze_phase` (run against real
    `result.json` files) populates it. Fixtures that need an already-confirmed winner fabricate
    one result and analyze it, mirroring what a real completed run would leave on disk.
    """
    phase = load_phase(phase_path)
    result_path = phase_path.parent / phase.runs[0].result
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps({"eval": {phase.metric: loss}}))


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
    """A generated `transfer` phase: one finalist's depth-ray LR bracket, all at 170M width."""
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


def _generate_multi_preset_phase(tmp_path: Path, *, submit: bool = False) -> Path:
    """Generate a phase spanning 400M/800M/1B (four cells each) via `_generate_phase`.

    No CLI phase spans presets anymore (`transfer` became a single-preset depth ray), but the
    multi-preset array and status machinery in `_write_jobs` and `oplm sweep status` is still
    live code. This drives the same generation path every phase command calls, with hand-built
    cells at the three multi-node presets.
    """
    config = _write_base_config(tmp_path)
    cells = [
        phases._cell(
            preset=preset,
            lr=lr,
            output_mult=1.0,
            depth_exponent=0.0,
            seed=42,
            global_examples=2048,
            max_steps=10000,
            warmup_steps=1000,
        )
        for preset in ("400M", "800M", "1B")
        for lr in (0.01, 0.0126, 0.016, 0.02)
    ]
    phase_path, _ = phases._generate_phase(
        name="grid",
        base_config=config,
        out=tmp_path / "grid",
        metric=None,
        source=None,
        cells=cells,
        num_processes=8,
        local=False,
        accelerate_config=None,
        submit=submit,
    )
    return phase_path


@pytest.fixture
def multi_preset_phase(tmp_path: Path) -> Path:
    """A generated phase spanning the 400M/800M/1B presets, one array job each."""
    return _generate_multi_preset_phase(tmp_path)


@pytest.fixture
def confirm_phase(tmp_path: Path) -> Path:
    """An analyzed `confirm` phase (800M) with its single candidate confirmed.

    Sourced from a bridge winner, then given a fabricated finite result and analyzed, so
    `.selected` carries the confirmed winner exactly as `oplm sweep scale` expects to read it --
    not just the empty `.selected` a bare `generate` step leaves behind.
    """
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
    phase_path = out / "phase.json"
    _fabricate_finite_result(phase_path)
    phases.analyze_phase(phase_path)
    return phase_path
