from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from scripts import mup_sweep
from scripts._mup_common import PhaseManifest, RunSpec, load_phase, write_phase
from typer.testing import CliRunner

from oplm.config import load_config

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


@pytest.fixture
def base_config(tmp_path: Path) -> Path:
    path = tmp_path / "base.yaml"
    path.write_text(
        """
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
    )
    return path


def test_smoke_generates_three_production_cells(base_config: Path, tmp_path: Path) -> None:
    out = tmp_path / "smoke"
    result = runner.invoke(
        mup_sweep.app,
        ["smoke", "--config", str(base_config), "--out", str(out)],
    )
    assert result.exit_code == 0, result.output

    phase = load_phase(out / "phase.json")
    assert phase.phase == "smoke"
    assert phase.metric == "eval/heldout/loss"
    assert [run.params["lr"] for run in phase.runs] == [0.0025, 0.01, 0.04]
    assert len((out / "commands.txt").read_text().splitlines()) == 3

    cfg = load_config(["--config", str(out / phase.runs[1].config)])
    assert cfg.model.hidden_size == 768
    assert cfg.model.num_hidden_layers == 24
    assert cfg.model.num_attention_heads == 12
    assert cfg.model.norm_strategy == "sandwich"
    assert cfg.model.canon_positions == ["A", "B", "C", "D"]
    assert cfg.model.residual_gate == "channel"
    assert cfg.model.mup_enable is True
    assert cfg.model.mup_base_width == 768
    assert cfg.model.mup_output_mult == 1.0
    assert cfg.train.optimizer == "muon"
    assert cfg.train.weight_decay == 0.01
    assert cfg.train.lr == 0.01
    assert cfg.train.scheduler == "wsd_linear"
    assert cfg.train.max_steps == 1000
    assert cfg.train.max_epochs is None
    assert cfg.train.warmup_steps == 100
    assert cfg.train.stable_steps == 900
    assert cfg.train.gradient_accumulation_steps == 8


def test_generation_rejects_nonproduction_weight_decay(
    base_config: Path, tmp_path: Path
) -> None:
    text = base_config.read_text().replace("weight_decay: 0.01", "weight_decay: 0.0")
    base_config.write_text(text)
    result = runner.invoke(
        mup_sweep.app,
        ["smoke", "--config", str(base_config), "--out", str(tmp_path / "smoke")],
    )
    assert result.exit_code != 0
    assert "weight_decay=0.01" in result.output


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("optimizer: muon", "optimizer: adamw", "optimizer=muon"),
        (
            "muon_adjust_lr_fn: original",
            "muon_adjust_lr_fn: match_rms_adamw",
            "muon_adjust_lr_fn=original",
        ),
    ],
)
def test_generation_rejects_nonproduction_optimizer(
    base_config: Path, tmp_path: Path, old: str, new: str, message: str
) -> None:
    base_config.write_text(base_config.read_text().replace(old, new))
    result = runner.invoke(
        mup_sweep.app,
        ["smoke", "--config", str(base_config), "--out", str(tmp_path / "smoke")],
    )
    assert result.exit_code != 0
    assert message in result.output


def test_generation_rejects_nonintegral_accumulation(
    base_config: Path, tmp_path: Path
) -> None:
    result = runner.invoke(
        mup_sweep.app,
        [
            "smoke",
            "--config",
            str(base_config),
            "--out",
            str(tmp_path / "smoke"),
            "--global-examples",
            "2050",
        ],
    )
    assert result.exit_code != 0
    assert "global examples" in result.output


def test_refine_uses_coarse_selected_lrs(base_config: Path, tmp_path: Path) -> None:
    coarse_dir = tmp_path / "coarse"
    coarse_dir.mkdir()
    write_phase(
        coarse_dir / "phase.json",
        PhaseManifest(
            1,
            "coarse",
            "eval/heldout/loss",
            None,
            [],
            [],
            [
                {"lr": 0.0063, "output_mult": 1.0},
                {"lr": 0.01, "output_mult": 1.0},
                {"lr": 0.016, "output_mult": 1.0},
            ],
        ),
    )
    out = tmp_path / "refine"
    result = runner.invoke(
        mup_sweep.app,
        [
            "refine",
            "--config",
            str(base_config),
            "--from",
            str(coarse_dir / "phase.json"),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    phase = load_phase(out / "phase.json")
    assert len(phase.runs) == 9
    assert {(run.params["lr"], run.params["output_mult"]) for run in phase.runs} == {
        (lr, output_mult)
        for lr in (0.0063, 0.01, 0.016)
        for output_mult in (0.5, 1.0, 2.0)
    }


def _write_smoke_phase(tmp_path: Path, metrics: dict[float, float | None]) -> Path:
    metric = "eval/heldout/loss"
    runs = [
        RunSpec(
            f"lr-{lr:g}",
            f"runs/lr-{lr:g}/run.yaml",
            f"runs/lr-{lr:g}/result.json",
            {"lr": lr},
        )
        for lr in (0.0025, 0.01, 0.04)
    ]
    path = tmp_path / "smoke" / "phase.json"
    write_phase(path, PhaseManifest(1, "smoke", metric, None, runs, [], []))
    for run in runs:
        value = metrics[float(run.params["lr"])]
        if value is not None:
            result = path.parent / run.result
            result.parent.mkdir(parents=True)
            result.write_text(json.dumps({"eval": {metric: value}}))
    return path


@pytest.mark.parametrize(
    ("high_metric", "expected_lrs"),
    [
        (1.2, [0.0025, 0.004, 0.0063, 0.01, 0.016, 0.025, 0.04]),
        (None, [0.0025, 0.004, 0.0063, 0.01, 0.016, 0.025]),
    ],
)
def test_coarse_uses_smoke_divergence_gate(
    base_config: Path,
    tmp_path: Path,
    high_metric: float | None,
    expected_lrs: list[float],
) -> None:
    source = _write_smoke_phase(tmp_path, {0.0025: 1.4, 0.01: 1.1, 0.04: high_metric})
    out = tmp_path / "coarse"
    result = runner.invoke(
        mup_sweep.app,
        [
            "coarse",
            "--config",
            str(base_config),
            "--from",
            str(source),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert [run.params["lr"] for run in load_phase(out / "phase.json").runs] == expected_lrs


@pytest.mark.parametrize("failed_lr", [0.0025, 0.01])
def test_coarse_rejects_nonfinite_required_smoke_cells(
    base_config: Path, tmp_path: Path, failed_lr: float
) -> None:
    metrics: dict[float, float | None] = {0.0025: 1.4, 0.01: 1.1, 0.04: 1.2}
    metrics[failed_lr] = None
    source = _write_smoke_phase(tmp_path, metrics)
    result = runner.invoke(
        mup_sweep.app,
        [
            "coarse",
            "--config",
            str(base_config),
            "--from",
            str(source),
            "--out",
            str(tmp_path / "coarse"),
        ],
    )
    assert result.exit_code != 0
    assert f"{failed_lr:g}" in result.output


def test_generation_requires_metric_for_ambiguous_eval_config(
    base_config: Path, tmp_path: Path
) -> None:
    base_config.write_text(
        base_config.read_text().replace(
            "      every: {steps: 500}\n",
            "      every: {steps: 500}\n"
            "    second:\n"
            "      path: /data/second.parquet\n"
            "      type: sequence\n"
            "      every: {steps: 500}\n",
        )
    )
    result = runner.invoke(
        mup_sweep.app,
        ["smoke", "--config", str(base_config), "--out", str(tmp_path / "smoke")],
    )
    assert result.exit_code != 0
    assert "--metric is required" in result.output
