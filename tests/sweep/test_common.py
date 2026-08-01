from __future__ import annotations

import json
from pathlib import Path

import pytest

from oplm.sweep.common import (
    PhaseManifest,
    RunSpec,
    accelerate_argv,
    gradient_accumulation_steps,
    load_phase,
    parse_candidates,
    relative_path,
    result_metric,
    write_phase,
)


def test_gradient_accumulation_uses_global_examples() -> None:
    assert gradient_accumulation_steps(2048, per_device_batch=32, world_size=8) == 8
    assert gradient_accumulation_steps(8192, per_device_batch=32, world_size=8) == 32


@pytest.mark.parametrize("global_examples", [0, 2050])
def test_gradient_accumulation_rejects_invalid_global_batch(global_examples: int) -> None:
    with pytest.raises(ValueError, match="global examples"):
        gradient_accumulation_steps(global_examples, per_device_batch=32, world_size=8)


def test_phase_manifest_roundtrips(tmp_path: Path) -> None:
    phase = PhaseManifest(
        version=1,
        phase="coarse",
        metric="eval/heldout/loss",
        source=None,
        runs=[
            RunSpec(
                id="170M-lr0.01-om1-a0-s42",
                config="runs/170M-lr0.01-om1-a0-s42/run.yaml",
                result="runs/170M-lr0.01-om1-a0-s42/result.json",
                params={"preset": "170M", "lr": 0.01, "seed": 42},
            )
        ],
        ranking=[],
        selected=[],
    )
    path = tmp_path / "phase.json"
    write_phase(path, phase)
    assert load_phase(path) == phase


def test_result_metric_requires_finite_validation_loss(tmp_path: Path) -> None:
    run = RunSpec("run", "runs/run/run.yaml", "runs/run/result.json", {"lr": 0.01})
    result = tmp_path / run.result
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps({"eval": {"eval/heldout/loss": 1.25}}))
    assert result_metric(tmp_path, run, "eval/heldout/loss") == 1.25

    result.write_text(json.dumps({"eval": {"eval/heldout/loss": float("nan")}}))
    assert result_metric(tmp_path, run, "eval/heldout/loss") is None
    result.unlink()
    assert result_metric(tmp_path, run, "eval/heldout/loss") is None


def test_accelerate_command_targets_one_resolved_cell(tmp_path: Path) -> None:
    argv = accelerate_argv(
        config=tmp_path / "run.yaml",
        result=tmp_path / "result.json",
        num_processes=8,
        accelerate_config=tmp_path / "accelerate.yaml",
    )
    assert argv == [
        "accelerate",
        "launch",
        "--config_file",
        str(tmp_path / "accelerate.yaml"),
        "--num_processes",
        "8",
        "-m",
        "oplm.sweep.run",
        "--config",
        str(tmp_path / "run.yaml"),
        "--result",
        str(tmp_path / "result.json"),
    ]


def test_candidate_and_relative_path_parsing(tmp_path: Path) -> None:
    assert parse_candidates("0.01:1,0.016:0.5", ("lr", "output_mult")) == [
        {"lr": 0.01, "output_mult": 1.0},
        {"lr": 0.016, "output_mult": 0.5},
    ]
    root = tmp_path / "sweep"
    source = root / "coarse" / "phase.json"
    target = root / "refine"
    assert relative_path(source, target) == Path("../coarse/phase.json")
