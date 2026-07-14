from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import typer
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


def _write_selected_phase(
    tmp_path: Path,
    phase: str,
    selected: list[dict[str, float]],
    runs: list[RunSpec] | None = None,
) -> Path:
    path = tmp_path / phase / "phase.json"
    write_phase(
        path,
        PhaseManifest(
            1,
            phase,
            "eval/heldout/loss",
            None,
            runs or [],
            [],
            selected,
        ),
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


def test_generation_rejects_nonproduction_weight_decay(base_config: Path, tmp_path: Path) -> None:
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


def test_generation_rejects_nonintegral_accumulation(base_config: Path, tmp_path: Path) -> None:
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
        (lr, output_mult) for lr in (0.0063, 0.01, 0.016) for output_mult in (0.5, 1.0, 2.0)
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


def test_later_phase_default_cell_counts() -> None:
    candidates = [
        {"lr": 0.01, "output_mult": 1.0},
        {"lr": 0.016, "output_mult": 0.5},
    ]
    source_runs = [
        RunSpec(
            id=f"refine-{index}",
            config=f"runs/refine-{index}/run.yaml",
            result=f"runs/refine-{index}/result.json",
            params={
                "preset": "170M",
                **candidate,
                "depth_exponent": 0.0,
                "seed": 42,
                "global_examples": 2048,
                "max_steps": 20000,
                "warmup_steps": 5000,
            },
        )
        for index, candidate in enumerate(candidates)
    ]

    assert len(mup_sweep._replicate_cells(source_runs, candidates, [42, 43, 44])) == 4
    assert (
        len(
            mup_sweep._transfer_cells(
                candidates,
                presets=["400M", "800M", "1B"],
                steps=[10000, 20000, 10000],
                exponents=[0.0, 0.25, 0.5],
                global_examples=2048,
                seed=42,
                warmup=5000,
            )
        )
        == 18
    )

    transferred = [{"lr": 0.01, "output_mult": 1.0, "depth_exponent": 0.25}]
    bridge = mup_sweep._bridge_cells(
        transferred,
        multipliers=[0.7, 1.0, 1.4, 2.0],
        global_examples=8192,
        seed=42,
        steps=10000,
        warmup=5000,
    )
    assert len(bridge) == 4
    assert [cell["lr"] for cell in bridge] == pytest.approx([0.007, 0.01, 0.014, 0.02])

    bridge_finalists = [
        {"lr": 0.01, "output_mult": 1.0, "depth_exponent": 0.25, "batch_mult": 1.0},
        {"lr": 0.014, "output_mult": 1.0, "depth_exponent": 0.25, "batch_mult": 1.4},
    ]
    assert (
        len(
            mup_sweep._confirm_cells(
                bridge_finalists,
                global_examples=8192,
                seed=42,
                steps=10000,
                warmup=5000,
            )
        )
        == 2
    )
    assert (
        len(
            mup_sweep._scale_cells(
                bridge_finalists[:1],
                presets=["50M", "170M", "400M", "800M", "1B"],
                global_examples=8192,
                seed=42,
                steps=100000,
                warmup=5000,
            )
        )
        == 5
    )


def test_replicate_generates_only_new_seeds_from_source_runs(
    base_config: Path, tmp_path: Path
) -> None:
    candidates = [
        {"lr": 0.01, "output_mult": 1.0},
        {"lr": 0.016, "output_mult": 0.5},
    ]
    source_runs = [
        RunSpec(
            id=f"refine-{index}",
            config=f"runs/refine-{index}/run.yaml",
            result=f"runs/refine-{index}/result.json",
            params={
                "preset": "170M",
                **candidate,
                "depth_exponent": 0.0,
                "seed": 42,
                "global_examples": 2048,
                "max_steps": 20000,
                "warmup_steps": 5000,
            },
        )
        for index, candidate in enumerate(candidates)
    ]
    source = _write_selected_phase(tmp_path, "refine", candidates, source_runs)
    out = tmp_path / "replicate"

    result = runner.invoke(
        mup_sweep.app,
        [
            "replicate",
            "--config",
            str(base_config),
            "--from",
            str(source),
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    phase = load_phase(out / "phase.json")
    assert phase.source == "../refine/phase.json"
    assert [
        (run.params["lr"], run.params["output_mult"], run.params["seed"]) for run in phase.runs
    ] == [
        (0.01, 1.0, 43),
        (0.01, 1.0, 44),
        (0.016, 0.5, 43),
        (0.016, 0.5, 44),
    ]
    assert {
        (
            run.params["preset"],
            run.params["depth_exponent"],
            run.params["global_examples"],
            run.params["max_steps"],
            run.params["warmup_steps"],
        )
        for run in phase.runs
    } == {("170M", 0.0, 2048, 20000, 5000)}


def test_transfer_generates_paired_candidates_across_default_models(
    base_config: Path, tmp_path: Path
) -> None:
    candidates = [
        {"lr": 0.01, "output_mult": 1.0},
        {"lr": 0.016, "output_mult": 0.5},
    ]
    source = _write_selected_phase(tmp_path, "replicate", candidates)
    out = tmp_path / "transfer"

    result = runner.invoke(
        mup_sweep.app,
        [
            "transfer",
            "--config",
            str(base_config),
            "--from",
            str(source),
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    params = [run.params for run in load_phase(out / "phase.json").runs]
    assert len(params) == 18
    assert {(cell["lr"], cell["output_mult"]) for cell in params} == {
        (0.01, 1.0),
        (0.016, 0.5),
    }
    assert {(cell["preset"], cell["max_steps"]) for cell in params} == {
        ("400M", 10000),
        ("800M", 20000),
        ("1B", 10000),
    }
    assert {cell["depth_exponent"] for cell in params} == {0.0, 0.25, 0.5}
    assert {(cell["global_examples"], cell["seed"], cell["warmup_steps"]) for cell in params} == {
        (2048, 42, 5000)
    }


def test_bridge_uses_only_explicit_top_transfer_candidate(
    base_config: Path, tmp_path: Path
) -> None:
    source = _write_selected_phase(
        tmp_path,
        "transfer",
        [
            {"lr": 0.01, "output_mult": 1.0, "depth_exponent": 0.25},
            {"lr": 0.02, "output_mult": 0.5, "depth_exponent": 0.5},
        ],
    )
    out = tmp_path / "bridge"

    result = runner.invoke(
        mup_sweep.app,
        [
            "bridge",
            "--config",
            str(base_config),
            "--from",
            str(source),
            "--out",
            str(out),
            "--candidates",
            "0.02:0.5:0.5",
        ],
    )

    assert result.exit_code == 0, result.output
    params = [run.params for run in load_phase(out / "phase.json").runs]
    assert [cell["lr"] for cell in params] == pytest.approx([0.014, 0.02, 0.028, 0.04])
    assert [cell["batch_mult"] for cell in params] == [0.7, 1.0, 1.4, 2.0]
    assert {
        (
            cell["preset"],
            cell["output_mult"],
            cell["depth_exponent"],
            cell["global_examples"],
            cell["seed"],
            cell["max_steps"],
            cell["warmup_steps"],
        )
        for cell in params
    } == {("170M", 0.5, 0.5, 8192, 42, 10000, 5000)}


def test_confirm_preserves_bridge_depth_and_batch_corrections(
    base_config: Path, tmp_path: Path
) -> None:
    finalists = [
        {"lr": 0.01, "output_mult": 1.0, "depth_exponent": 0.25, "batch_mult": 1.0},
        {"lr": 0.014, "output_mult": 0.5, "depth_exponent": 0.5, "batch_mult": 1.4},
    ]
    source = _write_selected_phase(tmp_path, "bridge-replicate", finalists)
    out = tmp_path / "confirm"

    result = runner.invoke(
        mup_sweep.app,
        [
            "confirm",
            "--config",
            str(base_config),
            "--from",
            str(source),
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    params = [run.params for run in load_phase(out / "phase.json").runs]
    assert len(params) == 2
    assert [
        (
            cell["lr"],
            cell["output_mult"],
            cell["depth_exponent"],
            cell["batch_mult"],
        )
        for cell in params
    ] == [
        (0.01, 1.0, 0.25, 1.0),
        (0.014, 0.5, 0.5, 1.4),
    ]
    assert {
        (
            cell["preset"],
            cell["global_examples"],
            cell["seed"],
            cell["max_steps"],
            cell["warmup_steps"],
        )
        for cell in params
    } == {("800M", 8192, 42, 10000, 5000)}


def test_scale_uses_only_confirmed_winner_across_default_presets(
    base_config: Path, tmp_path: Path
) -> None:
    source = _write_selected_phase(
        tmp_path,
        "confirm",
        [
            {"lr": 0.014, "output_mult": 0.5, "depth_exponent": 0.5, "batch_mult": 1.4},
            {"lr": 0.01, "output_mult": 1.0, "depth_exponent": 0.25, "batch_mult": 1.0},
        ],
    )
    out = tmp_path / "scale"

    result = runner.invoke(
        mup_sweep.app,
        [
            "scale",
            "--config",
            str(base_config),
            "--from",
            str(source),
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    params = [run.params for run in load_phase(out / "phase.json").runs]
    assert [cell["preset"] for cell in params] == ["50M", "170M", "400M", "800M", "1B"]
    assert {
        (
            cell["lr"],
            cell["output_mult"],
            cell["depth_exponent"],
            cell["batch_mult"],
            cell["global_examples"],
            cell["seed"],
            cell["max_steps"],
            cell["warmup_steps"],
        )
        for cell in params
    } == {(0.014, 0.5, 0.5, 1.4, 8192, 42, 100000, 5000)}


@pytest.mark.parametrize("command", ["replicate", "transfer", "bridge", "confirm", "scale"])
def test_later_phase_commands_require_selected_candidates(
    base_config: Path, tmp_path: Path, command: str
) -> None:
    source = _write_selected_phase(tmp_path, f"empty-{command}", [])
    result = runner.invoke(
        mup_sweep.app,
        [
            command,
            "--config",
            str(base_config),
            "--from",
            str(source),
            "--out",
            str(tmp_path / f"out-{command}"),
        ],
    )
    assert result.exit_code != 0
    assert "has no selected candidates" in result.output


def test_later_phase_list_validation(base_config: Path, tmp_path: Path) -> None:
    source = _write_selected_phase(
        tmp_path,
        "selected",
        [{"lr": 0.01, "output_mult": 1.0}],
        [
            RunSpec(
                "source",
                "runs/source/run.yaml",
                "runs/source/result.json",
                {
                    "preset": "170M",
                    "lr": 0.01,
                    "output_mult": 1.0,
                    "depth_exponent": 0.0,
                    "seed": 42,
                    "global_examples": 2048,
                    "max_steps": 20000,
                    "warmup_steps": 5000,
                },
            )
        ],
    )
    replicate = runner.invoke(
        mup_sweep.app,
        [
            "replicate",
            "--config",
            str(base_config),
            "--from",
            str(source),
            "--out",
            str(tmp_path / "replicate-invalid"),
            "--seeds",
            "43,44",
        ],
    )
    assert replicate.exit_code != 0
    assert "must include source seed 42" in replicate.output

    transfer = runner.invoke(
        mup_sweep.app,
        [
            "transfer",
            "--config",
            str(base_config),
            "--from",
            str(source),
            "--out",
            str(tmp_path / "transfer-invalid"),
            "--presets",
            "400M,800M",
            "--steps",
            "10000",
        ],
    )
    assert transfer.exit_code != 0
    assert "same number of values" in transfer.output

    with pytest.raises(typer.BadParameter, match="must list at least one value"):
        mup_sweep._parse_ints(",", name="--seeds")
    with pytest.raises(typer.BadParameter, match="must list at least one value"):
        mup_sweep._parse_strings(",", name="--presets")
