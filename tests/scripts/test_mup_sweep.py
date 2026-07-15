from __future__ import annotations

import json
import shlex
import subprocess
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


def _write_result(phase_dir: Path, run: RunSpec, loss: float) -> None:
    path = phase_dir / run.result
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"eval": {"eval/heldout/loss": loss}}))


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


@pytest.mark.parametrize(
    "metric",
    ["eval/heldout/accuracy", "eval/heldout/perplexity"],
)
def test_generation_rejects_explicit_non_loss_metric(
    base_config: Path, tmp_path: Path, metric: str
) -> None:
    result = runner.invoke(
        mup_sweep.app,
        [
            "smoke",
            "--config",
            str(base_config),
            "--out",
            str(tmp_path / "smoke"),
            "--metric",
            metric,
        ],
    )
    assert result.exit_code != 0
    assert "--metric must be eval/heldout/loss" in result.output


def test_generation_accepts_explicit_configured_loss_metric(
    base_config: Path, tmp_path: Path
) -> None:
    out = tmp_path / "smoke"
    result = runner.invoke(
        mup_sweep.app,
        [
            "smoke",
            "--config",
            str(base_config),
            "--out",
            str(out),
            "--metric",
            "eval/heldout/loss",
        ],
    )
    assert result.exit_code == 0, result.output
    assert load_phase(out / "phase.json").metric == "eval/heldout/loss"


def test_generation_resolves_relative_accelerate_config(
    base_config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    accelerate_config = tmp_path / "accelerate.yaml"
    accelerate_config.write_text("compute_environment: LOCAL_MACHINE\n")
    out = tmp_path / "smoke"
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        mup_sweep.app,
        [
            "smoke",
            "--config",
            str(base_config),
            "--out",
            str(out),
            "--accelerate-config",
            accelerate_config.name,
        ],
    )

    assert result.exit_code == 0, result.output
    command = shlex.split((out / "commands.txt").read_text().splitlines()[0])
    config_index = command.index("--config_file") + 1
    assert command[config_index] == str(accelerate_config.resolve())


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
                warmups=[1000, 2000, 1000],
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
    # 2 candidates × 3 presets × 4 depth exponents.
    assert len(params) == 24
    assert {(cell["lr"], cell["output_mult"]) for cell in params} == {
        (0.01, 1.0),
        (0.016, 0.5),
    }
    # Per-preset warmup is ~10% of each preset's horizon (400M/1B 10k → 1000; 800M 20k → 2000).
    assert {(cell["preset"], cell["max_steps"], cell["warmup_steps"]) for cell in params} == {
        ("400M", 10000, 1000),
        ("800M", 20000, 2000),
        ("1B", 10000, 1000),
    }
    assert {cell["depth_exponent"] for cell in params} == {0.0, 0.5, 0.75, 1.0}
    assert {(cell["global_examples"], cell["seed"]) for cell in params} == {(2048, 42)}


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
    } == {("170M", 0.5, 0.5, 8192, 42, 10000, 1000)}


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
    } == {("800M", 8192, 42, 10000, 1000)}


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
    # Normalize Rich's error-panel wrapping (border chars + line breaks) before matching.
    normalized = " ".join(transfer.output.replace("│", " ").split())
    assert "same number of values" in normalized

    with pytest.raises(typer.BadParameter, match="must list at least one value"):
        mup_sweep._parse_ints(",", name="--seeds")
    with pytest.raises(typer.BadParameter, match="must list at least one value"):
        mup_sweep._parse_strings(",", name="--presets")


def test_coarse_selects_interior_winner_and_neighbors(tmp_path: Path) -> None:
    lrs = [0.0025, 0.004, 0.0063, 0.01, 0.016, 0.025, 0.04]
    runs = [
        RunSpec(
            f"run-{index}",
            f"runs/run-{index}/run.yaml",
            f"runs/run-{index}/result.json",
            {"lr": lr, "output_mult": 1.0, "seed": 42},
        )
        for index, lr in enumerate(lrs)
    ]
    phase = PhaseManifest(1, "coarse", "eval/heldout/loss", None, runs, [], [])
    path = tmp_path / "phase.json"
    write_phase(path, phase)
    for run in runs:
        _write_result(tmp_path, run, abs(float(run.params["lr"]) - 0.01) + 1.0)

    mup_sweep.analyze_phase(path)

    analyzed = load_phase(path)
    assert [candidate["lr"] for candidate in analyzed.selected] == [0.0063, 0.01, 0.016]


def test_coarse_edge_winner_blocks_refinement(tmp_path: Path) -> None:
    runs = [
        RunSpec(
            f"run-{index}",
            f"runs/run-{index}/run.yaml",
            f"runs/run-{index}/result.json",
            {"lr": lr, "output_mult": 1.0, "seed": 42},
        )
        for index, lr in enumerate([0.0025, 0.004, 0.0063])
    ]
    path = tmp_path / "phase.json"
    write_phase(path, PhaseManifest(1, "coarse", "eval/heldout/loss", None, runs, [], []))
    for index, run in enumerate(runs):
        _write_result(tmp_path, run, 1.0 + index)
    mup_sweep.analyze_phase(path)
    assert load_phase(path).selected == []


def test_missing_and_nonfinite_results_are_ineligible(tmp_path: Path) -> None:
    runs = [
        RunSpec("ok", "runs/ok/run.yaml", "runs/ok/result.json", {"lr": 0.01}),
        RunSpec("missing", "runs/missing/run.yaml", "runs/missing/result.json", {"lr": 0.016}),
        RunSpec("nan", "runs/nan/run.yaml", "runs/nan/result.json", {"lr": 0.025}),
    ]
    path = tmp_path / "phase.json"
    write_phase(path, PhaseManifest(1, "confirm", "eval/heldout/loss", None, runs, [], []))
    _write_result(tmp_path, runs[0], 1.0)
    _write_result(tmp_path, runs[2], float("nan"))
    mup_sweep.analyze_phase(path)
    analyzed = load_phase(path)
    scores = {entry["id"]: entry["score"] for entry in analyzed.ranking}
    assert scores == {"ok": 1.0, "missing": None, "nan": None}
    assert analyzed.selected == [{"lr": 0.01}]


def _replicate_fixture(tmp_path: Path, *, losses: dict[float, list[float]]) -> tuple[Path, Path]:
    source_dir = tmp_path / "refine"
    replicate_dir = tmp_path / "replicate"
    source_dir.mkdir()
    replicate_dir.mkdir()
    candidates = [{"lr": lr, "output_mult": 1.0, "depth_exponent": 0.0} for lr in losses]
    source_runs = [
        RunSpec(
            f"lr-{lr:g}-seed-42",
            f"runs/lr-{lr:g}-seed-42/run.yaml",
            f"runs/lr-{lr:g}-seed-42/result.json",
            {**candidate, "seed": 42},
        )
        for lr, candidate in zip(losses, candidates, strict=True)
    ]
    replicate_runs = [
        RunSpec(
            f"lr-{lr:g}-seed-{seed}",
            f"runs/lr-{lr:g}-seed-{seed}/run.yaml",
            f"runs/lr-{lr:g}-seed-{seed}/result.json",
            {**candidate, "seed": seed},
        )
        for lr, candidate in zip(losses, candidates, strict=True)
        for seed in (43, 44)
    ]
    source_path = source_dir / "phase.json"
    replicate_path = replicate_dir / "phase.json"
    write_phase(
        source_path,
        PhaseManifest(1, "refine", "eval/heldout/loss", None, source_runs, [], candidates),
    )
    write_phase(
        replicate_path,
        PhaseManifest(
            1,
            "replicate",
            "eval/heldout/loss",
            "../refine/phase.json",
            replicate_runs,
            [],
            [],
        ),
    )
    for run in source_runs:
        lr = float(run.params["lr"])
        _write_result(source_dir, run, losses[lr][0])
    for run in replicate_runs:
        lr = float(run.params["lr"])
        seed_index = int(run.params["seed"]) - 42
        _write_result(replicate_dir, run, losses[lr][seed_index])
    return source_path, replicate_path


def _transfer_fixture(tmp_path: Path) -> Path:
    phase_dir = tmp_path / "transfer"
    phase_dir.mkdir()
    runs: list[RunSpec] = []
    for lr in (0.01, 0.016):
        for exponent in (0.0, 0.25, 0.5):
            for preset in ("400M", "800M", "1B"):
                run_id = f"{preset}-lr{lr:g}-a{exponent:g}"
                runs.append(
                    RunSpec(
                        run_id,
                        f"runs/{run_id}/run.yaml",
                        f"runs/{run_id}/result.json",
                        {
                            "preset": preset,
                            "lr": lr,
                            "output_mult": 1.0,
                            "depth_exponent": exponent,
                            "seed": 42,
                        },
                    )
                )
    path = phase_dir / "phase.json"
    write_phase(
        path,
        PhaseManifest(1, "transfer", "eval/heldout/loss", None, runs, [], []),
    )
    model_offset = {"400M": 0.0, "800M": 0.01, "1B": 0.02}
    for run in runs:
        lr = float(run.params["lr"])
        exponent = float(run.params["depth_exponent"])
        preset = str(run.params["preset"])
        if lr == 0.016 and exponent == 0.5 and preset == "1B":
            continue
        loss = 1.0 + model_offset[preset] + abs(exponent - 0.25) + (lr - 0.01) * 10
        _write_result(phase_dir, run, loss)
    return path


def test_replicate_ranks_three_seed_mean_and_reuses_source_seed(tmp_path: Path) -> None:
    # Source refine has seed 42 for two candidates; replicate has seeds 43 and 44.
    # Candidate 0.01 losses 1.0, 1.1, 1.2; candidate 0.016 losses 0.9, 1.5, 1.6.
    # Mean therefore selects 0.01 despite 0.016 winning seed 42.
    source_path, replicate_path = _replicate_fixture(
        tmp_path,
        losses={0.01: [1.0, 1.1, 1.2], 0.016: [0.9, 1.5, 1.6]},
    )
    assert source_path.exists()
    mup_sweep.analyze_phase(replicate_path)
    analyzed = load_phase(replicate_path)
    assert analyzed.selected[0]["lr"] == 0.01
    assert analyzed.ranking[0]["score"] == pytest.approx(1.1)


def test_transfer_sums_per_model_ranks_and_requires_all_models(tmp_path: Path) -> None:
    path = _transfer_fixture(tmp_path)
    mup_sweep.analyze_phase(path)
    analyzed = load_phase(path)
    assert analyzed.selected[0] == {
        "lr": 0.01,
        "output_mult": 1.0,
        "depth_exponent": 0.25,
    }
    incomplete = next(
        entry
        for entry in analyzed.ranking
        if entry["params"]["lr"] == 0.016 and entry["params"]["depth_exponent"] == 0.5
    )
    assert incomplete["score"] is None


def test_local_execution_is_sequential_and_stops_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[list[str]] = []

    def fake_run(argv: list[str], *, check: bool) -> None:
        seen.append(argv)
        if len(seen) == 2:
            raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(subprocess, "run", fake_run)
    commands = [["run", "one"], ["run", "two"], ["run", "three"]]
    with pytest.raises(subprocess.CalledProcessError):
        mup_sweep._run_local(commands)
    assert seen == commands[:2]
