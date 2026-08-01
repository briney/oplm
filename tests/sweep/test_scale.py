"""`oplm sweep scale`: a generate-only caller over `oplm.slurm`.

Unlike every other phase, `scale` runs no proxy cells and owns no ranking -- it reads the
confirmed winner from a `confirm` phase manifest, merges it into an ordinary training config
(`configs/scaling.yaml`), and renders one job script per preset. These tests run against the real
packaged `configs/scaling.yaml` (via `--config`) so a regression in that file is caught here too.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from oplm.cli import app
from oplm.config import load_config
from oplm.sweep.common import PhaseManifest, write_phase

runner = CliRunner()
SCALING = str(Path(__file__).resolve().parents[2] / "configs" / "scaling.yaml")


def _run_scale(confirm_phase: Path, out: Path, presets: str) -> None:
    result = runner.invoke(
        app,
        [
            "sweep",
            "scale",
            "--from",
            str(confirm_phase),
            "--config",
            SCALING,
            "--presets",
            presets,
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.stdout


def test_scale_writes_one_script_per_preset(confirm_phase: Path, tmp_path: Path) -> None:
    out = tmp_path / "scale"
    _run_scale(confirm_phase, out, "170M,400M")
    for preset, nodes in (("170M", 4), ("400M", 8)):
        script = out / "jobs" / f"oplm-{preset}-scale.sbatch"
        assert script.exists()
        text = script.read_text()
        assert f"#SBATCH --nodes={nodes}" in text
        # Not an array: one long production run per preset.
        assert "--array" not in text


def test_scale_writes_no_jobs_array_or_analyze_artifacts(
    confirm_phase: Path, tmp_path: Path
) -> None:
    """`scale` must emit per-preset scripts and nothing else.

    Every other phase writes a per-preset `.jobs` index, an `analyze.sbatch`, and a `submit.sh`
    (see `_write_jobs`); `scale`'s production runs are reviewed and submitted by hand, so none of
    that machinery -- nor the proxy-cell `runs/` tree -- belongs in its output directory.
    """
    out = tmp_path / "scale"
    _run_scale(confirm_phase, out, "170M,400M")
    jobs = out / "jobs"
    assert not (jobs / "analyze.sbatch").exists()
    assert not (jobs / "submit.sh").exists()
    assert not (out / "runs").exists()
    for preset in ("170M", "400M"):
        assert not (jobs / f"{preset}.jobs").exists()


def test_scale_carries_the_confirmed_winner(confirm_phase: Path, tmp_path: Path) -> None:
    out = tmp_path / "scale"
    _run_scale(confirm_phase, out, "170M")
    cfg = load_config(["--config", str(out / "170M" / "run.yaml")])
    winner = json.loads(confirm_phase.read_text())["selected"][0]
    assert cfg.train.lr == winner["lr"]
    assert cfg.train.mup_depth_lr_exponent == winner["depth_exponent"]
    assert cfg.model.mup_output_mult == winner["output_mult"]


def test_scale_keeps_the_full_eval_suite(confirm_phase: Path, tmp_path: Path) -> None:
    """Sweep cells run one eval; scale runs the production suite."""
    out = tmp_path / "scale"
    _run_scale(confirm_phase, out, "170M")
    cfg = load_config(["--config", str(out / "170M" / "run.yaml")])
    assert set(cfg.data.eval) >= {"uniref70", "omg70", "deepclust30", "casp14"}


def test_scale_has_no_submit_flag(confirm_phase: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "sweep",
            "scale",
            "--from",
            str(confirm_phase),
            "--config",
            SCALING,
            "--presets",
            "170M",
            "--out",
            str(tmp_path / "scale"),
            "--submit",
        ],
    )
    assert result.exit_code != 0


def test_scale_default_presets_exclude_50m_against_real_scaling_config(
    confirm_phase: Path, tmp_path: Path
) -> None:
    """The packaged default must not crash against the real production config.

    `configs/scaling.yaml` defines `nodes`/`max_batch_size` only for 170M/400M/800M/1B; the
    default `--presets` used to include `50M` and raised. Exercising the default (no `--presets`
    passed) against the real config is the regression test for that crash.
    """
    out = tmp_path / "scale"
    result = runner.invoke(
        app,
        ["sweep", "scale", "--from", str(confirm_phase), "--config", SCALING, "--out", str(out)],
    )
    assert result.exit_code == 0, result.stdout
    for preset in ("170M", "400M", "800M", "1B"):
        assert (out / "jobs" / f"oplm-{preset}-scale.sbatch").exists()


def test_scale_explicit_missing_preset_is_a_clean_error_not_a_keyerror(
    confirm_phase: Path, tmp_path: Path
) -> None:
    """Explicitly requesting a preset outside the slurm tables must not surface a raw KeyError."""
    result = runner.invoke(
        app,
        [
            "sweep",
            "scale",
            "--from",
            str(confirm_phase),
            "--config",
            SCALING,
            "--presets",
            "50M",
            "--out",
            str(tmp_path / "scale"),
        ],
    )
    assert result.exit_code != 0
    assert not isinstance(result.exception, KeyError)
    # BadParameter's message goes to stderr; `.output` is the merged stdout+stderr stream.
    assert "50M" in result.output


def test_scale_requires_a_selected_winner(tmp_path: Path) -> None:
    empty = tmp_path / "confirm" / "phase.json"
    write_phase(empty, PhaseManifest(1, "confirm", "eval/heldout/loss", None, [], [], []))

    result = runner.invoke(
        app,
        [
            "sweep",
            "scale",
            "--from",
            str(empty),
            "--config",
            SCALING,
            "--presets",
            "170M",
            "--out",
            str(tmp_path / "scale"),
        ],
    )
    assert result.exit_code != 0
    assert "no selected winner" in result.output
