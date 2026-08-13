from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from oplm.sweep import phases
from oplm.sweep.common import load_phase
from tests.slurm.test_submit import _install_stub
from tests.sweep.conftest import _write_base_config

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

runner = CliRunner()


def test_single_preset_phase_emits_one_array(coarse_phase: Path) -> None:
    jobs = coarse_phase.parent / "jobs"
    assert (jobs / "170M.sbatch").exists()
    assert (jobs / "170M.jobs").exists()
    assert (jobs / "analyze.sbatch").exists()
    assert (jobs / "submit.sh").exists()
    cells = (jobs / "170M.jobs").read_text().splitlines()
    assert len(cells) == 7
    assert "#SBATCH --array=0-6%4" in (jobs / "170M.sbatch").read_text()


def test_multi_preset_phase_emits_one_array_per_preset(multi_preset_phase: Path) -> None:
    """Slurm arrays need homogeneous resources, and node count varies by preset."""
    jobs = multi_preset_phase.parent / "jobs"
    for preset, nodes in (("400M", 4), ("800M", 8), ("1B", 8)):
        text = (jobs / f"{preset}.sbatch").read_text()
        assert f"#SBATCH --nodes={nodes}" in text
    submit = (jobs / "submit.sh").read_text()
    assert "--dependency=afterany:$A_400M:$A_800M:$A_1B" in submit


def test_transfer_emits_single_170m_array_across_depths(transfer_phase: Path) -> None:
    """The depth ray keeps every cell on the 170M 4-node allocation: one array total."""
    jobs = transfer_phase.parent / "jobs"
    text = (jobs / "170M.sbatch").read_text()
    assert "#SBATCH --nodes=4" in text
    assert "#SBATCH --array=0-8%4" in text
    ids = (jobs / "170M.jobs").read_text().split()
    assert len(ids) == 9  # 1 candidate x 3 depths x 3 bracket multipliers
    assert {run_id.split("-")[1] for run_id in ids} == {"d12", "d24", "d48"}


def test_array_job_wires_run_dir_progress_for_the_requeue_wrapper(coarse_phase: Path) -> None:
    """Each array task's no-progress guard must key off its own `$RUN_DIR/output`."""
    text = (coarse_phase.parent / "jobs" / "170M.sbatch").read_text()
    assert 'STEP_FILE="$RUN_DIR/output/.last_requeue_step"' in text


def test_analyze_job_has_no_progress_guard(coarse_phase: Path) -> None:
    """The analyze job is not a training job -- no checkpoints, no no-progress guard."""
    text = (coarse_phase.parent / "jobs" / "analyze.sbatch").read_text()
    assert "STEP_FILE" not in text
    assert "PREV_STEP" not in text


def test_analyze_job_is_cpu_only(coarse_phase: Path) -> None:
    text = (coarse_phase.parent / "jobs" / "analyze.sbatch").read_text()
    assert "--gres" not in text
    assert "oplm sweep analyze" in text
    assert "#SBATCH --time=01:00:00" in text


def test_cells_file_order_matches_manifest(coarse_phase: Path) -> None:
    manifest = json.loads(coarse_phase.read_text())
    cells = (coarse_phase.parent / "jobs" / "170M.jobs").read_text().split()
    assert cells == [run["id"] for run in manifest["runs"]]


def test_manifest_records_provenance(coarse_phase: Path) -> None:
    from importlib.metadata import version

    manifest = json.loads(coarse_phase.read_text())
    assert manifest["oplm_version"] == version("oplm")
    assert manifest["generated_at"]


def test_generated_scripts_are_valid_bash(coarse_phase: Path) -> None:
    for script in sorted((coarse_phase.parent / "jobs").glob("*.sbatch")):
        subprocess.run(["bash", "-n", str(script)], check=True)
    subprocess.run(["bash", "-n", str(coarse_phase.parent / "jobs" / "submit.sh")], check=True)


def test_confirm_phase_emits_its_single_array(confirm_phase: Path) -> None:
    """`confirm_phase` (800M, one candidate) exercises the single-preset path a second way."""
    jobs = confirm_phase.parent / "jobs"
    text = (jobs / "800M.sbatch").read_text()
    assert "#SBATCH --nodes=8" in text
    assert "#SBATCH --array=0-0%4" in text


def test_submit_flag_submits_all_and_records_job_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--submit` submits every entry via the fake `sbatch` on PATH and records the ids."""
    log = tmp_path / "sbatch.log"
    _install_stub(
        tmp_path,
        monkeypatch,
        "sbatch",
        f'#!/bin/bash\necho "$@" >> "{log}"\ncount=$(wc -l < "{log}")\necho $((900000 + count))\n',
    )
    config = _write_base_config(tmp_path)
    out = tmp_path / "coarse"
    result = runner.invoke(
        phases.app,
        ["coarse", "--config", str(config), "--out", str(out), "--submit"],
    )
    assert result.exit_code == 0, result.output
    manifest = load_phase(out / "phase.json")
    assert manifest.job_ids is not None
    assert set(manifest.job_ids) == {"A_170M", "ANALYZE"}
    assert log.exists()


def test_local_and_submit_are_mutually_exclusive(tmp_path: Path) -> None:
    config = _write_base_config(tmp_path)
    result = runner.invoke(
        phases.app,
        [
            "coarse",
            "--config",
            str(config),
            "--out",
            str(tmp_path / "coarse"),
            "--local",
            "--submit",
        ],
    )
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


def test_scale_command_has_no_submit_flag(tmp_path: Path) -> None:
    """`scale` generates only -- it has no `--submit` (or `--local`) flag at all."""
    result = runner.invoke(phases.app, ["scale", "--help"])
    assert result.exit_code == 0
    assert "--submit" not in result.output
    assert "--local" not in result.output
