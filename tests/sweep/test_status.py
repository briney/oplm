"""Tests for `oplm sweep status`.

`status` reports each cell's state (complete / non-finite / running / missing) and prints the
per-preset `sbatch --array=...` resubmit line for whatever is incomplete. The tricky parts are
covered explicitly:

- Per-preset, zero-based index numbering must match that preset's `.jobs` index file
  (`test_status_multi_preset_indices_match_jobs_files`).
- Whether a result-less cell is "running" or "missing" must be resolved from *that preset's own*
  array job id, not from whether any job anywhere in the phase happens to still be queued
  (`test_status_distinguishes_running_from_missing_per_preset`).
- `squeue` being unreachable must never be silently rendered as "missing" -- that would read as
  "safe to resubmit" for jobs that may still be running
  (`test_status_reports_unreachable_scheduler_explicitly`).
- A phase that was never submitted has no such ambiguity: "missing" regardless of whether
  `squeue` happens to be on PATH
  (`test_status_missing_state_independent_of_scheduler_when_never_submitted`).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from oplm.cli import app
from oplm.sweep import phases
from oplm.sweep.common import load_phase
from tests.slurm.test_submit import _install_stub
from tests.sweep.conftest import _write_base_config, _write_selected

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

runner = CliRunner()

# `sbatch` stub shared by every test that needs real job ids: logs its argv and returns a
# monotonically increasing id per call, exactly like `tests/slurm/test_submit.py::fake_sbatch`.
_FAKE_SBATCH = (
    '#!/bin/bash\necho "$@" >> "{log}"\ncount=$(wc -l < "{log}")\necho $((900000 + count))\n'
)


def _install_fake_sbatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    log = tmp_path / "sbatch.log"
    _install_stub(tmp_path, monkeypatch, "sbatch", _FAKE_SBATCH.format(log=log))
    return log


def test_status_reports_cell_states_and_resubmit_line(coarse_phase: Path) -> None:
    manifest = json.loads(coarse_phase.read_text())
    runs = manifest["runs"]
    # One complete, one non-finite, the rest missing.
    for index, value in ((0, 3.0), (1, float("nan"))):
        result_path = coarse_phase.parent / runs[index]["result"]
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps({"eval": {"eval/heldout/loss": value}}))

    result = runner.invoke(app, ["sweep", "status", str(coarse_phase)])
    assert result.exit_code == 0, result.output
    assert "complete" in result.stdout
    assert "non-finite" in result.stdout
    assert "missing" in result.stdout
    # Indices 1..6 need rerunning: the non-finite one and the five with no result.
    assert "--array=1,2,3,4,5,6" in result.stdout


def test_status_multi_preset_indices_match_jobs_files(transfer_phase: Path) -> None:
    """400M/800M/1B each get their own zero-based numbering, matching their `.jobs` file."""
    manifest = load_phase(transfer_phase)
    phase_dir = transfer_phase.parent
    runs_by_preset: dict[str, list] = {}
    for run in manifest.runs:
        runs_by_preset.setdefault(str(run.params["preset"]), []).append(run)
    assert set(runs_by_preset) == {"400M", "800M", "1B"}

    # Per preset: index 0 complete, index 1 non-finite, everything after that missing.
    for runs in runs_by_preset.values():
        for index, value in ((0, 2.0), (1, float("nan"))):
            result_path = phase_dir / runs[index].result
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(json.dumps({"eval": {"eval/heldout/loss": value}}))

    result = runner.invoke(app, ["sweep", "status", str(transfer_phase)])
    assert result.exit_code == 0, result.output
    assert "complete" in result.stdout
    assert "non-finite" in result.stdout
    assert "missing" in result.stdout

    for preset, runs in runs_by_preset.items():
        jobs_file_ids = (phase_dir / "jobs" / f"{preset}.jobs").read_text().split()
        # The `.jobs` file is ground truth for what `sbatch --array=` index means; confirm this
        # preset's runs line up with it in the same order before trusting derived indices below.
        assert jobs_file_ids == [run.id for run in runs]
        expected_indices = ",".join(str(i) for i in range(1, len(runs)))
        assert (
            f"resubmit {preset}: sbatch --array={expected_indices} jobs/{preset}.sbatch"
            in result.stdout
        )


def test_status_distinguishes_running_from_missing_per_preset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cell with no result is 'running' only if *that preset's* array job is still queued.

    Regression guard for treating `running_job_ids(...)` as a single phase-wide flag: if any one
    preset's array job were still live, every other preset's still-missing cells would wrongly
    read as 'running' too.
    """
    config = _write_base_config(tmp_path)
    source = _write_selected(
        tmp_path / "replicate" / "phase.json",
        "replicate",
        [{"lr": 0.01, "output_mult": 1.0}],
    )
    _install_fake_sbatch(tmp_path, monkeypatch)
    out = tmp_path / "transfer"
    result = runner.invoke(
        phases.app,
        ["transfer", "--config", str(config), "--from", str(source), "--out", str(out), "--submit"],
    )
    assert result.exit_code == 0, result.output
    phase_path = out / "phase.json"
    manifest = load_phase(phase_path)
    assert manifest.job_ids is not None
    live_id = manifest.job_ids["A_400M"]
    # squeue reports only 400M's array job as still queued; 800M/1B have "finished".
    _install_stub(tmp_path, monkeypatch, "squeue", f"#!/bin/bash\nprintf '{live_id}\\n'\n")

    status_result = runner.invoke(app, ["sweep", "status", str(phase_path)])
    assert status_result.exit_code == 0, status_result.output
    assert "running" in status_result.stdout
    # 400M's un-run cells are "running" and must not be offered for resubmission.
    assert "resubmit 400M" not in status_result.stdout
    assert "resubmit 800M: sbatch --array=0,1,2,3 jobs/800M.sbatch" in status_result.stdout
    assert "resubmit 1B: sbatch --array=0,1,2,3 jobs/1B.sbatch" in status_result.stdout


def test_status_reports_unreachable_scheduler_explicitly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """squeue being off PATH must read as 'cannot check', never as 'everything finished'."""
    config = _write_base_config(tmp_path)
    _install_fake_sbatch(tmp_path, monkeypatch)
    out = tmp_path / "coarse"
    result = runner.invoke(
        phases.app,
        ["coarse", "--config", str(config), "--out", str(out), "--submit"],
    )
    assert result.exit_code == 0, result.output
    phase_path = out / "phase.json"
    manifest = load_phase(phase_path)
    # index 0 complete, index 1 non-finite; indices 2..6 have no result at all.
    for index, value in ((0, 3.0), (1, float("nan"))):
        result_path = phase_path.parent / manifest.runs[index].result
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps({"eval": {"eval/heldout/loss": value}}))

    monkeypatch.setenv("PATH", "")  # squeue (and everything else) unreachable
    status_result = runner.invoke(app, ["sweep", "status", str(phase_path)])
    assert status_result.exit_code == 0, status_result.output
    assert "cannot query the scheduler" in status_result.stdout
    assert "unknown" in status_result.stdout
    # The critical assertion: never silently "missing" when the scheduler couldn't be asked.
    assert "missing" not in status_result.stdout
    # The non-finite cell's state never depended on the scheduler; still flagged for resubmit.
    assert "resubmit 170M: sbatch --array=1 jobs/170M.sbatch" in status_result.stdout


def test_status_missing_state_independent_of_scheduler_when_never_submitted(
    coarse_phase: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A phase that was never submitted is unambiguously 'missing', squeue or not."""
    monkeypatch.setenv("PATH", "")
    result = runner.invoke(app, ["sweep", "status", str(coarse_phase)])
    assert result.exit_code == 0, result.output
    assert "missing" in result.stdout
    assert "unknown" not in result.stdout
    assert "cannot query the scheduler" not in result.stdout
