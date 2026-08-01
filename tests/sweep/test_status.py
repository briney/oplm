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
- Submitted-ness must also be resolved *per preset*, not phase-wide: a preset that was never
  submitted is "missing" -- and gets resubmit guidance -- even when a sibling preset in the same
  phase was submitted
  (`test_status_partial_submission_missing_not_unknown_for_unsubmitted_preset`).
- A preset withheld from resubmit guidance (every open cell "running" or "unknown") must be
  called out explicitly, not silently dropped
  (`test_status_flags_running_preset_as_skipped_not_silently_omitted`,
  `test_status_flags_unknown_preset_as_skipped_not_silently_omitted`).
- A `squeue` that exists on PATH but whose controller is wedged (times out) must be treated the
  same as an absent `squeue` -- `unknown` cells, no resubmit guidance -- never as "missing"
  (`test_status_reports_unknown_not_missing_when_squeue_times_out`).
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from oplm.cli import app
from oplm.slurm.submit import running_job_ids as _real_running_job_ids
from oplm.sweep import phases
from oplm.sweep.common import load_phase, write_phase
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


def test_status_partial_submission_missing_not_unknown_for_unsubmitted_preset(
    transfer_phase: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A never-submitted preset must be 'missing' -- and get resubmit guidance -- even when a
    sibling preset in the same phase was submitted.

    Regression guard for Finding 1: `_cell_state` used to receive a phase-wide `submitted =
    bool(job_ids)` flag. Once *any* preset had a job id, every other (never-submitted) preset's
    open cells read as `unknown` instead of `missing`, and `unknown` is excluded from the
    resubmit line -- so the never-submitted preset silently vanished from resubmit guidance
    entirely. This manifest mirrors the finding's live repro: `job_ids` names 400M (and the
    downstream analyze job) but not 800M/1B, with `squeue` off PATH.
    """
    manifest = load_phase(transfer_phase)
    manifest.job_ids = {"A_400M": "900001", "ANALYZE": "900099"}
    write_phase(transfer_phase, manifest)

    monkeypatch.setenv("PATH", "")  # squeue unreachable
    result = runner.invoke(app, ["sweep", "status", str(transfer_phase)])
    assert result.exit_code == 0, result.output

    # 400M really was submitted: with the scheduler unreachable its cells are genuinely
    # ambiguous, so `unknown` (and no resubmit line for it) is correct.
    assert "unknown" in result.stdout
    assert "resubmit 400M" not in result.stdout
    # 800M/1B were never submitted at all: unambiguously missing, and must still be offered for
    # resubmission -- this is the line the bug made disappear.
    assert "resubmit 800M: sbatch --array=0,1,2,3 jobs/800M.sbatch" in result.stdout
    assert "resubmit 1B: sbatch --array=0,1,2,3 jobs/1B.sbatch" in result.stdout


def test_status_flags_running_preset_as_skipped_not_silently_omitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A preset withheld from resubmit guidance because every open cell is 'running' must say so.

    Regression guard for Finding 2: omitting the preset's resubmit line with no other comment
    reads as "nothing needed here", indistinguishable from "complete" to an operator skimming
    the output.
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
    _install_stub(tmp_path, monkeypatch, "squeue", f"#!/bin/bash\nprintf '{live_id}\\n'\n")

    status_result = runner.invoke(app, ["sweep", "status", str(phase_path)])
    assert status_result.exit_code == 0, status_result.output
    assert "resubmit 400M" not in status_result.stdout
    assert "skip 400M" in status_result.stdout
    assert "still running" in status_result.stdout


def test_status_flags_unknown_preset_as_skipped_not_silently_omitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A preset withheld because every open cell is 'unknown' must say so, and distinguish the
    reason (scheduler unreachable) from the 'running' case above."""
    config = _write_base_config(tmp_path)
    _install_fake_sbatch(tmp_path, monkeypatch)
    out = tmp_path / "coarse"
    result = runner.invoke(
        phases.app,
        ["coarse", "--config", str(config), "--out", str(out), "--submit"],
    )
    assert result.exit_code == 0, result.output
    phase_path = out / "phase.json"

    monkeypatch.setenv("PATH", "")  # squeue unreachable; no results on disk at all
    status_result = runner.invoke(app, ["sweep", "status", str(phase_path)])
    assert status_result.exit_code == 0, status_result.output
    assert "resubmit 170M" not in status_result.stdout
    assert "skip 170M" in status_result.stdout
    assert "scheduler unreachable" in status_result.stdout


def test_status_reports_unknown_not_missing_when_squeue_times_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `squeue` that exists on PATH but whose controller is wedged must degrade to `unknown`
    cells with no resubmit guidance, exactly like an absent `squeue` -- never a confident
    `missing` that invites a duplicate submission of a job that may still be queued or training.

    Regression guard for Critical 1: `status` used to derive `scheduler_reachable` from
    `shutil.which("squeue") is not None`, which stays `True` even when `squeue` itself hangs
    against a wedged controller; `running_job_ids`'s own timeout was silently folded into an
    ordinary empty result, so every unresolved (but genuinely submitted) cell read as `missing`
    plus an actionable `resubmit ...: sbatch --array=...` line. `phases.running_job_ids` is
    monkeypatched to inject a short timeout (real `squeue` queries default to 30s) so the test
    doesn't have to wait out the wedged stub's full duration.
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
    # Every preset (400M/800M/1B) got submitted; none has any result on disk.
    assert {"A_400M", "A_800M", "A_1B"} <= manifest.job_ids.keys()

    # squeue exists on PATH -- shutil.which("squeue") would find it -- but its controller never
    # answers.
    _install_stub(tmp_path, monkeypatch, "squeue", "#!/bin/bash\nsleep 2\n")
    monkeypatch.setattr(
        phases, "running_job_ids", lambda ids: _real_running_job_ids(ids, timeout=0.2)
    )

    started = time.monotonic()
    status_result = runner.invoke(app, ["sweep", "status", str(phase_path)])
    elapsed = time.monotonic() - started
    assert status_result.exit_code == 0, status_result.output
    assert elapsed < 2, f"status blocked for {elapsed:.2f}s past the injected 0.2s timeout"

    assert "unknown" in status_result.stdout
    # The critical assertion: a wedged squeue must never demote a submitted cell to "missing" --
    # that is exactly the confident-but-wrong signal that invites a duplicate sbatch submission.
    assert "missing" not in status_result.stdout
    for preset in ("400M", "800M", "1B"):
        assert f"resubmit {preset}" not in status_result.stdout
