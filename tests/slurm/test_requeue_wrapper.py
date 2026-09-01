"""Bash-level tests for the requeue wrapper `render_job` appends after the training `srun`.

The wrapper cannot be exercised end-to-end (no real `srun`/`scontrol` in CI), so these tests
extract the wrapper's text (everything from `STATUS=$?` onward) out of a fully rendered script and
execute it directly under `bash`, with a synthetic exit-status preamble standing in for the
training command and a stub `scontrol` on PATH recording its invocation.
"""

from __future__ import annotations

import os
import subprocess
from typing import TYPE_CHECKING

import pytest

from oplm.slurm.config import SlurmConfig
from oplm.slurm.render import JobSpec, render_job
from tests.slurm.test_config import RAW
from tests.slurm.test_submit import _install_stub

if TYPE_CHECKING:
    from pathlib import Path

SLURM = SlurmConfig.from_mapping({**RAW, "max_requeues": 20})


def _wrapper_text(*, progress_dir: str | None) -> str:
    """The wrapper text (from `STATUS=$?` to end of script) for a single-job spec."""
    spec = JobSpec(
        name="oplm-train",
        nodes=1,
        time_limit="168:00:00",
        command="python -m oplm.train --config cfg.yaml",
        progress_dir=progress_dir,
    )
    text = render_job(spec, SLURM)
    return text[text.index("STATUS=$?") :]


def _run_wrapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    preamble: str,
    progress_dir: str | None,
    job_id: str = "424242",
    restarts: int | None = None,
    scontrol_log: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Render the wrapper, prepend `preamble` (sets `$?` for `STATUS=$?`), and run it.

    Runs under exactly the production shell options: the real script has `set -euo pipefail`
    active from the top, then `set +e` right before the srun (so a nonzero training exit
    doesn't immediately kill the script) -- `set -e`/`set +e` toggle `errexit` only, they never
    touch `pipefail`, which stays on the whole time. A harness that dropped `set -euo pipefail`
    (as an earlier version of this file did) cannot catch pipefail-only bugs in the wrapper's
    command substitutions -- see `CURRENT_STEP=$(... || true)` in `render.py`.

    Args:
        preamble: A shell snippet run immediately before the wrapper (under `set +e`, matching
            where the real srun runs), whose exit status the wrapper's `STATUS=$?` captures
            (e.g. `"true"`, `"false"`, `"(exit 85)"`).
        progress_dir: Passed straight through to the rendered `JobSpec`.
        restarts: Value for `SLURM_RESTART_COUNT`; omitted (unset) when `None`.
        scontrol_log: Where the stub `scontrol` records its argv; defaults to a fresh file.
    """
    if scontrol_log is None:
        scontrol_log = tmp_path / "scontrol.log"
    _install_stub(
        tmp_path,
        monkeypatch,
        "scontrol",
        f'#!/bin/bash\necho "$@" >> "{scontrol_log}"\n',
    )
    script = tmp_path / "wrapper.sh"
    script.write_text(
        f"set -euo pipefail\nset +e\n{preamble}\n{_wrapper_text(progress_dir=progress_dir)}\n"
    )
    env = dict(os.environ)
    env["SLURM_JOB_ID"] = job_id
    if restarts is not None:
        env["SLURM_RESTART_COUNT"] = str(restarts)
    else:
        env.pop("SLURM_RESTART_COUNT", None)
    return subprocess.run(
        ["bash", str(script)], capture_output=True, text=True, env=env, check=False
    )


def _make_checkpoint(progress_dir: Path, step: int, *, suffix: str = "") -> None:
    (progress_dir / f"checkpoint-{step}{suffix}").mkdir(parents=True)


# --- case 1: exit 0 -> no requeue --------------------------------------------------------


def test_exit_zero_reports_complete_and_does_not_requeue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "scontrol.log"
    result = _run_wrapper(
        tmp_path, monkeypatch, preamble="true", progress_dir=None, scontrol_log=log
    )
    assert result.returncode == 0
    assert "training complete" in result.stdout
    assert not log.exists() or log.read_text() == ""


# --- case 2: exit 85 -> requeue even at high restart count below budget -----------------


def test_drain_exit_at_budget_cap_does_not_requeue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exit 85 bypasses the no-progress guard, but never the budget cap itself."""
    log = tmp_path / "scontrol.log"
    result = _run_wrapper(
        tmp_path,
        monkeypatch,
        preamble="(exit 85)",
        progress_dir=None,
        restarts=20,
        scontrol_log=log,
    )
    assert result.returncode == 85
    assert "requeue budget (20) exhausted" in result.stderr
    assert not log.exists() or log.read_text() == ""


def test_drain_exit_requeues_even_with_stale_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    progress_dir = tmp_path / "output"
    progress_dir.mkdir()
    _make_checkpoint(progress_dir, 100)
    (progress_dir / ".last_requeue_step").write_text("100")
    log = tmp_path / "scontrol.log"

    result = _run_wrapper(
        tmp_path,
        monkeypatch,
        preamble="(exit 85)",
        progress_dir=str(progress_dir),
        restarts=15,
        job_id="555",
        scontrol_log=log,
    )
    assert "requeueing" in result.stdout
    assert log.read_text().strip() == "requeue 555"


# --- case 2b: the production drain shape -- exit flattened to 1, marker present ---------


def test_drain_marker_requeues_despite_flattened_exit_and_stale_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `.drained` marker makes exit 1 a drain: guard bypassed, marker consumed.

    This is what a real multi-GPU drain looks like to the wrapper: the trainer ranks
    exit 85, but `accelerate launch` reports the worker failure as its own exit 1
    (torchelastic ChildFailedError), so `$STATUS` carries no drain information at all.
    The marker the trainer writes after the drain checkpoint commits must (a) requeue
    unconditionally even with a stale recorded step (the crash-loop guard must not fire),
    (b) leave the step file unwritten (the drain leg is fail-open, same as exit 85), and
    (c) be deleted, so a *later* genuine crash is not misread as a drain.
    """
    progress_dir = tmp_path / "output"
    progress_dir.mkdir()
    _make_checkpoint(progress_dir, 100)
    (progress_dir / ".last_requeue_step").write_text("100")
    (progress_dir / ".drained").write_text("100\n")
    log = tmp_path / "scontrol.log"

    drained = _run_wrapper(
        tmp_path,
        monkeypatch,
        preamble="false",
        progress_dir=str(progress_dir),
        restarts=1,
        scontrol_log=log,
    )
    assert "crash loop" not in drained.stderr
    assert "requeueing" in drained.stdout
    assert "drained=1" in drained.stdout
    assert log.read_text().strip() == "requeue 424242"
    # (b) the drain leg records nothing...
    assert (progress_dir / ".last_requeue_step").read_text().strip() == "100"
    # (c) ...and the marker is consumed.
    assert not (progress_dir / ".drained").exists()

    # With the marker gone, the same exit 1 at the same step is a genuine crash again:
    # the pre-seeded step file (100) now applies, so the no-progress guard trips.
    crashed = _run_wrapper(
        tmp_path,
        monkeypatch,
        preamble="false",
        progress_dir=str(progress_dir),
        restarts=2,
        scontrol_log=log,
    )
    assert "crash loop" in crashed.stderr
    assert log.read_text().splitlines() == ["requeue 424242"]


# --- case 3: exit 1 twice at the same step -> second invocation does NOT requeue --------


def test_crash_loop_same_step_stops_requeueing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    progress_dir = tmp_path / "output"
    progress_dir.mkdir()
    _make_checkpoint(progress_dir, 100)
    log = tmp_path / "scontrol.log"

    first = _run_wrapper(
        tmp_path,
        monkeypatch,
        preamble="false",
        progress_dir=str(progress_dir),
        restarts=0,
        scontrol_log=log,
    )
    assert "requeueing" in first.stdout
    assert log.read_text().strip() == "requeue 424242"

    second = _run_wrapper(
        tmp_path,
        monkeypatch,
        preamble="false",
        progress_dir=str(progress_dir),
        restarts=1,
        scontrol_log=log,
    )
    assert second.returncode != 0
    assert "crash loop" in second.stderr
    # The stub was not invoked a second time.
    assert log.read_text().strip() == "requeue 424242"


def test_drain_then_crash_at_same_step_still_requeues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A drain must not arm the crash-loop guard for the *first* crash that follows it.

    Regression for an Important review finding: the wrapper used to write `STEP_FILE` on
    every requeue path, including the drain (exit 85) path. So the sequence
    drain -> requeue -> crash before the next checkpoint left `CURRENT_STEP == PREV_STEP`
    and tripped the no-progress guard after a *single* non-drain failure -- ending the
    requeue loop of a healthy, merely-preempted job. The guard's contract is two
    consecutive *non-drain* failures with no progress between them, so the drain leg must
    not record a step at all (fail-open).
    """
    progress_dir = tmp_path / "output"
    progress_dir.mkdir()
    _make_checkpoint(progress_dir, 100)
    log = tmp_path / "scontrol.log"

    drained = _run_wrapper(
        tmp_path,
        monkeypatch,
        preamble="(exit 85)",
        progress_dir=str(progress_dir),
        restarts=0,
        scontrol_log=log,
    )
    assert "requeueing" in drained.stdout
    # The drain leg records nothing: there is no non-drain failure to compare against yet.
    assert not (progress_dir / ".last_requeue_step").exists()

    crashed = _run_wrapper(
        tmp_path,
        monkeypatch,
        preamble="false",
        progress_dir=str(progress_dir),
        restarts=1,
        scontrol_log=log,
    )
    assert "crash loop" not in crashed.stderr
    assert "requeueing" in crashed.stdout
    assert log.read_text().splitlines() == ["requeue 424242", "requeue 424242"]
    # ... and this first non-drain failure *does* record the step, so a second one at the
    # same step still trips the guard.
    assert (progress_dir / ".last_requeue_step").read_text().strip() == "100"

    second_crash = _run_wrapper(
        tmp_path,
        monkeypatch,
        preamble="false",
        progress_dir=str(progress_dir),
        restarts=2,
        scontrol_log=log,
    )
    assert "crash loop" in second_crash.stderr
    assert log.read_text().splitlines() == ["requeue 424242", "requeue 424242"]


# --- case 4: exit 1 with an advanced step -> requeues again ------------------------------


def test_progress_since_last_restart_keeps_requeueing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    progress_dir = tmp_path / "output"
    progress_dir.mkdir()
    _make_checkpoint(progress_dir, 100)
    log = tmp_path / "scontrol.log"

    first = _run_wrapper(
        tmp_path,
        monkeypatch,
        preamble="false",
        progress_dir=str(progress_dir),
        restarts=0,
        scontrol_log=log,
    )
    assert "requeueing" in first.stdout

    _make_checkpoint(progress_dir, 200)
    second = _run_wrapper(
        tmp_path,
        monkeypatch,
        preamble="false",
        progress_dir=str(progress_dir),
        restarts=1,
        scontrol_log=log,
    )
    assert "requeueing" in second.stdout
    assert log.read_text().splitlines() == ["requeue 424242", "requeue 424242"]
    assert (progress_dir / ".last_requeue_step").read_text().strip() == "200"


# --- case 5: budget exhausted -> no requeue ----------------------------------------------


def test_budget_exhausted_exits_without_requeueing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "scontrol.log"
    result = _run_wrapper(
        tmp_path,
        monkeypatch,
        preamble="false",
        progress_dir=None,
        restarts=20,
        scontrol_log=log,
    )
    assert result.returncode == 1
    assert "requeue budget (20) exhausted" in result.stderr
    assert not log.exists() or log.read_text() == ""


# --- regression: no checkpoint has ever been committed (a job's first restart cycle) ----


def test_empty_progress_dir_still_requeues(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The pre-checkpoint-1 case: `ls checkpoint-*` matches nothing at all.

    Under the script's top-level `set -o pipefail`, `ls`'s nonzero exit (no match) and
    `grep`'s nonzero exit (no numeric line) both propagate into the `CURRENT_STEP=$(...)`
    assignment; without a fix, `set -e` aborts the whole wrapper right there -- no stdout, no
    `STEP_FILE`, no `scontrol requeue` -- instead of falling through to `CURRENT_STEP=0`. This
    is the most common real case: every job hits it on its first restart, before checkpoint 1
    exists.
    """
    progress_dir = tmp_path / "output"
    progress_dir.mkdir()
    log = tmp_path / "scontrol.log"

    result = _run_wrapper(
        tmp_path,
        monkeypatch,
        preamble="false",
        progress_dir=str(progress_dir),
        restarts=0,
        scontrol_log=log,
    )
    assert result.returncode == 0
    assert "requeueing" in result.stdout
    assert "step=0" in result.stdout
    assert log.read_text().strip() == "requeue 424242"
    assert (progress_dir / ".last_requeue_step").read_text().strip() == "0"


# --- case 6 / committed-only property: .tmp and .old dirs are not committed steps -------


@pytest.mark.parametrize("suffix", [".tmp", ".old"])
def test_uncommitted_checkpoint_suffix_is_not_seen_as_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str
) -> None:
    """`checkpoint-500.tmp` / `checkpoint-500.old` must not count as a committed step 500."""
    progress_dir = tmp_path / "output"
    progress_dir.mkdir()
    _make_checkpoint(progress_dir, 500, suffix=suffix)
    log = tmp_path / "scontrol.log"

    result = _run_wrapper(
        tmp_path,
        monkeypatch,
        preamble="false",
        progress_dir=str(progress_dir),
        restarts=0,
        scontrol_log=log,
    )
    assert "step=0" in result.stdout
    assert (progress_dir / ".last_requeue_step").read_text().strip() == "0"


# --- progress_dir=None keeps the layer usable for non-training jobs ---------------------


def test_no_progress_dir_requeues_on_budget_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "scontrol.log"
    for restarts in (0, 1, 2):
        result = _run_wrapper(
            tmp_path,
            monkeypatch,
            preamble="false",
            progress_dir=None,
            restarts=restarts,
            scontrol_log=log,
        )
        assert "requeueing" in result.stdout
    assert log.read_text().splitlines() == ["requeue 424242"] * 3
