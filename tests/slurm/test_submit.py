from __future__ import annotations

import logging
import os
import stat
import time
from pathlib import Path

import pytest

from oplm.slurm.render import SubmitEntry
from oplm.slurm.submit import running_job_ids, submit_all, submit_job


def _install_stub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, body: str) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    script = bin_dir / name
    script.write_text(body)
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")


@pytest.fixture
def fake_sbatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Put a recording `sbatch` stub on PATH; no cluster required."""
    log = tmp_path / "sbatch.log"
    _install_stub(
        tmp_path,
        monkeypatch,
        "sbatch",
        "#!/bin/bash\n"
        f'echo "$@" >> "{log}"\n'
        f'count=$(wc -l < "{log}")\n'
        "echo $((812344 + count))\n",
    )
    return log


def test_submit_job_returns_parsed_id(fake_sbatch: Path, tmp_path: Path) -> None:
    script = tmp_path / "job.sbatch"
    script.write_text("#!/bin/bash\n")
    assert submit_job(script) == "812345"
    assert fake_sbatch.read_text().strip() == f"--parsable {script}"


def test_submit_job_builds_afterany_dependency(fake_sbatch: Path, tmp_path: Path) -> None:
    script = tmp_path / "analyze.sbatch"
    script.write_text("#!/bin/bash\n")
    submit_job(script, depends_on=["100", "200", "300"])
    assert "--dependency=afterany:100:200:300" in fake_sbatch.read_text()


def test_submit_all_threads_ids_into_dependencies(fake_sbatch: Path, tmp_path: Path) -> None:
    for name in ("400M.sbatch", "800M.sbatch", "1B.sbatch", "analyze.sbatch"):
        (tmp_path / name).write_text("#!/bin/bash\n")
    ids = submit_all(
        [
            SubmitEntry(var="A_400M", script=Path("400M.sbatch")),
            SubmitEntry(var="A_800M", script=Path("800M.sbatch")),
            # Not a dependency of ANALYZE. Present so an implementation that threads every
            # prior id (instead of only the ones named in `entry.depends_on`) gets caught:
            # without this entry, ANALYZE's correct dependency set and "every id so far"
            # happen to coincide, and the test could not tell them apart.
            SubmitEntry(var="A_1B", script=Path("1B.sbatch")),
            SubmitEntry(
                var="ANALYZE", script=Path("analyze.sbatch"), depends_on=("A_400M", "A_800M")
            ),
        ],
        base_dir=tmp_path,
    )
    assert ids == {
        "A_400M": "812345",
        "A_800M": "812346",
        "A_1B": "812347",
        "ANALYZE": "812348",
    }
    lines = fake_sbatch.read_text().splitlines()
    # Exact match on the final invocation: catches wrong order, missing ids, and leakage of
    # A_1B's id, none of which a loose substring check on "afterany:812345:812346" could rule
    # out on its own (that substring is also a prefix of a wider, buggy dependency string).
    assert (
        lines[-1]
        == f"--parsable --dependency=afterany:812345:812346 {tmp_path / 'analyze.sbatch'}"
    )


def test_submit_job_raises_on_sbatch_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_stub(
        tmp_path, monkeypatch, "sbatch", '#!/bin/bash\necho "queue full" >&2\nexit 1\n'
    )
    job = tmp_path / "job.sbatch"
    job.write_text("#!/bin/bash\n")
    with pytest.raises(RuntimeError, match="queue full"):
        submit_job(job)


def test_submit_job_raises_on_empty_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """sbatch exiting 0 with no stdout (transient hiccup, swallowed output) must not yield ''."""
    _install_stub(tmp_path, monkeypatch, "sbatch", "#!/bin/bash\nexit 0\n")
    job = tmp_path / "job.sbatch"
    job.write_text("#!/bin/bash\n")
    with pytest.raises(RuntimeError, match=r"job\.sbatch"):
        submit_job(job)


def test_submit_job_raises_on_non_numeric_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """sbatch exiting 0 with non-numeric stdout (format change, wrapper text) must be caught."""
    _install_stub(
        tmp_path,
        monkeypatch,
        "sbatch",
        "#!/bin/bash\necho 'sbatch: warning: something odd happened'\n",
    )
    job = tmp_path / "job.sbatch"
    job.write_text("#!/bin/bash\n")
    with pytest.raises(RuntimeError, match=r"job\.sbatch"):
        submit_job(job)


def test_running_job_ids_parses_squeue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_stub(
        tmp_path, monkeypatch, "squeue", "#!/bin/bash\nprintf '812345_3\\n812347\\n'\n"
    )
    assert running_job_ids(["812345", "812346", "812347"]) == {"812345", "812347"}


def test_running_job_ids_empty_when_squeue_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "")
    assert running_job_ids(["812345"]) == set()


def test_running_job_ids_returns_live_ids_on_nonzero_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """squeue exits nonzero when any queried id has aged out, but still prints live ids to
    stdout; those must still be reported rather than discarded."""
    _install_stub(
        tmp_path,
        monkeypatch,
        "squeue",
        "#!/bin/bash\n"
        "printf '812345\\n'\n"
        "echo 'squeue: error: Invalid job id specified' >&2\n"
        "exit 1\n",
    )
    with caplog.at_level(logging.WARNING):
        assert running_job_ids(["812345", "999999"]) == {"812345"}
    assert "Invalid job id specified" in caplog.text


def test_running_job_ids_empty_stdout_and_nonzero_exit_returns_empty_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A genuine query failure (no ids parsed, nonzero exit) returns an empty set and logs a
    warning, rather than silently looking identical to a successful empty query."""
    _install_stub(
        tmp_path,
        monkeypatch,
        "squeue",
        "#!/bin/bash\necho 'squeue: error: Invalid job id specified' >&2\nexit 1\n",
    )
    with caplog.at_level(logging.WARNING):
        assert running_job_ids(["999999"]) == set()
    assert "Invalid job id specified" in caplog.text


def test_running_job_ids_returns_on_timeout_instead_of_hanging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A wedged scheduler controller must not hang `running_job_ids` forever.

    Regression guard for Finding 3: the `squeue` subprocess call had no timeout at all, so a
    wedged controller would hang `oplm sweep status` indefinitely instead of degrading to the
    `unknown` path that already exists for an unreachable scheduler. `timeout` is passed
    explicitly (well under the sleeping stub's duration) so the test stays fast rather than
    waiting out the real 30s default.
    """
    _install_stub(tmp_path, monkeypatch, "squeue", "#!/bin/bash\nsleep 2\n")
    started = time.monotonic()
    with caplog.at_level(logging.WARNING):
        result = running_job_ids(["812345"], timeout=0.1)
    elapsed = time.monotonic() - started
    assert result == set()
    assert elapsed < 2, f"running_job_ids blocked for {elapsed:.2f}s past its 0.1s timeout"
    assert "timed out" in caplog.text
