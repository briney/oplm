"""Submit generated job scripts and query their state."""

from __future__ import annotations

import logging
import shutil
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from oplm.slurm.render import SubmitEntry

logger = logging.getLogger(__name__)

# Wall-clock budget for a single `squeue` query. `status` is an interactive command; a wedged
# controller must degrade to the `unknown` path already in place rather than hang forever. Not
# applied to `sbatch` -- see the comment at that call site for why.
SQUEUE_TIMEOUT_SECONDS = 30.0


def submit_job(script: Path, *, depends_on: Sequence[str] = ()) -> str:
    """Submit one script with ``sbatch --parsable`` and return its job id.

    Dependencies use ``afterany`` so a diverged upstream run cannot wedge a downstream job:
    under ``afterok`` the first run that exits nonzero would leave its dependent permanently
    in ``DependencyNeverSatisfied``.

    Args:
        script: Path to the sbatch script.
        depends_on: Job ids this submission waits on.

    Returns:
        The Slurm job id.

    Raises:
        RuntimeError: If ``sbatch`` exits non-zero, or if it exits zero but its stdout does not
            contain a plausible (non-empty, all-digit) Slurm job id — e.g. a transient scheduler
            hiccup or a site wrapper that swallows output.
    """
    argv = ["sbatch", "--parsable"]
    if depends_on:
        argv.append(f"--dependency=afterany:{':'.join(depends_on)}")
    argv.append(str(script))
    # Deliberately no timeout here (unlike `running_job_ids`'s `squeue` call): a timed-out
    # `sbatch` could still have submitted the job before we gave up waiting, and a caller that
    # then retries on a `RuntimeError` would double-submit. `squeue` is a read-only query with no
    # such risk, so it gets a timeout and this does not.
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"sbatch failed for {script} (exit {result.returncode}): {result.stderr.strip()}"
        )
    # --parsable prints "<jobid>" or "<jobid>;<cluster>".
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(
            f"sbatch exited 0 for {script} but did not print a valid job id: "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
    return job_id


def submit_all(entries: Sequence[SubmitEntry], *, base_dir: Path) -> dict[str, str]:
    """Submit every entry in order, threading earlier job ids into later dependencies.

    If a submission partway through raises, the entries submitted before it are already queued
    on the scheduler; this function does not attempt to cancel them, and the ids collected so
    far are lost with the exception (the caller sees only the partial submission on the
    scheduler itself, e.g. via ``squeue`` or the sbatch stderr in the raised error).

    Args:
        entries: Jobs to submit, in dependency order.
        base_dir: Directory the entries' script paths are relative to.

    Returns:
        Mapping of each entry's ``var`` to its job id.

    Raises:
        RuntimeError: If any ``sbatch`` submission fails.
    """
    ids: dict[str, str] = {}
    for entry in entries:
        depends = [ids[var] for var in entry.depends_on]
        job_id = submit_job(base_dir / entry.script, depends_on=depends)
        ids[entry.var] = job_id
        logger.info("submitted %s as %s", entry.script, job_id)
    return ids


def _parse_squeue_stdout(stdout: str) -> set[str]:
    # Array elements report as "<arrayjobid>_<index>"; match on the base id.
    return {line.strip().split("_", 1)[0] for line in stdout.splitlines() if line.strip()}


def running_job_ids(job_ids: Sequence[str], *, timeout: float = SQUEUE_TIMEOUT_SECONDS) -> set[str]:
    """Return the subset of ``job_ids`` still known to the scheduler.

    An empty return means one of three things: no queried job is currently known to the
    scheduler (the normal steady state — an id ages out of ``squeue`` once that job finishes,
    this is not an error), ``squeue`` is unavailable (e.g. off-cluster), or the query timed out
    against a wedged controller (see below) -- in all three cases callers degrade to
    filesystem-only status rather than crashing or hanging. A caller that must distinguish these
    cases has to check scheduler availability itself (e.g. its own ``shutil.which("squeue")``)
    rather than infer it from an empty result here.

    ``squeue --jobs`` exits non-zero if *any* of the queried ids has aged out, even though it still
    prints the ids that remain live to stdout. This function parses stdout regardless of exit
    status, so a poll against a mix of running and finished ids still reports the running ones. A
    non-zero exit is logged as a warning (with ``squeue``'s stderr) so a genuine query failure is
    visible, but whatever ids were parsed from stdout are still returned.

    A wedged scheduler controller can leave ``squeue`` hanging indefinitely; ``timeout`` bounds
    that wait so an interactive command like ``oplm sweep status`` degrades to the already-present
    "scheduler unreachable" path instead of hanging forever. A timeout is logged as a warning, the
    same as a non-zero exit, and whatever partial stdout the process managed to emit (typically
    none, since the process is killed on POSIX before it can be collected) is still parsed.

    Args:
        job_ids: Base job ids to look up (array jobs report as ``<id>_<index>`` in ``squeue``;
            this matches on the base id).
        timeout: Seconds to wait for ``squeue`` before giving up. Defaults to
            ``SQUEUE_TIMEOUT_SECONDS``; overridable so tests need not wait 30 seconds.

    Returns:
        The subset of ``job_ids`` that ``squeue`` still reports.
    """
    if not job_ids or shutil.which("squeue") is None:
        return set()
    try:
        result = subprocess.run(
            ["squeue", "--noheader", "--format=%i", "--jobs", ",".join(job_ids)],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        logger.warning(
            "squeue timed out after %.0fs while querying %d job id(s)", timeout, len(job_ids)
        )
        # `text=True` makes this `str` at runtime; `TimeoutExpired` is only typed `AnyStr | None`
        # generically, and POSIX leaves it `None` anyway (see docstring above).
        partial_stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        return _parse_squeue_stdout(partial_stdout)
    if result.returncode != 0:
        logger.warning(
            "squeue exited %d while querying %d job id(s): %s",
            result.returncode,
            len(job_ids),
            result.stderr.strip(),
        )
    return _parse_squeue_stdout(result.stdout)
