"""Submit generated job scripts and query their state."""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
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


def _decode_stdout(stdout: bytes | str | None) -> str:
    """Decode a captured subprocess stdout payload, tolerating ``bytes``, ``str``, or ``None``.

    ``subprocess.TimeoutExpired.stdout`` is ``bytes`` even when the parent ``subprocess.run`` call
    passed ``text=True``: that flag only governs how a *completed* call's output is decoded (via
    ``Popen.communicate``'s text wrapping on the happy path). A ``TimeoutExpired`` is raised while
    still waiting on the process, and carries whatever raw bytes had already been read from the
    pipe before it was killed, with no text decoding applied -- verified empirically on this
    project's interpreter (CPython 3.12.13): a ``TimeoutExpired`` from a ``text=True`` call still
    has ``stdout`` of type ``bytes``. Malformed trailing bytes (a line truncated mid-character by
    the kill) are replaced rather than raising, since partial output is inherently best-effort.
    """
    if stdout is None:
        return ""
    if isinstance(stdout, bytes):
        return stdout.decode("utf-8", errors="replace")
    return stdout


@dataclass(frozen=True)
class SchedulerQuery:
    """The result of asking the scheduler which of a set of job ids are still live.

    ``reachable`` is ``False`` when ``squeue`` is absent from PATH or the query timed out against
    a wedged controller -- in both cases ``ids`` carries no information about job state, and a
    caller must not treat the resulting empty ``ids`` as "nothing is running" (conflating the two
    is what let a wedged controller produce confident, wrong resubmit guidance for jobs that may
    still be queued or running). ``reachable`` is ``True`` even when ``squeue`` exited non-zero
    with partial results: that is its ordinary behavior once any one queried id has aged out, not
    scheduler unavailability.

    Attributes:
        ids: The subset of the queried job ids the scheduler reported as still known to it.
        reachable: Whether the scheduler could be asked at all.
    """

    ids: set[str]
    reachable: bool


def running_job_ids(
    job_ids: Sequence[str], *, timeout: float = SQUEUE_TIMEOUT_SECONDS
) -> SchedulerQuery:
    """Ask the scheduler which of ``job_ids`` it still knows about.

    ``squeue --jobs`` exits non-zero if *any* of the queried ids has aged out, even though it still
    prints the ids that remain live to stdout. This function parses stdout regardless of exit
    status, so a poll against a mix of running and finished ids still reports the running ones,
    with ``reachable=True`` -- a non-zero exit here is ``squeue``'s normal steady state (an id
    ages out once that job finishes), not a failure. It is still logged as a warning (with
    ``squeue``'s stderr) so a genuine query failure is visible.

    A wedged scheduler controller can leave ``squeue`` hanging indefinitely; ``timeout`` bounds
    that wait so an interactive command like ``oplm sweep status`` degrades to
    ``reachable=False`` instead of hanging forever. Whatever partial stdout the process had
    already emitted before being killed is decoded (see ``_decode_stdout``) and parsed into
    ``ids`` -- a live job id ``squeue`` reported just before wedging must not be thrown away.

    ``squeue`` being entirely absent from PATH (e.g. off-cluster) also yields
    ``reachable=False``, with an empty ``ids``.

    Args:
        job_ids: Base job ids to look up (array jobs report as ``<id>_<index>`` in ``squeue``;
            this matches on the base id).
        timeout: Seconds to wait for ``squeue`` before giving up. Defaults to
            ``SQUEUE_TIMEOUT_SECONDS``; overridable so tests need not wait 30 seconds.

    Returns:
        A ``SchedulerQuery`` pairing the live subset of ``job_ids`` with whether the scheduler
        could be reached at all.
    """
    if not job_ids:
        return SchedulerQuery(ids=set(), reachable=True)
    if shutil.which("squeue") is None:
        return SchedulerQuery(ids=set(), reachable=False)
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
        partial_ids = _parse_squeue_stdout(_decode_stdout(exc.stdout))
        return SchedulerQuery(ids=partial_ids, reachable=False)
    if result.returncode != 0:
        logger.warning(
            "squeue exited %d while querying %d job id(s): %s",
            result.returncode,
            len(job_ids),
            result.stderr.strip(),
        )
    return SchedulerQuery(ids=_parse_squeue_stdout(result.stdout), reachable=True)
