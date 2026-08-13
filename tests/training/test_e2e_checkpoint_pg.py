"""Dedicated checkpoint process group (Task 5.1b): 2-rank async-save pilot.

Task 5.1's HSDP e2e pilot hit a live ``gloo::EnforceNotMet ... collective mismatch``
abort: ``dcp.async_save`` runs its write-coordination collectives on a background
thread, over whatever process group it is given -- previously the DEFAULT process
group, the same one the training loop's own collectives (control-bundle reduce,
gradient all-reduce/reduce-scatter) run on from the main thread. Two threads issuing
collectives on one group interleave nondeterministically. The fix (Task 5.1b) builds
one dedicated GLOO process group for checkpoint I/O at ``Trainer.__init__`` and threads
it into every ``dcp.save``/``dcp.async_save`` call (see
``oplm.training.checkpoint.build_checkpoint_process_group``).

This module's e2e test drives the reproduction shape directly (aggressive
``save_every=1`` async saves under plain DDP, no HSDP needed) via
``_checkpoint_pg_worker.py``, launched twice under ``torch.distributed.run
--nproc_per_node=2`` mirroring ``test_e2e_dcp.py``'s reshard-worker subprocess pattern.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

_FIRST_LAUNCH_STEPS = 6
_SECOND_LAUNCH_STEPS = 10


def _child_env() -> dict[str, str]:
    """Environment for the subprocess ranks: CPU-only, repo root on ``PYTHONPATH``.

    Mirrors ``test_e2e_dcp.py``'s reshard-worker env and ``test_e2e_hsdp.py``'s
    ``_child_env``. ``CUDA_VISIBLE_DEVICES=""`` is required (not merely tidy):
    Accelerate's ``wait_for_everyone`` calls
    ``torch.distributed.barrier(device_ids=[local_process_index])`` even for a gloo
    process group, which maps the local index onto a CUDA ordinal and fails once it
    exceeds the visible GPU count. The worker imports ``tests.training.conftest``, so
    the repo root -- not just ``src/`` -- must be on the child's ``PYTHONPATH``.
    """
    repo_root = Path(__file__).resolve().parents[2]
    existing = os.environ.get("PYTHONPATH", "")
    child_pythonpath = os.pathsep.join(
        p for p in (str(repo_root), str(repo_root / "src"), existing) if p
    )
    return {
        **os.environ,
        "ACCELERATE_USE_CPU": "true",
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONPATH": child_pythonpath,
    }


def _run_pilot(
    run_dir: Path, training_parquet: Path, out_dir: Path, *, max_steps: int, auto_resume: bool
) -> list[dict[str, object]]:
    """Launch the 2-rank worker and return both ranks' recorded payloads."""
    worker = Path(__file__).with_name("_checkpoint_pg_worker.py")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "--rdzv_backend=c10d",
            "--rdzv_endpoint=localhost:0",
            str(worker),
            str(run_dir),
            str(training_parquet),
            str(out_dir),
            str(max_steps),
            "true" if auto_resume else "false",
        ],
        check=True,
        timeout=300,
        env=_child_env(),
    )
    return [json.loads((out_dir / f"rank{rank}.json").read_text()) for rank in (0, 1)]


def test_aggressive_async_save_cadence_survives_and_resumes(
    training_parquet: Path, tmp_path: Path
) -> None:
    """``save_every=1`` on 2 DDP ranks completes, commits every checkpoint, and resumes.

    Before Task 5.1b's fix, this exact configuration -- periodic async saves landing
    every single step, so a background write's DCP collectives overlap the very next
    step's training-loop collectives on the same default process group -- is the
    reproduction shape of the Task 5.1 pilot's live abort. It is not deterministic
    enough on this box to force reliably every run (see the task report for why), so
    this test is not a proof of the pre-fix bug by itself; it is one of two lines of
    evidence (the other being the unit-level process-group assertions in
    ``test_checkpoint.py``) that the fix is in place and does not regress correctness.

    Phase 1: a fresh 2-rank run trains ``_FIRST_LAUNCH_STEPS`` steps and commits one
    checkpoint every step (``save_total_limit`` set high enough that none rotate away).
    Phase 2: a second 2-rank launch with ``auto_resume=True`` picks up the newest
    committed checkpoint and continues to ``_SECOND_LAUNCH_STEPS`` at the same cadence.
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    first_results = _run_pilot(
        run_dir, training_parquet, out_dir, max_steps=_FIRST_LAUNCH_STEPS, auto_resume=False
    )
    for result in first_results:
        assert result["resumed_from_step"] == 0
        assert result["global_step"] == _FIRST_LAUNCH_STEPS

    committed = sorted(
        int(p.name.removeprefix("checkpoint-"))
        for p in run_dir.iterdir()
        if p.is_dir()
        and p.name.startswith("checkpoint-")
        and p.name.removeprefix("checkpoint-").isdigit()
    )
    assert committed == list(range(1, _FIRST_LAUNCH_STEPS + 1))
    assert list(run_dir.glob("checkpoint-*.tmp")) == []

    second_results = _run_pilot(
        run_dir, training_parquet, out_dir, max_steps=_SECOND_LAUNCH_STEPS, auto_resume=True
    )
    for result in second_results:
        assert result["resumed_from_step"] == _FIRST_LAUNCH_STEPS
        assert result["global_step"] == _SECOND_LAUNCH_STEPS

    committed_after_resume = sorted(
        int(p.name.removeprefix("checkpoint-"))
        for p in run_dir.iterdir()
        if p.is_dir()
        and p.name.startswith("checkpoint-")
        and p.name.removeprefix("checkpoint-").isdigit()
    )
    assert committed_after_resume == list(range(1, _SECOND_LAUNCH_STEPS + 1))
    assert list(run_dir.glob("checkpoint-*.tmp")) == []
