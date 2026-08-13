"""Resume-target rank-agreement tests (Task 1.7 fix round, code review Important finding).

Two prior reviews flagged the wandb-vs-resume ordering (fixed in the Task 1.7 main
commit); the follow-up review of that fix additionally flagged that the resume-target
*resolution* itself -- the ``auto_resume`` scan for the newest committed checkpoint -- was
unguarded against cross-rank filesystem-visibility skew: every rank scanned the directory
itself, with no barrier and (even with one) no guarantee that a non-main rank's own
directory listing agrees with the main rank's on a multi-node shared filesystem (stale
listing caches, not just in-flight renames). ``oplm.training.trainer._resolve_resume_target``
fixes this by scanning on the main process only and broadcasting the result, making the
resume target rank-identical *by construction* rather than by filesystem-visibility
coincidence.

Coverage here:

- ``test_two_ranks_agree_on_the_resolved_resume_target`` (slow): the required minimum-bar
  test -- two real CPU/gloo processes construct the Trainer against a shared output_dir
  that already holds a committed checkpoint; both ranks must resolve the identical
  checkpoint path and restore the identical global_step.
- ``test_non_main_rank_never_scans_for_the_latest_checkpoint`` (fast, unit): isolates the
  narrower claim that the scan itself is gated on ``is_main_process``, using a lightweight
  fake accelerator stub (single-process pytest run -> ``accelerate.utils.
  broadcast_object_list`` falls through to a real no-op ``PartialState``, so no actual
  distributed group is needed to exercise the gating logic).

A recovery-pending ``checkpoint-<N>.old`` staging scenario (the specific race the barrier
alone would not have closed) was considered but not added as a *third* subprocess case: with
the fix in place the scan never runs on non-main ranks at all, so the ``.old``-recovery
timing is irrelevant to the outcome (the two cases above already fully exercise the fixed
code path), and staging that timing window reliably inside a 2-process subprocess harness
(forcing the main rank's rename to lag a non-main rank's scan) would need artificial
sleeps/hooks that add flakiness without adding coverage. See the Task 1.7 fix report for the
full reasoning.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.training.conftest import tiny_train_cfg


@pytest.mark.slow
def test_two_ranks_agree_on_the_resolved_resume_target(
    training_parquet: Path, tmp_path: Path
) -> None:
    """Two CPU/gloo ranks resolve the exact same auto_resume checkpoint and global_step.

    Phase 1 (this process, single-rank) trains to step 4 and checkpoints. Phase 2 launches
    ``tests/training/_resume_broadcast_worker.py`` under ``torch.distributed.run
    --nproc_per_node=2`` (forced onto CPU via ``ACCELERATE_USE_CPU``, mirroring
    ``tests/training/test_e2e_drain.py``'s subprocess pattern) against the same output_dir,
    with ``auto_resume=True`` and no explicit ``resume_from``. Both ranks must agree.
    """
    from oplm.training.trainer import Trainer

    run_dir = tmp_path / "run"
    cfg1 = tiny_train_cfg(run_dir, training_parquet, max_steps=4, save_every=4, log_every=1)
    Trainer(cfg1, callbacks=[]).train()
    assert (run_dir / "checkpoint-4").is_dir()

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    worker = Path(__file__).with_name("_resume_broadcast_worker.py")

    # The worker imports tests.training.conftest (tiny_train_cfg), so the repo root -- not
    # just src/ -- must be on the child's PYTHONPATH; this parent process only inherits
    # src/ from its own launch env (see the ENV GOTCHA note in the task brief).
    repo_root = Path(__file__).resolve().parents[2]
    src_dir = repo_root / "src"
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    child_pythonpath = os.pathsep.join(
        p for p in (str(repo_root), str(src_dir), existing_pythonpath) if p
    )
    # CUDA_VISIBLE_DEVICES="" alongside ACCELERATE_USE_CPU=true: on a box with >=1 GPU,
    # accelerate's wait_for_everyone still calls torch.distributed.barrier(device_ids=
    # [local_process_index]) even for a gloo/MULTI_CPU process group, which maps
    # local_process_index onto a CUDA device ordinal and fails with "invalid device
    # ordinal" once local_process_index exceeds the visible GPU count (e.g. rank 1 on a
    # 1-GPU box). Hiding CUDA entirely from the child processes avoids that codepath.
    env = {
        **os.environ,
        "ACCELERATE_USE_CPU": "true",
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONPATH": child_pythonpath,
    }
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
            "8",
        ],
        check=True,
        timeout=300,
        env=env,
    )

    rank0 = json.loads((out_dir / "rank0.json").read_text())
    rank1 = json.loads((out_dir / "rank1.json").read_text())

    assert rank0["resolved_resume_target"] == rank1["resolved_resume_target"]
    assert rank0["resolved_resume_target"] is not None
    assert rank0["resolved_resume_target"].endswith("checkpoint-4")
    assert rank0["global_step"] == rank1["global_step"] == 4


def test_non_main_rank_never_scans_for_the_latest_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-main rank's ``_resolve_resume_target`` never calls ``latest_checkpoint`` itself.

    Only the main process is allowed to scan the filesystem; a committed checkpoint exists
    here specifically so that, if the gating were ever removed, the scan (and this test)
    would notice it running.
    """
    from oplm.training import checkpoint as checkpoint_module
    from oplm.training.trainer import _resolve_resume_target

    (tmp_path / "checkpoint-4").mkdir()
    (tmp_path / "checkpoint-4" / "trainer_state.json").write_text("{}")

    calls: list[object] = []
    original_latest_checkpoint = checkpoint_module.latest_checkpoint

    def _counting_latest_checkpoint(output_dir: object) -> object:
        calls.append(output_dir)
        return original_latest_checkpoint(output_dir)  # ty: ignore[invalid-argument-type]

    monkeypatch.setattr(checkpoint_module, "latest_checkpoint", _counting_latest_checkpoint)

    class _FakeNonMainAccelerator:
        """Minimal stand-in exposing only what ``_resolve_resume_target`` reads."""

        is_main_process = False
        num_processes = 2

    result = _resolve_resume_target(
        _FakeNonMainAccelerator(),  # ty: ignore[invalid-argument-type]
        resume_from=None,
        auto_resume=True,
        output_dir=str(tmp_path),
    )

    assert calls == [], "non-main rank must never scan for the latest checkpoint itself"
    # In a real multi-process run this rank would receive the broadcast value from the main
    # process; in this single-process unit test there is no real distributed group to
    # broadcast over, so accelerate's broadcast_object_list is a no-op and the un-broadcast
    # local result (None, since this rank contributed nothing) passes through unchanged.
    assert result is None
