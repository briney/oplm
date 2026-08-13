"""Worker for the 2-rank remote-upload collective test (Task 4.2 fix round, Important
review finding). Run under ``torch.distributed.run``.

Exercises the REAL ``build_upload_group`` / ``build_upload_job`` / ``UploadManager`` /
``RemoteStore`` collective path over a genuine 2-process GLOO group -- no fakes on the
``torch.distributed`` side -- simulating a 2-node, 1-rank-per-node topology (each rank is
``local_process_index == 0`` of its own single-rank "node"). A real ``torch.distributed.run
--nproc_per_node=2`` launch on one machine is genuinely a single *local* node with 2
processes (``LOCAL_WORLD_SIZE=2``, ``LOCAL_RANK`` 0 and 1) -- that topology would only ever
produce ONE node leader, which would not exercise the multi-leader identity-check path this
test exists to cover (the Critical review finding: a bare barrier pairs calls by order, not
identity, across independently-drop-oldest-queued node leaders). So this worker overrides
``LOCAL_WORLD_SIZE`` to ``"1"`` in its own process (torchrun sets it to the true local count
before this script's ``main()`` runs; overriding it here, before any of this module's calls,
is what makes every rank simulate being alone on its own node) and uses a lightweight
accelerator stand-in (exposing only the 4 attributes ``build_upload_group``/
``build_upload_job``/``UploadManager`` actually read) instead of a real ``Accelerator``,
whose own ``local_process_index`` is derived from torchrun's real (single-node) ``LOCAL_RANK``
and can't be overridden the same way.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class _FakeAccelerator:
    """Exposes only what build_upload_group/build_upload_job/UploadManager read."""

    is_main_process: bool
    num_processes: int
    process_index: int
    local_process_index: int


def main(remote_uri: str, checkpoint_dir_str: str, out_dir: str) -> None:
    """Run the real collective upload path on this rank; rank 0 records the outcome.

    Args:
        remote_uri: ``file://`` URI for the ``RemoteStore`` both ranks upload to.
        checkpoint_dir_str: Shared local checkpoint directory both ranks write their own
            files into (mirrors a real shared/network ``output_dir``).
        out_dir: Directory to write ``result.json`` into (rank 0 only).
    """
    # Override torchrun's real (single-node) LOCAL_WORLD_SIZE before anything in
    # oplm.training.remote reads it -- see module docstring for why.
    os.environ["LOCAL_WORLD_SIZE"] = "1"

    import torch.distributed as dist

    from oplm.training.remote import (
        RemoteStore,
        UploadManager,
        build_upload_group,
        build_upload_job,
    )

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2

    accelerator = _FakeAccelerator(
        is_main_process=(rank == 0),
        num_processes=world_size,
        process_index=rank,
        local_process_index=0,  # every rank simulates being alone on its own node
    )

    upload_group = build_upload_group(accelerator)
    assert upload_group is not None

    checkpoint_dir = Path(checkpoint_dir_str)
    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()  # every rank waits for the shared dir before writing into it

    (checkpoint_dir / f"__{rank}_0.distcp").write_text(f"shard-{rank}")
    (checkpoint_dir / f"rng_state_{rank}.pt").write_text(f"rng-{rank}")
    if rank == 0:
        (checkpoint_dir / "trainer_state.json").write_text("{}")
        (checkpoint_dir / "config.yaml").write_text("cfg")

    dist.barrier()  # every rank's files must exist before either partitions/uploads

    manager = UploadManager(RemoteStore(remote_uri), accelerator, upload_group=upload_group)
    job = build_upload_job(
        checkpoint_dir,
        accelerator,
        permanent=False,
        save_total_limit=3,
        keep_every_n_steps=None,
    )
    manager.submit(job)
    manager.drain(timeout=30.0)

    if rank == 0:
        result = RemoteStore(remote_uri).latest_committed()
        payload = {
            "name": result[0] if result is not None else None,
            "files": sorted(result[1]["files"]) if result is not None else None,
        }
        Path(out_dir, "result.json").write_text(json.dumps(payload))

    dist.destroy_process_group()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
