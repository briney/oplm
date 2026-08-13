"""Muon-under-FSDP2 spike (Task 0.3).

Phase 5 (HSDP) needs to know whether ``torch.optim.Muon`` can run directly on
the DTensor params FSDP2's ``fully_shard`` produces, or whether it needs a
gather -> Newton-Schulz -> scatter adapter (torchtitan-style distributed
Muon). This launches a 2-process CPU/gloo subprocess (same pattern as Task
0.1's ``tests/data/test_double_sharding.py``) that shards a toy 2-layer
``nn.Linear`` stack with ``fully_shard`` and attempts to construct and step
``torch.optim.Muon`` over the resulting 2-D DTensor params, then probes
whether the Phase-2 checkpoint path (``get_state_dict``) round-trips the
sharded optimizer state.

The test stays green either way: if Muon rejects DTensor params, the *skip*
message records the verdict; if it accepts them, the assertions below record
the opposite verdict. See ``_fsdp2_muon_spike_worker.py`` for the subprocess
entry point, and
``docs/superpowers/specs/2026-08-12-fault-tolerant-training-design.md`` §7
plus ``TODOS.md`` Task 5.2 for the recorded outcome.

VERDICT: ``torch.optim.Muon`` works as-is on FSDP2 DTensor params (2-process
CPU/gloo, 3 steps): construction and ``step()`` raised no exception, the loss
decreased monotonically each step, the sharded parameter changed, and
``get_state_dict(model, [muon])`` round-tripped the optimizer state (momentum
buffers stayed sharded ``DTensor``s, ``Shard(dim=0)``). No gather -> NS ->
scatter adapter is required; Task 5.2 can be skipped.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.slow
def test_muon_optimizer_on_fsdp2_dtensor_params(tmp_path: Path) -> None:
    """Muon construction + 3 steps + checkpoint probe on FSDP2-sharded DTensor params.

    Launches ``_fsdp2_muon_spike_worker.py`` under
    ``torch.distributed.run --nproc_per_node=2`` (CPU/gloo). Each rank writes its
    outcome to ``rank<i>.json``. If Muon rejected the DTensor params (at
    construction or during ``step()``), this test *records that verdict via
    skip* rather than failing, per the Task 0.3 brief. Otherwise it asserts the
    spike's success criteria: no exception, loss decreased, params changed,
    and (informationally) whether the checkpoint round-trip succeeded.
    """
    worker = Path(__file__).with_name("_fsdp2_muon_spike_worker.py")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "--rdzv_backend=c10d",
            "--rdzv_endpoint=localhost:0",
            str(worker),
            str(tmp_path),
        ],
        check=True,
        timeout=300,
    )

    results = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in (0, 1)]

    for result in results:
        if result["status"] in ("construct_failed", "step_failed"):
            pytest.skip(f"Muon+DTensor unsupported: {result['error']}")

    for result in results:
        assert result["status"] == "ok"
        losses = result["losses"]
        assert len(losses) == 3
        assert losses[-1] < losses[0], f"loss did not decrease: {losses}"
        assert result["param_changed"], "Muon step did not change the sharded parameter"

    # Informational: record but don't hard-fail on the checkpoint probe, since
    # the brief asks us to *record* whether get_state_dict round-trips, not to
    # gate the Muon-acceptance verdict on it.
    checkpoint_statuses = {result["checkpoint_status"] for result in results}
    assert checkpoint_statuses == {"ok"}, (
        f"expected get_state_dict to round-trip on all ranks, got {results}"
    )
