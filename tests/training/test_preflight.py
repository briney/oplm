"""Unit + multi-process tests for the startup preflight check (Task 1.8, fix round).

`run_preflight` is called first in `Trainer.__init__`, right after the Accelerator is
constructed and before anything touches checkpoints or data, so a sick node fails fast
and attributably instead of hanging mid-training. The single-process tests below exercise
it against a lightweight `SimpleNamespace` stand-in for `Accelerator`
(device/process_index/num_processes) -- the contract under test there is "allocate +
matmul", not accelerate's own plumbing.

The multi-process contract -- every rank must reach the collective exchange before
anyone raises, so a locally-failing rank does not leave healthy ranks hanging in a later
collective for the full process-group timeout -- is exercised two ways:

- Fast unit tests monkeypatch `preflight_module.gather_object` to fabricate the exchange
  result, isolating the "gather before raise" and "raise from gathered results, not just
  local status" logic without a real process group.
- `test_two_ranks_abort_attributably_when_one_ranks_local_check_fails` (slow) drives a
  real 2-process CPU/gloo subprocess (`_preflight_worker.py`) where rank 1's local check
  is forced to fail, and asserts both ranks record an attributable failure naming rank 1
  and its hostname -- not a silent hang.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from oplm.training import preflight as preflight_module
from oplm.training.preflight import run_preflight


def _make_accelerator(num_processes: int = 1, process_index: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        device=torch.device("cpu"),
        process_index=process_index,
        num_processes=num_processes,
    )


# ---------------------------------------------------------------------------
# Single-process: no collective, local failure raises directly.
# ---------------------------------------------------------------------------


def test_run_preflight_passes_clean_single_process() -> None:
    """A healthy single-process rank passes without raising."""
    run_preflight(_make_accelerator(num_processes=1))


def test_run_preflight_logs_rank_host_device(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="oplm.training.preflight"):
        run_preflight(_make_accelerator(num_processes=1))

    host = socket.gethostname()
    messages = [record.message for record in caplog.records]
    assert any("rank=0" in m and f"host={host}" in m and "device=cpu" in m for m in messages)


def test_run_preflight_raises_with_hostname_on_matmul_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing matmul (a stand-in for a sick GPU) raises RuntimeError naming this host."""

    def _boom(*args: object, **kwargs: object) -> torch.Tensor:
        raise RuntimeError("CUDA error: unspecified launch failure")

    monkeypatch.setattr(preflight_module.torch, "matmul", _boom)

    with pytest.raises(RuntimeError) as exc_info:
        run_preflight(_make_accelerator(num_processes=1))
    assert socket.gethostname() in str(exc_info.value)


def test_run_preflight_raises_with_hostname_on_allocation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any failure inside the check (not just the matmul) is wrapped the same way."""

    def _boom(*args: object, **kwargs: object) -> torch.Tensor:
        raise RuntimeError("out of memory")

    monkeypatch.setattr(preflight_module.torch, "empty", _boom)

    with pytest.raises(RuntimeError) as exc_info:
        run_preflight(_make_accelerator(num_processes=1))
    assert socket.gethostname() in str(exc_info.value)


def test_run_preflight_single_process_never_gathers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Single-process has no group to exchange with -- gather_object must not be called."""
    calls: list[object] = []
    monkeypatch.setattr(
        preflight_module,
        "gather_object",
        lambda payload: (calls.append(payload), payload)[1],
    )
    run_preflight(_make_accelerator(num_processes=1))
    assert calls == []


# ---------------------------------------------------------------------------
# Multi-process (fabricated exchange via monkeypatched gather_object): a locally-failing
# rank must reach the exchange before raising, and any rank must raise from the gathered
# results, not just its own local status.
# ---------------------------------------------------------------------------


def test_run_preflight_reaches_gather_before_raising_on_local_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rank whose local check fails must still reach the exchange, not raise before it.

    Regression test for the fix-round finding: raising before the gather leaves healthy
    ranks blocked in a later collective for the full process-group timeout.
    """
    host = socket.gethostname()
    boom_message = "CUDA error: unspecified launch failure"

    def _boom(*args: object, **kwargs: object) -> torch.Tensor:
        raise RuntimeError(boom_message)

    monkeypatch.setattr(preflight_module.torch, "matmul", _boom)

    gather_calls: list[list[tuple[int, str, str | None]]] = []

    def _fake_gather(payload: list[tuple[int, str, str | None]]):
        gather_calls.append(payload)
        # Fabricate a 2-rank exchange: this rank's real (failing) payload plus a
        # healthy rank 0.
        return [(0, "other-host", None), *payload]

    monkeypatch.setattr(preflight_module, "gather_object", _fake_gather)

    with pytest.raises(RuntimeError) as exc_info:
        run_preflight(_make_accelerator(num_processes=2, process_index=1))

    assert gather_calls == [[(1, host, boom_message)]], (
        "local failure must not skip the collective exchange"
    )
    assert f"rank=1 host={host}" in str(exc_info.value)
    assert boom_message in str(exc_info.value)


def test_run_preflight_raises_naming_a_different_failed_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rank whose OWN local check passed still raises if the gather reports a failure
    elsewhere -- attribution comes from the gathered results, not local status alone."""
    monkeypatch.setattr(
        preflight_module,
        "gather_object",
        lambda payload: [*payload, (1, "sick-node", "out of memory")],
    )

    with pytest.raises(RuntimeError) as exc_info:
        run_preflight(_make_accelerator(num_processes=2, process_index=0))

    assert "rank=1 host=sick-node" in str(exc_info.value)
    assert "out of memory" in str(exc_info.value)


def test_run_preflight_passes_when_all_gathered_results_ok(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        preflight_module,
        "gather_object",
        lambda payload: [*payload, (1, "other-host", None)],
    )
    run_preflight(_make_accelerator(num_processes=2, process_index=0))  # must not raise


# ---------------------------------------------------------------------------
# Real 2-process CPU/gloo subprocess: rank 1's local check fails.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_two_ranks_abort_attributably_when_one_ranks_local_check_fails(tmp_path: Path) -> None:
    """Rank 1's local check fails; both ranks must abort quickly, attributing rank 1.

    Launches `_preflight_worker.py` under `torch.distributed.run --nproc_per_node=2`
    (CPU/gloo). Rank 1 patches its own local check to fail (via env var); each rank
    writes its outcome to `out_dir/rank<i>.json` and then re-raises so the process exits
    nonzero and torchrun's own aggregated stderr also carries the message.
    """
    worker = Path(__file__).with_name("_preflight_worker.py")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    repo_root = Path(__file__).resolve().parents[2]
    src_dir = repo_root / "src"
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    child_pythonpath = os.pathsep.join(
        p for p in (str(repo_root), str(src_dir), existing_pythonpath) if p
    )
    env = {
        **os.environ,
        "ACCELERATE_USE_CPU": "true",
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONPATH": child_pythonpath,
        "OPLM_TEST_FAIL_RANK1": "1",
    }

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "--rdzv_backend=c10d",
            "--rdzv_endpoint=localhost:0",
            str(worker),
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )

    assert result.returncode != 0, (
        f"expected a nonzero exit from the injected rank-1 failure\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    rank0 = json.loads((out_dir / "rank0.json").read_text())
    rank1 = json.loads((out_dir / "rank1.json").read_text())
    host = socket.gethostname()

    # Both ranks must have raised -- the healthy rank 0 too, attributing rank 1 -- not
    # just the locally-failing rank 1.
    assert rank0["error"] is not None, f"rank 0 should have aborted too, got {rank0}"
    assert rank1["error"] is not None, f"rank 1 should have aborted, got {rank1}"
    assert f"rank=1 host={host}" in rank0["error"]
    assert f"rank=1 host={host}" in rank1["error"]

    # torchrun's aggregated stderr must also carry the attribution (not a generic,
    # unattributed collective-timeout error naming nobody).
    assert host in result.stderr
