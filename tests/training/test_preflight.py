"""Unit tests for the startup preflight check (Task 1.8).

`run_preflight` is called first in `Trainer.__init__`, right after the Accelerator is
constructed and before anything touches checkpoints or data, so a sick node fails fast
and attributably instead of hanging mid-training. These tests exercise it against a
lightweight `SimpleNamespace` stand-in for `Accelerator` (device/process_index/
num_processes/reduce) rather than a real one, since the contract under test is
allocate + matmul + (conditional) collective, not accelerate's own plumbing.
"""

from __future__ import annotations

import logging
import socket
from types import SimpleNamespace

import pytest
import torch

from oplm.training import preflight as preflight_module
from oplm.training.preflight import run_preflight


def _make_accelerator(num_processes: int = 1, reduce_calls: list | None = None) -> SimpleNamespace:
    calls = reduce_calls if reduce_calls is not None else []

    def _reduce(tensor: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
        calls.append(tensor)
        return tensor

    return SimpleNamespace(
        device=torch.device("cpu"),
        process_index=0,
        num_processes=num_processes,
        reduce=_reduce,
    )


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


def test_run_preflight_skips_collective_on_single_process() -> None:
    calls: list[torch.Tensor] = []
    run_preflight(_make_accelerator(num_processes=1, reduce_calls=calls))
    assert calls == []


def test_run_preflight_reduces_when_distributed() -> None:
    calls: list[torch.Tensor] = []
    run_preflight(_make_accelerator(num_processes=2, reduce_calls=calls))
    assert len(calls) == 1


def test_run_preflight_raises_with_hostname_on_reduce_failure() -> None:
    """A stalled/failing collective is caught and attributed the same way as a local failure."""

    def _boom(tensor: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
        raise RuntimeError("NCCL error: unhandled system error")

    accelerator = SimpleNamespace(
        device=torch.device("cpu"), process_index=0, num_processes=2, reduce=_boom
    )
    with pytest.raises(RuntimeError) as exc_info:
        run_preflight(accelerator)
    assert socket.gethostname() in str(exc_info.value)
