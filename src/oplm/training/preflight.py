"""Startup preflight check (Task 1.8): catch a sick node before it touches anything else.

Called first in :meth:`~oplm.training.trainer.Trainer.__init__`, right after the
Accelerator is constructed and before the checkpoint-cleanup/resume-resolution block —
a dead GPU or broken NCCL path must fail here, in seconds and attributably, rather than
hanging mid-training or corrupting a resume by touching checkpoint state first.
"""

from __future__ import annotations

import logging
import socket
from typing import TYPE_CHECKING

import torch
from accelerate.utils import gather_object

if TYPE_CHECKING:
    from accelerate import Accelerator

logger = logging.getLogger(__name__)

# 64 MiB, expressed in float32 elements.
_ALLOC_ELEMENTS = 64 * 1024 * 1024 // 4
_MATMUL_DIM = 1024


def _run_local_checks(device: torch.device) -> str | None:
    """Run this rank's local alloc+matmul check.

    Args:
        device: This rank's device.

    Returns:
        ``None`` on success, or a string describing the failure.
    """
    try:
        buffer = torch.empty(_ALLOC_ELEMENTS, dtype=torch.float32, device=device)
        matrix = torch.randn(_MATMUL_DIM, _MATMUL_DIM, device=device)
        torch.matmul(matrix, matrix)
        del buffer, matrix
    except Exception as exc:  # noqa: BLE001 - any failure here means this node is unhealthy
        return str(exc)
    return None


def run_preflight(accelerator: Accelerator) -> None:
    """Verify every rank's device and allocator are healthy, attributably, together.

    Every rank always runs its own local alloc+matmul check first, WITHOUT raising on a
    local failure yet. When running with more than one process, every rank then always
    participates in exactly one round of collective exchange
    (:func:`accelerate.utils.gather_object`) of its own ``(rank, host, error)`` --
    unconditionally, regardless of whether its local check passed or failed. Only after
    that exchange does any rank raise, and if any rank's local check failed, EVERY rank
    raises, naming every failing rank.

    This symmetry matters: if a locally-failing rank raised immediately instead, every
    healthy rank would be left blocking in this (or the next) collective for the full
    process-group timeout before dying with a generic, unattributed collective-timeout
    error -- exactly the "hangs instead of failing fast" failure mode this check exists
    to prevent. The gather also doubles as the collective-health probe itself (it
    exercises the same comm path a real collective would), so no separate
    reduce/all-reduce is needed just to prove the group is alive.

    Args:
        accelerator: The trainer's just-constructed Accelerator.

    Raises:
        RuntimeError: If this rank's local check failed (single-process, no group to
            exchange with), or if ANY rank's local check failed (multi-process, after
            the exchange). The message names every failing rank's index, hostname, and
            local error.
    """
    host = socket.gethostname()
    device = accelerator.device
    rank = accelerator.process_index
    logger.info("preflight: rank=%d host=%s device=%s", rank, host, device)

    error = _run_local_checks(device)

    if accelerator.num_processes == 1:
        if error is not None:
            raise RuntimeError(f"preflight check failed on host {host}: {error}")
        return

    # Unconditional on every rank, regardless of `error` above -- a rank whose local
    # check just failed must still reach this exchange, or the healthy ranks hang here
    # waiting for it instead of failing fast.
    results: list[tuple[int, str, str | None]] = gather_object([(rank, host, error)])
    failures = [(r, h, e) for r, h, e in results if e is not None]
    if failures:
        details = "; ".join(f"rank={r} host={h}: {e}" for r, h, e in failures)
        raise RuntimeError(f"preflight check failed: {details}")
