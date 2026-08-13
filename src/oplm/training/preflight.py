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

if TYPE_CHECKING:
    from accelerate import Accelerator

logger = logging.getLogger(__name__)

# 64 MiB, expressed in float32 elements.
_ALLOC_ELEMENTS = 64 * 1024 * 1024 // 4
_MATMUL_DIM = 1024


def run_preflight(accelerator: Accelerator) -> None:
    """Verify this rank's device, allocator, and collective path are healthy.

    Allocates a 64 MB tensor, runs one 1024x1024 matmul, and — only when running with
    more than one process — reduces a small probe tensor across the group. Must be
    called unconditionally on every rank (same discipline as the resume-target
    broadcast): the collective only fires under ``accelerator.num_processes > 1``, but
    every rank has to reach it together or the group hangs.

    Args:
        accelerator: The trainer's just-constructed Accelerator.

    Raises:
        RuntimeError: If allocation, the matmul, or the collective fails on this rank.
            The message names this rank's hostname so a sick node is attributable.
    """
    host = socket.gethostname()
    device = accelerator.device
    logger.info("preflight: rank=%d host=%s device=%s", accelerator.process_index, host, device)
    try:
        buffer = torch.empty(_ALLOC_ELEMENTS, dtype=torch.float32, device=device)
        matrix = torch.randn(_MATMUL_DIM, _MATMUL_DIM, device=device)
        torch.matmul(matrix, matrix)
        del buffer, matrix
        if accelerator.num_processes > 1:
            probe = torch.ones(1, device=device)
            accelerator.reduce(probe, reduction="sum")
    except Exception as exc:  # noqa: BLE001 - any failure here means this node is unhealthy
        raise RuntimeError(f"preflight check failed on host {host}: {exc}") from exc
