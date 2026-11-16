"""Distributed Muon: two DDP-style ranks must reproduce the single-process update."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tests.training._distributed_muon_worker import run_steps


@pytest.mark.slow
def test_two_ranks_match_one_process(tmp_path: Path) -> None:
    """Each rank computes half the updates; both end bit-equal to one process doing all."""
    worker = Path(__file__).with_name("_distributed_muon_worker.py")
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
    expected = run_steps()
    for rank in (0, 1):
        got = torch.load(tmp_path / f"rank{rank}.pt")
        for a, b in zip(got, expected, strict=True):
            assert torch.equal(a, b)
