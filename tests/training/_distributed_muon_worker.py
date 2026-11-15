"""Worker for ``test_distributed_muon.py``: a few Muon steps on toy matrices.

Run under ``torch.distributed.run`` it joins a gloo group and each rank orthogonalizes
only the matrices it owns; imported and called directly it is the single-process
reference. Every rank seeds the same params and gradients, as DDP would deliver them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from oplm.training.optim import Muon

# Three of one shape and a lone wide one: with 2 ranks, rank 1 owns nothing in the
# second shape and must still join its all-reduce.
SHAPES = [(8, 8), (8, 8), (8, 8), (8, 24)]


def run_steps() -> list[torch.Tensor]:
    """Take three seeded Muon steps and return the final parameters."""
    gen = torch.Generator().manual_seed(0)
    params = [torch.nn.Parameter(torch.randn(shape, generator=gen)) for shape in SHAPES]
    optimizer = Muon(params, lr=0.02, weight_decay=0.01)
    for _ in range(3):
        for param in params:
            param.grad = torch.randn(param.shape, generator=gen)
        optimizer.step()
    return [param.detach() for param in params]


if __name__ == "__main__":
    import torch.distributed as dist

    dist.init_process_group(backend="gloo")
    assert dist.get_world_size() == 2
    torch.save(run_steps(), Path(sys.argv[1]) / f"rank{dist.get_rank()}.pt")
    dist.destroy_process_group()
