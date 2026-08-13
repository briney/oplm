"""Worker for the Muon-under-FSDP2 spike (Task 0.3). Run under torch.distributed.run.

Builds a toy 2-layer ``nn.Linear`` stack, shards it with FSDP2's ``fully_shard``
over a 1-D CPU/gloo mesh (so every parameter becomes a 2-D DTensor), and
attempts to construct and step ``torch.optim.Muon`` directly over those
DTensor params. Records the outcome (construction/step success or failure,
per-step loss, and whether ``get_state_dict`` round-trips the optimizer state)
to ``out_dir/rank<i>.json`` so the parent test process can render the verdict.

This is the spike artifact referenced by
``docs/superpowers/specs/2026-08-12-fault-tolerant-training-design.md`` §7 and
``TODOS.md`` Task 0.3 / Task 5.2.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any


def main(out_dir: str) -> None:
    """Run the spike on this rank and write its outcome to ``out_dir/rank<i>.json``.

    Args:
        out_dir: Directory to write ``rank<i>.json`` into (must already exist).
    """
    import torch
    import torch.distributed as dist
    from torch import nn
    from torch.distributed.checkpoint.state_dict import get_state_dict
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.manual_seed(0)

    mesh = init_device_mesh("cpu", (world_size,))

    # Toy 2-layer regression model; bias=False keeps every parameter 2D so the
    # whole model is Muon-eligible (mirrors the real trainer's rule of routing
    # only 2D hidden weights to Muon; see src/oplm/training/optim.py).
    model = nn.Sequential(
        nn.Linear(16, 16, bias=False),
        nn.Linear(16, 16, bias=False),
    )
    for layer in model:
        fully_shard(layer, mesh=mesh)
    fully_shard(model, mesh=mesh)

    x = torch.randn(8, 16)
    y = torch.randn(8, 16)

    result: dict[str, Any] = {"rank": rank}

    try:
        muon = torch.optim.Muon(model.parameters(), lr=0.02)
    except (NotImplementedError, RuntimeError) as exc:
        result["status"] = "construct_failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        _write(out_dir, rank, result)
        dist.destroy_process_group()
        return

    initial_weight = next(model.parameters()).full_tensor().clone()
    losses: list[float] = []
    try:
        for _ in range(3):
            muon.zero_grad()
            loss = nn.functional.mse_loss(model(x), y)
            loss.backward()
            muon.step()
            losses.append(loss.item())
    except (NotImplementedError, RuntimeError) as exc:
        result["status"] = "step_failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        _write(out_dir, rank, result)
        dist.destroy_process_group()
        return

    final_weight = next(model.parameters()).full_tensor()
    result["status"] = "ok"
    result["losses"] = losses
    result["param_changed"] = not torch.allclose(initial_weight, final_weight)

    try:
        _, optim_state_dict = get_state_dict(model, [muon])
        result["checkpoint_status"] = "ok"
        result["checkpoint_state_keys"] = list(optim_state_dict.keys())
    except (NotImplementedError, RuntimeError) as exc:
        result["checkpoint_status"] = "failed"
        result["checkpoint_error"] = f"{type(exc).__name__}: {exc}"

    _write(out_dir, rank, result)
    dist.destroy_process_group()


def _write(out_dir: str, rank: int, result: dict[str, Any]) -> None:
    """Serialize ``result`` to ``out_dir/rank<rank>.json``."""
    Path(out_dir, f"rank{rank}.json").write_text(json.dumps(result))


if __name__ == "__main__":
    main(sys.argv[1])
