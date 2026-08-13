"""Mesh-shape resolution for ``train.parallelism=hsdp`` (Task 5.1).

The 2-D device mesh HSDP needs -- ``(num_nodes, gpus_per_node)`` with dim names
``("replicate", "shard")`` -- is derived purely from ``WORLD_SIZE`` /
``LOCAL_WORLD_SIZE``, so the arithmetic and its failure modes are unit-testable
without a process group. The end-to-end behavior of the mesh (a real 2-rank
CPU/gloo pilot) is covered by ``test_e2e_hsdp.py``.
"""

from __future__ import annotations

import pytest

from oplm.training.parallel import resolve_mesh_dims


def test_single_node_degrades_to_plain_fsdp() -> None:
    """One node: the replicate dim is 1, so HSDP degenerates to plain FSDP."""
    assert resolve_mesh_dims(world_size=8, local_world_size=8) == (1, 8)


def test_multi_node_splits_replicate_and_shard_dims() -> None:
    """Four 8-GPU nodes shard within a node and replicate across nodes."""
    assert resolve_mesh_dims(world_size=32, local_world_size=8) == (4, 8)


def test_missing_local_world_size_assumes_one_node() -> None:
    """A launcher that does not export ``LOCAL_WORLD_SIZE`` degrades to one node."""
    assert resolve_mesh_dims(world_size=4, local_world_size=None) == (1, 4)


def test_world_size_one_points_the_user_at_ddp() -> None:
    """A single-process run cannot shard: refuse with an actionable message."""
    with pytest.raises(ValueError, match="ddp"):
        resolve_mesh_dims(world_size=1, local_world_size=1)


def test_local_world_size_not_dividing_world_size_raises() -> None:
    """A ``LOCAL_WORLD_SIZE`` inconsistent with the topology raises, naming the env var."""
    with pytest.raises(RuntimeError, match="LOCAL_WORLD_SIZE"):
        resolve_mesh_dims(world_size=12, local_world_size=8)


def test_local_world_size_larger_than_world_size_raises() -> None:
    """``LOCAL_WORLD_SIZE > WORLD_SIZE`` is incoherent and raises rather than clamping."""
    with pytest.raises(RuntimeError, match="LOCAL_WORLD_SIZE"):
        resolve_mesh_dims(world_size=4, local_world_size=8)


@pytest.mark.parametrize("bad_local", [0, -1])
def test_non_positive_local_world_size_is_ignored(bad_local: int) -> None:
    """A garbage ``LOCAL_WORLD_SIZE`` is treated as absent (single node), not fatal."""
    assert resolve_mesh_dims(world_size=4, local_world_size=bad_local) == (1, 4)
