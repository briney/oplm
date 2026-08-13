"""Mesh shape and mixed-precision policy for ``train.parallelism=hsdp`` (Task 5.1).

The 2-D device mesh HSDP needs -- ``(num_nodes, gpus_per_node)`` with dim names
``("replicate", "shard")`` -- is derived purely from ``WORLD_SIZE`` /
``LOCAL_WORLD_SIZE``, and the ``MixedPrecisionPolicy`` purely from
``train.mixed_precision``, so both and their failure modes are unit-testable without a
process group. The end-to-end behavior (a real 2-rank CPU/gloo pilot, in both fp32 and
bf16) is covered by ``test_e2e_hsdp.py``.
"""

from __future__ import annotations

import logging

import pytest
import torch

from oplm.training.parallel import _mixed_precision_policy, resolve_mesh_dims


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


def test_missing_local_world_size_warns_about_global_sharding(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The single-node assumption is warned about: on a multi-node job it shards globally.

    Sharding every parameter across every rank turns each all-gather into a cross-node
    collective -- an order-of-magnitude bandwidth cliff that is otherwise silent (nothing
    fails; the run is just slow).
    """
    with caplog.at_level(logging.WARNING, logger="oplm.training.parallel"):
        assert resolve_mesh_dims(world_size=16, local_world_size=None) == (1, 16)
    assert "LOCAL_WORLD_SIZE" in caplog.text


def test_present_local_world_size_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    """A well-formed ``LOCAL_WORLD_SIZE`` produces no warning noise."""
    with caplog.at_level(logging.WARNING, logger="oplm.training.parallel"):
        resolve_mesh_dims(world_size=16, local_world_size=8)
    assert caplog.text == ""


# --- mixed-precision policy ------------------------------------------------------


def test_bf16_policy_computes_in_bf16_and_reduces_in_fp32() -> None:
    """bf16 all-gathers parameters in bf16 but reduce-scatters gradients in fp32.

    The fp32 reduce is a deliberate divergence from Accelerate's FSDP2 default (which
    reduces in the compute dtype) -- see ``_mixed_precision_policy``'s docstring.
    """
    policy = _mixed_precision_policy("bf16")
    assert policy is not None
    assert policy.param_dtype == torch.bfloat16
    assert policy.reduce_dtype == torch.float32


def test_no_mixed_precision_leaves_the_default_policy() -> None:
    """``"no"`` returns ``None``, i.e. ``fully_shard``'s own default (native dtypes)."""
    assert _mixed_precision_policy("no") is None


def test_fp16_policy_is_refused() -> None:
    """fp16 raises here too, not only in ``TrainConfig`` -- the second line of defense."""
    with pytest.raises(ValueError, match="fp16"):
        _mixed_precision_policy("fp16")
