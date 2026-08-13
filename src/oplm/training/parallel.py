"""FSDP2/HSDP sharding for ``train.parallelism="hsdp"`` (Phase 5, spec §7).

**Architecture decision: native ``fully_shard``, not Accelerate's FSDP2 integration.**

Accelerate 1.13 *does* ship an FSDP2 path (``FullyShardedDataParallelPlugin(fsdp_version
=2)`` + ``ParallelismConfig(dp_replicate_size=..., dp_shard_size=...)``), but this module
deliberately does not use it. Under ``parallelism="hsdp"`` the model is sharded here and
**not** passed through ``accelerator.prepare``; only optimizers and schedulers are
prepared. Accelerate still provides everything else it provided before -- process-group
setup, device resolution, ``reduce``/``broadcast`` collectives, gradient-accumulation
bookkeeping, trackers -- so the trainer's control flow, rank-sync points, and checkpoint
commit protocol are byte-for-byte the same on both paths. The reasons, in the order they
decided it:

1. **Accelerate's FSDP2 never engages on CPU.** ``DistributedType.FSDP`` is only selected
   when the base distributed type is ``MULTI_GPU``/``MULTI_XPU``/etc. (``accelerate/
   state.py``); a CPU/gloo run is ``MULTI_CPU``, which falls through to plain DDP with the
   FSDP plugin silently ignored. That would make the HSDP pilot untestable in CI -- the
   e2e gate for this feature (``tests/training/test_e2e_hsdp.py``) runs 2 CPU/gloo ranks.
   Native ``fully_shard`` works on a CPU mesh (proven by the Task 0.3 Muon spike).
2. **It would reorder ``torch.compile``.** Accelerate's ``_prepare_fsdp2`` applies
   ``torch.compile`` from the dynamo plugin *before* ``fully_shard``, and only from the
   plugin -- bypassing the trainer's carefully-ordered compile block (SAC/``optimize_ddp``
   interaction, Dynamo cache-size sizing, the ``mix_order_reduction`` workaround). Keeping
   the model out of ``prepare`` leaves the existing "shard, then compile" ordering intact.
3. **It rewrites the optimizers underneath us.** ``_prepare_fsdp2`` replaces every
   optimizer's params with 1-element scratch tensors, shards, then re-maps by canonicalized
   parameter *name*, raising ``KeyError``/``ValueError`` on any name it cannot match -- a
   path with documented sharp edges for tied embeddings, which ``OplmForMaskedLM`` has
   (``lm_head.decoder.weight`` <-> the input embedding). Building the optimizers *after*
   sharding, as this module's callers do, sidesteps the remapping entirely.

The checkpoint path is unaffected by the choice either way: Phase 2 saves through
``get_state_dict``/``set_state_dict``, which handle DTensor parameters natively (spec §7),
so an HSDP checkpoint loads under DDP and vice versa.

**Known limitation (deliberate, documented):** ``Accelerator.no_sync`` looks for a
``no_sync`` attribute on the model, which ``FSDPModule`` does not define, so under
gradient accumulation every micro-batch reduce-scatters its gradients instead of only the
last one. That is *correct* (gradients accumulate into the sharded ``.grad``), just extra
communication. The fix is FSDP2's ``set_requires_gradient_sync``/``set_is_last_backward``
pair, which has to be driven per micro-batch from the training loop; it is a throughput
optimization, not a correctness fix, and is deliberately left out of Task 5.1 rather than
half-wired.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from accelerate import Accelerator
    from torch import nn
    from torch.distributed.device_mesh import DeviceMesh

logger = logging.getLogger(__name__)

# Mesh dim names, in mesh-dim order: dim 0 replicates (across nodes), dim 1 shards
# (within a node). fully_shard reads a 2-D mesh as (Replicate(), Shard(0)) -- HSDP --
# regardless of the names, which exist for readability and for the e2e's assertions.
MESH_DIM_NAMES = ("replicate", "shard")

# Compute dtype per train.mixed_precision. "fp16" is absent on purpose: TrainConfig
# rejects hsdp + fp16 (the GradScaler's inf-check is not shard-aware), and this mapping
# is the second line of defense.
_PARAM_DTYPES = {"bf16": torch.bfloat16}


def resolve_mesh_dims(world_size: int, local_world_size: int | None) -> tuple[int, int]:
    """Return the ``(replicate, shard)`` mesh dims for an HSDP run.

    Shards within a node and replicates across nodes: ``(world_size //
    local_world_size, local_world_size)``. A single-node job degrades to ``(1,
    world_size)``, i.e. plain FSDP over every rank -- the intended behavior, not a
    fallback.

    The ``LOCAL_WORLD_SIZE`` consistency requirement mirrors
    :func:`oplm.training.remote.build_upload_group`'s: the launcher's per-node rank count
    must actually divide the world size, or the mesh would not correspond to the physical
    topology (sharding across nodes and replicating within one, silently trading NVLink
    bandwidth for inter-node bandwidth). A missing or nonsensical ``LOCAL_WORLD_SIZE``
    (``None`` or ``<= 0``) is treated as "everyone is on one node", the same
    degrade-don't-guess rule ``remote._local_world_size`` uses.

    Args:
        world_size: Total number of ranks (``accelerator.num_processes``).
        local_world_size: Ranks on this node, from ``LOCAL_WORLD_SIZE``; ``None`` when the
            launcher does not export it.

    Returns:
        ``(num_nodes, gpus_per_node)`` -- the shape passed to ``init_device_mesh``.

    Raises:
        ValueError: ``world_size <= 1``. There is nothing to shard across, and silently
            degrading to a 1-rank "mesh" would hide a misconfigured launch.
        RuntimeError: ``local_world_size`` does not divide ``world_size`` (or exceeds it).
    """
    if world_size <= 1:
        raise ValueError(
            f"train.parallelism='hsdp' requires more than one process (world size is "
            f"{world_size}). FSDP2 has nothing to shard across on a single process -- use "
            "train.parallelism='ddp' (the default) for single-process runs."
        )

    if local_world_size is None or local_world_size <= 0:
        local_world_size = world_size

    if local_world_size > world_size or world_size % local_world_size != 0:
        raise RuntimeError(
            f"LOCAL_WORLD_SIZE={local_world_size} does not evenly divide WORLD_SIZE="
            f"{world_size}, so the HSDP mesh cannot match the physical topology (shard "
            "within a node, replicate across nodes). Fix the launcher's LOCAL_WORLD_SIZE, "
            "or use train.parallelism='ddp' for this topology."
        )

    return world_size // local_world_size, local_world_size


def build_hsdp_mesh(accelerator: Accelerator) -> DeviceMesh:
    """Build the 2-D ``("replicate", "shard")`` device mesh for this run.

    Args:
        accelerator: The trainer's Accelerator (supplies the world size and the device
            type the mesh's collectives run on).

    Returns:
        A ``DeviceMesh`` of shape :func:`resolve_mesh_dims`, on ``accelerator.device``'s
        device type.

    Raises:
        ValueError: Single-process run -- see :func:`resolve_mesh_dims`.
        RuntimeError: Inconsistent ``LOCAL_WORLD_SIZE`` -- see :func:`resolve_mesh_dims`.
    """
    from torch.distributed.device_mesh import init_device_mesh

    raw_local = os.environ.get("LOCAL_WORLD_SIZE")
    local_world_size = int(raw_local) if raw_local is not None and raw_local.isdigit() else None
    mesh_shape = resolve_mesh_dims(accelerator.num_processes, local_world_size)

    mesh = init_device_mesh(accelerator.device.type, mesh_shape, mesh_dim_names=MESH_DIM_NAMES)
    logger.info(
        "HSDP device mesh: %s=%d, %s=%d on device type %r",
        MESH_DIM_NAMES[0],
        mesh_shape[0],
        MESH_DIM_NAMES[1],
        mesh_shape[1],
        accelerator.device.type,
    )
    return mesh


def _mixed_precision_policy(mixed_precision: str) -> Any | None:
    """Return the ``MixedPrecisionPolicy`` for ``train.mixed_precision``, or ``None``.

    ``None`` (for ``"no"``) leaves ``fully_shard``'s own default policy in place, i.e.
    everything in the parameters' native dtype.

    Note the deliberate divergence from Accelerate's FSDP2 defaults, which reduce
    gradients in the compute dtype: this uses ``reduce_dtype=float32`` so the
    reduce-scatter accumulates in fp32 while parameters are all-gathered in bf16. That is
    the standard large-scale pretraining recipe (bf16's 8-bit mantissa loses meaningful
    gradient precision when summed over thousands of ranks) and it costs bandwidth, not
    memory: the fp32 master weights already exist either way.

    Args:
        mixed_precision: ``cfg.train.mixed_precision``.

    Raises:
        ValueError: An unsupported value under HSDP (``"fp16"``); ``TrainConfig`` already
            rejects this combination, so reaching here means the config was bypassed.
    """
    if mixed_precision == "no":
        return None

    param_dtype = _PARAM_DTYPES.get(mixed_precision)
    if param_dtype is None:
        raise ValueError(
            f"train.parallelism='hsdp' does not support mixed_precision="
            f"{mixed_precision!r}; use 'bf16' or 'no'."
        )

    from torch.distributed.fsdp import MixedPrecisionPolicy

    return MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=torch.float32)


def apply_hsdp(model: nn.Module, accelerator: Accelerator, *, mixed_precision: str) -> nn.Module:
    """Shard ``model`` in place with FSDP2 over an HSDP mesh, and return it.

    Applies ``fully_shard`` bottom-up: once per :class:`~oplm.model.transformer.OplmBlock`
    (so each block's parameters are one all-gather/reduce-scatter group, which is what
    makes the memory saving and the comm/compute overlap possible) and once on the root
    (covering the embedding, the final norm, and the MLM head, which no block claimed).

    Called *before* ``build_optimizers``: ``fully_shard`` replaces the module's parameters
    with ``DTensor`` ones, so an optimizer built earlier would hold references to tensors
    the model no longer uses. Called *after* ``gradient_checkpointing_enable``, which only
    flips per-block flags consumed inside ``OplmBlock.forward`` -- the activation
    checkpoint then wraps the sharded block's forward, matching the "apply AC before
    ``fully_shard``" ordering that Accelerate's own FSDP2 path uses.

    The model is moved to ``accelerator.device`` first, exactly as
    ``accelerator.prepare`` would have done on the DDP path -- so peak init memory is
    unchanged from DDP (a full replica per rank, briefly, before sharding). Meta-device /
    rank-0-only materialization is a separate optimization, not wired here.

    Args:
        model: The freshly constructed, unwrapped ``OplmForMaskedLM``.
        accelerator: The trainer's Accelerator.
        mixed_precision: ``cfg.train.mixed_precision``.

    Returns:
        The same module object, now an ``FSDPModule`` (``fully_shard`` mutates in place;
        the return exists so callers read as ``model = apply_hsdp(model, ...)``).

    Raises:
        RuntimeError: No ``OplmBlock`` was found, which would silently degrade to
            root-only sharding (one giant communication group, no overlap).
        ValueError: Single-process run, or an unsupported ``mixed_precision``.
    """
    from torch.distributed.fsdp import fully_shard

    from oplm.model.transformer import OplmBlock

    mesh = build_hsdp_mesh(accelerator)
    model.to(accelerator.device)

    shard_kwargs: dict[str, Any] = {"mesh": mesh}
    policy = _mixed_precision_policy(mixed_precision)
    if policy is not None:
        shard_kwargs["mp_policy"] = policy

    blocks = [module for module in model.modules() if isinstance(module, OplmBlock)]
    if not blocks:
        raise RuntimeError(
            "train.parallelism='hsdp' found no OplmBlock submodules to shard; refusing to "
            "fall back to sharding only the root module (that would put every parameter "
            "in a single communication group, negating FSDP2's memory and overlap "
            "benefits). This indicates the model's block class changed without updating "
            "oplm.training.parallel."
        )

    for block in blocks:
        fully_shard(block, **shard_kwargs)
    fully_shard(model, **shard_kwargs)

    logger.info(
        "Applied FSDP2 fully_shard to %d OplmBlock(s) + the root module (mixed_precision=%r)",
        len(blocks),
        mixed_precision,
    )
    return model
