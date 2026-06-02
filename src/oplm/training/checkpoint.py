"""Checkpoint saving and loading via PyTorch Distributed Checkpoint (DCP).

The resumable training state (the FSDP2-sharded model and the primary optimizer)
is written with ``torch.distributed.checkpoint`` so it round-trips correctly
across ranks regardless of the shard layout. On rank 0 only, this module also
writes ``trainer_state.json`` metadata, a re-loadable ``config.yaml``, and a
``from_pretrained``-loadable HuggingFace export under ``hf/``. The HF export is
produced from a *full* (unsharded, CPU-offloaded) model state dict, so it is a
plain ``model.safetensors`` independent of the FSDP2 sharding.

This module has no Accelerate dependency. It assumes a process group is already
initialized — the Trainer calls ``dist.init_process_group`` before any
checkpointing, which holds for single-GPU runs launched via ``torchrun`` too.

.. note::
   Checkpoints written here are **not** compatible with the legacy Accelerate
   ``save_state`` format. The ``hf/`` export remains loadable for inference via
   :meth:`OplmForMaskedLM.from_pretrained` regardless; only the trainer-state
   resume path is format-specific.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omegaconf import OmegaConf

if TYPE_CHECKING:
    import torch
    from torch import nn

    from oplm.config import OplmConfig

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: OplmConfig,
    output_dir: str,
    global_step: int,
    epoch: int,
    samples_seen: int,
    tokens_seen: int,
    save_total_limit: int = 3,
) -> None:
    """Save a resumable training checkpoint with PyTorch Distributed Checkpoint.

    All ranks participate in the DCP save of the sharded model and primary
    optimizer state, and in the collective gather of the full model state dict.
    On rank 0 only, this then writes ``trainer_state.json`` metadata, a
    re-loadable ``config.yaml``, and a HuggingFace export under ``hf/`` (model
    weights + tokenizer) suitable for :meth:`OplmForMaskedLM.from_pretrained`,
    and rotates old checkpoints to respect ``save_total_limit``.

    Only the *primary* optimizer's state is checkpointed; this mirrors the
    :func:`load_checkpoint` resume path, which restores that single optimizer.

    Args:
        model: The (FSDP2-sharded, possibly FP8-converted / compiled) model.
        optimizer: The primary optimizer whose state is saved alongside the model.
        cfg: Full OPLM configuration (serialized for reproducibility).
        output_dir: Base output directory (checkpoints saved under subdirs).
        global_step: Current global training step.
        epoch: Current epoch number.
        samples_seen: Cumulative training samples processed globally.
        tokens_seen: Cumulative training tokens processed.
        save_total_limit: Maximum number of checkpoints to keep.
    """
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        get_state_dict,
    )

    ckpt_path = Path(output_dir) / f"checkpoint-{global_step}"

    # Collective: every rank contributes its shard of the model + optimizer state.
    model_sd, optim_sd = get_state_dict(model, optimizer)
    dcp.save({"model": model_sd, "optimizer": optim_sd}, checkpoint_id=str(ckpt_path))

    # Collective: gather the full, unsharded, CPU-offloaded model weights for the HF
    # export. full_state_dict=True all-gathers every sharded DTensor, so this MUST run
    # on all ranks (calling it inside the rank-0 guard would deadlock); only rank 0
    # then writes the files. Non-zero ranks discard their copy.
    full_sd = get_model_state_dict(
        model,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )

    if dist.get_rank() == 0:
        _write_trainer_state(ckpt_path, global_step, epoch, samples_seen, tokens_seen)
        _write_config_yaml(ckpt_path, cfg)
        _save_hf_export(full_sd, cfg, ckpt_path / "hf")
        _rotate_checkpoints(Path(output_dir), save_total_limit)

    dist.barrier()


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: str,
) -> dict[str, Any]:
    """Load a DCP training checkpoint in place and return trainer-state metadata.

    Restores the sharded model and primary optimizer state via
    ``torch.distributed.checkpoint``, then reads ``trainer_state.json`` on rank 0
    and broadcasts it so every rank returns identical resume metadata.

    Args:
        model: The (FSDP2-sharded) model to restore in place.
        optimizer: The primary optimizer to restore in place.
        checkpoint_dir: Path to the checkpoint directory (``checkpoint-N``).

    Returns:
        Dict with keys ``global_step``, ``epoch``, ``tokens_seen``, and
        (for checkpoints written by this module) ``samples_seen``.

    Raises:
        FileNotFoundError: If the checkpoint directory or ``trainer_state.json``
            is missing.
    """
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    ckpt_path = Path(checkpoint_dir)
    # Validate on every rank before any collective so a missing checkpoint raises
    # uniformly instead of deadlocking the ranks that did find it.
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    state_path = ckpt_path / "trainer_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"trainer_state.json not found in {checkpoint_dir}")

    # get_state_dict yields the current (correctly-sharded) structure; dcp.load fills
    # it from disk, and set_state_dict applies it back to the model and optimizer.
    model_sd, optim_sd = get_state_dict(model, optimizer)
    dcp.load({"model": model_sd, "optimizer": optim_sd}, checkpoint_id=str(ckpt_path))
    set_state_dict(model, optimizer, model_state_dict=model_sd, optim_state_dict=optim_sd)

    # trainer_state.json is tiny: rank 0 reads it and broadcasts so the resume state
    # is rank-identical even if the file is not visible on every node's filesystem.
    state_obj: list[dict[str, Any] | None] = [None]
    if dist.get_rank() == 0:
        state_obj[0] = json.loads(state_path.read_text())
    dist.broadcast_object_list(state_obj, src=0)

    state = state_obj[0]
    if state is None:  # pragma: no cover - broadcast always populates from rank 0
        raise RuntimeError("Failed to broadcast trainer state from rank 0")
    return state


def _write_trainer_state(
    ckpt_path: Path,
    global_step: int,
    epoch: int,
    samples_seen: int,
    tokens_seen: int,
) -> None:
    """Write the resumable trainer-state metadata as ``trainer_state.json``."""
    state = {
        "global_step": global_step,
        "epoch": epoch,
        "samples_seen": samples_seen,
        "tokens_seen": tokens_seen,
    }
    (ckpt_path / "trainer_state.json").write_text(json.dumps(state, indent=2))


def _write_config_yaml(ckpt_path: Path, cfg: OplmConfig) -> None:
    """Write a re-loadable ``config.yaml`` (model + train + data) for reproducibility."""
    from dataclasses import asdict

    config_dict = {
        "model": cfg.model.to_dict(),
        "train": asdict(cfg.train),
        "data": asdict(cfg.data),
    }
    (ckpt_path / "config.yaml").write_text(OmegaConf.to_yaml(OmegaConf.create(config_dict)))


def _save_hf_export(full_sd: dict[str, Any], cfg: OplmConfig, hf_dir: Path) -> None:
    """Write a ``from_pretrained``-loadable HF export from a full model state dict.

    Builds a fresh CPU-resident :class:`OplmForMaskedLM` (matching the training
    architecture), loads the gathered full state dict into it, then calls
    ``save_pretrained`` plus the tokenizer export. Operating on a fresh copy keeps
    the live FSDP2-sharded model untouched and sidesteps Accelerate's
    ``unwrap_model`` utility — the gathered ``full_sd`` already has clean,
    wrapper-free keys (no ``_orig_mod.`` / FSDP prefixes).

    Args:
        full_sd: Full (unsharded) model state dict gathered on rank 0.
        cfg: Full OPLM configuration; ``cfg.model`` is the HF model config.
        hf_dir: Destination directory for ``config.json`` + ``model.safetensors``.
    """
    from oplm.data import get_tokenizer
    from oplm.model import OplmForMaskedLM

    export_model = OplmForMaskedLM(cfg.model)
    export_model.load_state_dict(full_sd, strict=True)
    export_model.save_pretrained(hf_dir)  # config.json + model.safetensors
    get_tokenizer().save_pretrained(hf_dir)  # tokenizer files for round-trip


def _rotate_checkpoints(output_dir: Path, save_total_limit: int) -> None:
    """Delete oldest checkpoints to keep at most ``save_total_limit``."""
    if save_total_limit <= 0:
        return

    checkpoint_dirs = sorted(
        (d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")),
        key=lambda d: int(d.name.split("-", 1)[1]),
    )

    while len(checkpoint_dirs) > save_total_limit:
        oldest = checkpoint_dirs.pop(0)
        logger.info("Removing old checkpoint: %s", oldest)
        shutil.rmtree(oldest)
