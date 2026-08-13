"""Checkpoint saving and loading for training resumption."""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful

if TYPE_CHECKING:
    from collections.abc import Sequence

    from accelerate import Accelerator

    from oplm.config import OplmConfig

logger = logging.getLogger(__name__)

_CHECKPOINT_PREFIX = "checkpoint-"
_TMP_SUFFIX = ".tmp"
_OLD_SUFFIX = ".old"
_LATEST_POINTER_NAME = "latest"
_KEEP_MARKER_NAME = "KEEP"
_RNG_SIDECAR_PREFIX = "rng_state_"


def _unwrap_optimizer(optimizer: Any) -> Any:
    """Unwrap Accelerate's ``AcceleratedOptimizer`` to the underlying torch optimizer.

    ``AcceleratedOptimizer`` (what ``accelerator.prepare`` returns) exposes the wrapped
    ``torch.optim.Optimizer`` as ``.optimizer``. A bare optimizer that was never passed
    through ``accelerator.prepare`` (e.g. in single-process unit tests) has no such
    attribute and passes through unchanged.
    """
    return getattr(optimizer, "optimizer", optimizer)


def _unwrap_scheduler(scheduler: Any) -> Any:
    """Unwrap Accelerate's ``AcceleratedScheduler`` to the underlying LR scheduler.

    Mirrors :func:`_unwrap_optimizer`: ``AcceleratedScheduler`` exposes the wrapped
    scheduler as ``.scheduler``; a bare scheduler passes through unchanged.
    """
    return getattr(scheduler, "scheduler", scheduler)


class _ModelOptState(Stateful):
    """Model + optimizer state via ``torch.distributed.checkpoint.state_dict``.

    ``get_state_dict``/``set_state_dict`` strip DDP's ``module.`` prefix and resolve
    ``torch.compile``'s ``OptimizedModule`` wrapping automatically (verified: a compiled
    module's state dict keys come back exactly as if it had never been compiled), and
    handle FSDP2/DTensor sharded parameters uniformly. This is what makes the checkpoint
    produced by :func:`save_checkpoint` valid regardless of world size or parallelism
    strategy -- DDP today, HSDP from Phase 5 on -- without a checkpoint format change.
    """

    def __init__(self, model: torch.nn.Module, optimizers: Sequence[Any]) -> None:
        self._model = model
        self._optimizers = [_unwrap_optimizer(o) for o in optimizers]

    def state_dict(self) -> dict[str, Any]:
        model_state, optim_state = get_state_dict(self._model, self._optimizers)
        return {"model": model_state, "optimizers": optim_state}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        set_state_dict(
            self._model,
            self._optimizers,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optimizers"],
        )


def _write_rng_sidecar(tmp_dir: Path, rank: int) -> None:
    """Write this rank's python/numpy/torch RNG state to a per-rank sidecar file.

    DCP dedupes identical-valued entries written under the same key across ranks, but
    each rank's RNG state is *not* identical -- every rank advances its own generators
    independently -- so routing it through the DCP state dict would produce
    ``world_size`` non-deduplicated copies under one key instead of the intended
    per-rank values. A tiny per-rank ``torch.save`` sidecar avoids that: no collective
    is needed, and each rank only ever reads its own file back on load.
    """
    rng_state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()
    torch.save(rng_state, tmp_dir / f"{_RNG_SIDECAR_PREFIX}{rank}.pt")


def _restore_rng_sidecar(checkpoint_dir: Path, rank: int) -> None:
    """Restore this rank's RNG state from the sidecar written by :func:`_write_rng_sidecar`.

    A missing sidecar (e.g. a checkpoint saved at a different world size than the one
    loading it) is logged and skipped rather than raised: reshard-on-load RNG semantics
    are Task 2.4's concern. This minimal load-side conversion only needs same-world-size
    resume to work for Task 2.1/2.2.
    """
    sidecar_path = checkpoint_dir / f"{_RNG_SIDECAR_PREFIX}{rank}.pt"
    if not sidecar_path.is_file():
        logger.warning(
            "RNG sidecar not found for rank %d at %s; RNG state not restored", rank, sidecar_path
        )
        return

    rng_state: dict[str, Any] = torch.load(sidecar_path, weights_only=False)
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in rng_state:
        torch.cuda.set_rng_state_all(rng_state["torch_cuda"])


def save_checkpoint(
    accelerator: Accelerator,
    model: torch.nn.Module,
    optimizers: Sequence[Any],
    schedulers: Sequence[Any],
    cfg: OplmConfig,
    output_dir: str,
    global_step: int,
    epoch: int,
    samples_seen: int,
    tokens_seen: int,
    save_total_limit: int = 3,
    keep_every_n_steps: int | None = None,
    extra_state: dict[str, Any] | None = None,
    cursor: Any | None = None,
) -> Path:
    """Save a training checkpoint atomically via a tmp-dir + rename commit.

    Every rank writes into a staging directory named ``checkpoint-<step>.tmp/``. Model
    and optimizer state go through ``torch.distributed.checkpoint`` (DCP) via
    :class:`_ModelOptState`, which uses ``torch.distributed.checkpoint.state_dict`` to
    produce a parallelism- and world-size-agnostic checkpoint -- DDP's ``module.``
    prefix and ``torch.compile``'s ``OptimizedModule`` wrapping are resolved
    automatically, and FSDP2/DTensor sharded parameters are handled uniformly, which is
    what unlocks HSDP (Phase 5) and reshard-on-load (Task 2.4) without a further format
    change. Each rank also writes its own RNG state (python/numpy/torch-CPU/torch-CUDA)
    to a small ``rng_state_<rank>.pt`` sidecar via plain ``torch.save`` -- see
    :func:`_write_rng_sidecar` for why RNG state cannot go through the DCP state dict
    itself. The main process additionally writes ``trainer_state.json`` metadata, a
    re-loadable ``config.yaml``, and a HuggingFace export under ``hf/`` suitable for
    ``OplmForMaskedLM.from_pretrained``. Only after every rank has finished writing does
    the main process rename the staging directory to the final ``checkpoint-<step>/``
    name, rewrite the ``latest`` pointer file, and rotate old checkpoints. This
    guarantees a process killed mid-save leaves behind only an invisible ``.tmp``
    directory -- never a resume candidate.

    Re-saving at the same step (e.g. a requeue that saves before reaching a later step)
    never deletes the previously committed dir before the new one exists: the old
    ``checkpoint-<step>/`` is first moved aside to ``checkpoint-<step>.old/``, the ``.tmp``
    dir is renamed onto the now-free ``checkpoint-<step>/`` name, and only then is the
    ``.old`` dir removed. A kill between those two renames leaves a recoverable
    ``checkpoint-<step>.old/`` dir (restored by :func:`clean_stale_checkpoint_dirs` at next
    trainer start) rather than a gap where neither the old nor the new checkpoint survives.

    Args:
        accelerator: The HuggingFace Accelerator instance.
        model: The (possibly wrapped) model to export under ``hf/`` and checkpoint via DCP.
        optimizers: All optimizers to checkpoint, in a fixed order that ``load_checkpoint``
            must be called with too (may be Accelerate-wrapped or bare torch optimizers).
        schedulers: All LR schedulers to checkpoint, in the same order as ``optimizers``
            (may be Accelerate-wrapped or bare).
        cfg: Full OPLM configuration (serialized for reproducibility).
        output_dir: Base output directory (checkpoints saved under subdirs).
        global_step: Current global training step.
        epoch: Current epoch number.
        samples_seen: Cumulative training samples processed globally.
        tokens_seen: Cumulative training tokens processed.
        save_total_limit: Maximum number of rolling checkpoints to keep. Permanent
            checkpoints (see ``keep_every_n_steps`` and :func:`mark_permanent`) are
            excluded from both the count and deletion.
        keep_every_n_steps: When set, checkpoints whose step is a multiple of this
            value are permanent and never rotated away, regardless of
            ``save_total_limit``.
        extra_state: Additional key/value pairs merged into ``trainer_state.json``
            (e.g. the ``keep_every_n_hours`` bookkeeping keys ``first_checkpoint_unix``
            and ``last_time_keep_index``, or the ``wandb_run_id`` persisted so a resumed
            run continues the same W&B run instead of starting a new one). Merged on top
            of the base keys below, so callers must not collide with ``global_step``,
            ``epoch``, ``samples_seen``, ``tokens_seen``, or ``cursor``.
        cursor: Reserved for the data-loading cursor dataclass landing in Task 3.1
            (``oplm.training.cursor.DataCursor``); typed loosely (``Any | None``) since
            that type does not exist yet, and ``None`` for every caller until then. When
            set, its ``dataclasses.asdict`` form is recorded in ``trainer_state.json``
            under ``cursor`` for human inspection (the authoritative resumable copy will
            live wherever Task 3.1 places it).

    Returns:
        The path the checkpoint is committed to (``<output_dir>/checkpoint-<global_step>``),
        identical on every rank regardless of which rank actually performs the rename.
    """
    from oplm.config import serialize_config
    from oplm.data import get_tokenizer

    tmp_dir = Path(output_dir) / f"{_CHECKPOINT_PREFIX}{global_step}{_TMP_SUFFIX}"
    final_dir = tmp_dir.with_name(f"{_CHECKPOINT_PREFIX}{global_step}")

    # Model + optimizer state through DCP (all ranks participate; dcp.save is
    # internally collective and creates tmp_dir as part of that collective, so it
    # exists on every rank by the time this call returns -- verified single-process,
    # with torch.compile, and 2-rank DDP without requiring a pre-created directory).
    dcp_state: dict[str, Any] = {
        "app": _ModelOptState(model, optimizers),
        "schedulers": [_unwrap_scheduler(s).state_dict() for s in schedulers],
    }
    dcp.save(dcp_state, checkpoint_id=str(tmp_dir))

    # Per-rank RNG sidecar (Task 2.1 spec): not part of the DCP state dict (see
    # _write_rng_sidecar), written before the commit barrier below.
    _write_rng_sidecar(tmp_dir, accelerator.process_index)

    if accelerator.is_main_process:
        # Save trainer state
        state: dict[str, Any] = {
            "global_step": global_step,
            "epoch": epoch,
            "samples_seen": samples_seen,
            "tokens_seen": tokens_seen,
        }
        if cursor is not None:
            state["cursor"] = asdict(cursor) if is_dataclass(cursor) else cursor
        if extra_state:
            state.update(extra_state)
        state_path = tmp_dir / "trainer_state.json"
        state_path.write_text(json.dumps(state, indent=2))

        # Save config (model is the HF OplmConfig; train/data are dataclasses)
        config_path = tmp_dir / "config.yaml"
        config_path.write_text(serialize_config(cfg))

        # HuggingFace export for from_pretrained-style downstream loading
        hf_dir = tmp_dir / "hf"
        unwrapped = accelerator.unwrap_model(model)
        # torch.compile wraps the model in OptimizedModule; peel it off to reach
        # the underlying PreTrainedModel for save_pretrained.
        if hasattr(unwrapped, "_orig_mod"):
            unwrapped = unwrapped._orig_mod
        unwrapped.save_pretrained(hf_dir)  # config.json + model.safetensors
        get_tokenizer().save_pretrained(hf_dir)  # tokenizer files for round-trip

    # Barrier: every rank has finished writing into tmp_dir before anyone commits it.
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        aside_dir = final_dir.with_name(f"{final_dir.name}{_OLD_SUFFIX}")
        if final_dir.exists():
            # Re-saving at the same step after a requeue: move the previous commit
            # aside first — never delete it before the new one is renamed into place.
            # A kill between these two renames leaves a recoverable checkpoint-<N>.old/
            # dir instead of a window where neither the old nor the new dir exists.
            final_dir.rename(aside_dir)
        tmp_dir.rename(final_dir)
        if aside_dir.exists():
            shutil.rmtree(aside_dir)
        _write_latest_pointer(Path(output_dir), final_dir.name)
        _rotate_checkpoints(
            Path(output_dir), save_total_limit, keep_every_n_steps=keep_every_n_steps
        )

    # Second barrier: no rank proceeds (e.g. to a subsequent resume) until the
    # rename, latest-pointer update, and rotation on the main process are done.
    accelerator.wait_for_everyone()

    return final_dir


def load_checkpoint(
    accelerator: Accelerator,
    model: torch.nn.Module,
    optimizers: Sequence[Any],
    schedulers: Sequence[Any],
    checkpoint_dir: str,
) -> dict[str, Any]:
    """Load a training checkpoint and return trainer state metadata.

    Restores model and optimizer state via ``torch.distributed.checkpoint`` (DCP; see
    :class:`_ModelOptState`), restores each scheduler's state, and restores this rank's
    RNG state from its sidecar (see :func:`_restore_rng_sidecar`). Reads and returns the
    trainer state dict.

    This is a **minimal** conversion of the load path to the DCP format that
    :func:`save_checkpoint` (Task 2.1) now writes -- just enough to keep resume working
    end to end. Task 2.2 owns the full load rework (schema validation, world-size
    mismatch fallback, etc.); this function does not attempt any of that.

    Args:
        accelerator: The HuggingFace Accelerator instance.
        model: The (possibly wrapped) model to restore state into.
        optimizers: All optimizers to restore, in the exact order passed to the
            :func:`save_checkpoint` call that produced ``checkpoint_dir`` (may be
            Accelerate-wrapped or bare).
        schedulers: All LR schedulers to restore, in the same order as ``optimizers``
            (may be Accelerate-wrapped or bare).
        checkpoint_dir: Path to the checkpoint directory.

    Returns:
        Dict with keys ``global_step``, ``epoch``, ``tokens_seen``, and
        optionally ``samples_seen`` for backward compatibility.

    Raises:
        FileNotFoundError: If the checkpoint directory or state file is missing.
    """
    ckpt_path = Path(checkpoint_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    unwrapped_schedulers = [_unwrap_scheduler(s) for s in schedulers]
    dcp_state: dict[str, Any] = {
        "app": _ModelOptState(model, optimizers),
        "schedulers": [s.state_dict() for s in unwrapped_schedulers],
    }
    dcp.load(dcp_state, checkpoint_id=str(ckpt_path))
    for scheduler, scheduler_state in zip(
        unwrapped_schedulers, dcp_state["schedulers"], strict=True
    ):
        scheduler.load_state_dict(scheduler_state)

    _restore_rng_sidecar(ckpt_path, accelerator.process_index)

    state_path = ckpt_path / "trainer_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"trainer_state.json not found in {checkpoint_dir}")

    state: dict[str, Any] = json.loads(state_path.read_text())
    return state


def latest_checkpoint(output_dir: Path) -> Path | None:
    """Return the highest-step *committed* checkpoint under ``output_dir``, or ``None``.

    Checkpoints are named ``checkpoint-<global_step>`` (see
    :func:`save_checkpoint`), so ordering is numeric on the suffix — lexicographic ordering
    would rank ``checkpoint-9000`` above ``checkpoint-10000``. In-flight ``checkpoint-<step>.tmp``
    staging directories and ``checkpoint-<step>.old`` replace-in-progress directories are never
    committed and are always ignored (the numeric-suffix check alone excludes both, since neither
    ``<step>.tmp`` nor ``<step>.old`` is all-digit), so a process killed mid-save or mid-replace can
    never produce a resume candidate.

    Args:
        output_dir: The training output directory (may not exist yet).

    Returns:
        The path to the committed checkpoint directory with the highest numeric step, or
        ``None`` if ``output_dir`` does not exist or holds no well-formed checkpoint
        directory.
    """
    if not output_dir.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for path in output_dir.glob(f"{_CHECKPOINT_PREFIX}*"):
        if not path.is_dir() or path.name.endswith(_TMP_SUFFIX):
            continue
        suffix = path.name.removeprefix(_CHECKPOINT_PREFIX)
        if suffix.isdigit():
            candidates.append((int(suffix), path))
    if not candidates:
        return None
    return max(candidates)[1]


def clean_stale_checkpoint_dirs(output_dir: Path) -> None:
    """Resolve any torn ``.tmp``/``.old`` checkpoint dirs left by a killed-mid-save process.

    Two kinds of leftovers are possible from an interrupted :func:`save_checkpoint`:

    - ``checkpoint-<step>.tmp/``: a staging dir that never got renamed onto its final name.
      It was never committed and is unconditionally deleted — by definition torn and
      unusable (the commit rename removes the ``.tmp`` suffix in the same step that creates
      the final dir).
    - ``checkpoint-<step>.old/``: the *previous* commit at that step, moved aside during a
      same-step replace (see :func:`save_checkpoint`) and not yet cleaned up. If
      ``checkpoint-<step>`` also exists, the replace finished committing before the kill, so
      the ``.old`` dir is stale and is deleted. If ``checkpoint-<step>`` does *not* exist, the
      kill landed between the two renames — the old commit is the only surviving checkpoint
      at that step, so it is recovered by renaming ``.old`` back onto the final name.

    Call once at trainer start, on the main process only, before any resume logic runs.

    Args:
        output_dir: The training output directory (may not exist yet).
    """
    if not output_dir.is_dir():
        return

    for path in output_dir.glob(f"{_CHECKPOINT_PREFIX}*{_TMP_SUFFIX}"):
        if path.is_dir():
            logger.info("Removing stale checkpoint staging dir: %s", path)
            shutil.rmtree(path)

    for aside_dir in output_dir.glob(f"{_CHECKPOINT_PREFIX}*{_OLD_SUFFIX}"):
        if not aside_dir.is_dir():
            continue
        final_dir = aside_dir.with_name(aside_dir.name.removesuffix(_OLD_SUFFIX))
        if final_dir.exists():
            # The replace that created this .old dir finished committing before the
            # kill: the new checkpoint is in place, so the old one is truly stale.
            logger.info("Removing stale replaced-checkpoint dir: %s", aside_dir)
            shutil.rmtree(aside_dir)
        else:
            # The kill landed between the two renames: the old commit is the only
            # checkpoint that survives at this step. Recover it.
            logger.warning(
                "Recovering checkpoint %s from an interrupted same-step replace (%s)",
                final_dir,
                aside_dir,
            )
            aside_dir.rename(final_dir)


def _write_latest_pointer(output_dir: Path, checkpoint_name: str) -> None:
    """Atomically rewrite the ``latest`` pointer file to name a committed checkpoint dir.

    Args:
        output_dir: The training output directory.
        checkpoint_name: Name (not path) of the newly committed checkpoint directory.
    """
    pointer_path = output_dir / _LATEST_POINTER_NAME
    tmp_path = pointer_path.with_name(f"{_LATEST_POINTER_NAME}.tmp")
    tmp_path.write_text(f"{checkpoint_name}\n")
    os.replace(tmp_path, pointer_path)


def mark_permanent(checkpoint_dir: Path) -> None:
    """Mark a committed checkpoint directory as permanent (exempt from rotation).

    Writes a ``KEEP`` marker file inside ``checkpoint_dir``. :func:`_rotate_checkpoints`
    treats any checkpoint dir containing this marker as permanent regardless of its step,
    in addition to the ``keep_every_n_steps`` step-boundary exemption. Used by the trainer
    to implement the ``keep_every_n_hours`` retention rule (Task 1.3).

    Args:
        checkpoint_dir: Path to a committed ``checkpoint-<step>/`` directory.
    """
    (checkpoint_dir / _KEEP_MARKER_NAME).write_text("")


def _is_committed_checkpoint(d: Path) -> bool:
    """Return True if ``d`` is a committed ``checkpoint-<step>/`` directory."""
    if not d.is_dir() or d.name.endswith(_TMP_SUFFIX):
        return False
    return (
        d.name.startswith(_CHECKPOINT_PREFIX) and d.name.removeprefix(_CHECKPOINT_PREFIX).isdigit()
    )


def _is_permanent_checkpoint(d: Path, step: int, keep_every_n_steps: int | None) -> bool:
    """Return True if a committed checkpoint dir is exempt from rotation.

    A checkpoint is permanent if its step falls on the ``keep_every_n_steps`` boundary,
    or if a ``KEEP`` marker file has been written into its directory (see
    :func:`mark_permanent`).
    """
    if keep_every_n_steps is not None and step % keep_every_n_steps == 0:
        return True
    return (d / _KEEP_MARKER_NAME).exists()


def _rotate_checkpoints(
    output_dir: Path,
    save_total_limit: int,
    *,
    keep_every_n_steps: int | None = None,
) -> None:
    """Delete oldest rolling checkpoints to keep at most ``save_total_limit``.

    ``.tmp`` staging directories and ``.old`` replace-in-progress directories are ignored
    entirely: they neither count toward the limit nor are they ever deleted here (they are
    resolved separately by :func:`clean_stale_checkpoint_dirs`).

    Permanent checkpoints — those on a ``keep_every_n_steps`` step boundary, or with a
    ``KEEP`` marker written via :func:`mark_permanent` — are excluded from both the
    rolling count and deletion; only non-permanent ("rolling") checkpoints are counted
    against ``save_total_limit`` and eligible for removal.

    Args:
        output_dir: The training output directory.
        save_total_limit: Maximum number of rolling checkpoints to keep.
        keep_every_n_steps: When set, checkpoints whose step is a multiple of this
            value are permanent and excluded from rotation.
    """
    if save_total_limit <= 0:
        return

    committed = sorted(
        (d for d in output_dir.iterdir() if _is_committed_checkpoint(d)),
        key=lambda d: int(d.name.removeprefix(_CHECKPOINT_PREFIX)),
    )

    rolling_dirs = [
        d
        for d in committed
        if not _is_permanent_checkpoint(
            d, int(d.name.removeprefix(_CHECKPOINT_PREFIX)), keep_every_n_steps
        )
    ]

    while len(rolling_dirs) > save_total_limit:
        oldest = rolling_dirs.pop(0)
        logger.info("Removing old checkpoint: %s", oldest)
        shutil.rmtree(oldest)
