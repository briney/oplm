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
from torch.distributed.checkpoint.filesystem import FileSystemReader
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
_SCALER_SIDECAR_NAME = "scaler.pt"

# Escape hatch for _restore_rng_sidecar: set to "1" to start this rank's RNG fresh
# instead of hard-erroring when its sidecar is missing (e.g. a deliberate world-size
# change across a resume). See _restore_rng_sidecar for the full rationale.
_ALLOW_MISSING_RNG_ENV = "OPLM_ALLOW_MISSING_RNG"

# Schedule-related TrainConfig fields compared for exact equality by
# validate_schedule_compat. max_steps is deliberately EXCLUDED from this tuple -- it
# gets its own asymmetric policy (see validate_schedule_compat's docstring) because,
# unlike these fields, deliberately *increasing* it across a resume to keep training
# past the original target is an existing, tested, desired workflow (both explicit
# resume_from and auto_resume), not the accidental-drift hazard this check exists to
# catch. Also not exhaustive of everything that affects the LR trajectory: e.g.
# mixed_precision does not belong here at all.
_SCHEDULE_COMPAT_FIELDS = ("warmup_steps", "stable_steps", "scheduler", "lr", "min_lr")


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

    A missing sidecar is a **hard error** by default: silently skipping RNG restore was
    a swallowed-failure hole (a resumed run would quietly diverge from what it would have
    produced without the interruption, with no signal that anything was wrong). The
    likely cause is a world-size change across the resume -- a checkpoint saved with N
    ranks only has sidecars ``rng_state_0.pt``..``rng_state_<N-1>.pt`` -- since
    reshard-on-load RNG semantics are Task 2.4's concern, not this one.

    The single escape hatch is the ``OPLM_ALLOW_MISSING_RNG=1`` environment variable: when
    set, a missing sidecar is logged and skipped (this rank's RNG starts fresh) instead of
    raising. There is deliberately no config-level bypass -- an explicit ``train.
    resume_from`` does not relax this check, so an operator has to opt in explicitly and
    visibly (an env var, not a config field that could be committed and forgotten).
    """
    sidecar_path = checkpoint_dir / f"{_RNG_SIDECAR_PREFIX}{rank}.pt"
    if not sidecar_path.is_file():
        if os.environ.get(_ALLOW_MISSING_RNG_ENV) == "1":
            logger.warning(
                "RNG sidecar not found for rank %d at %s; %s=1 is set, so this rank's RNG "
                "state is starting fresh instead of being restored.",
                rank,
                sidecar_path,
                _ALLOW_MISSING_RNG_ENV,
            )
            return
        raise RuntimeError(
            f"RNG sidecar not found for rank {rank} at {sidecar_path}. This usually means "
            "the checkpoint was saved at a different world size than the one resuming "
            "(each rank's RNG state lives in its own rng_state_<rank>.pt sidecar, so a "
            "checkpoint saved with N ranks only has sidecars 0..N-1). To resume anyway and "
            f"start this rank's RNG fresh, set the environment variable "
            f"{_ALLOW_MISSING_RNG_ENV}=1."
        )

    rng_state: dict[str, Any] = torch.load(sidecar_path, weights_only=False)
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in rng_state:
        torch.cuda.set_rng_state_all(rng_state["torch_cuda"])


def _write_scaler_sidecar(tmp_dir: Path, accelerator: Accelerator) -> None:
    """Save the fp16 ``GradScaler``'s state to a ``scaler.pt`` sidecar, main process only.

    ``accelerator.save_state`` used to serialize this (Phase 1); the DCP rewrite dropped
    it with no replacement, silently losing scale/growth-tracker state on every fp16
    resume. Mirrors Accelerate's own approach (``accelerate.checkpointing.
    save_accelerator_state``): a single shared file, not a per-rank sidecar, because
    Accelerate keeps every rank's scaler in sync -- there is exactly one scale value to
    persist, so only the main process needs to write it.

    A no-op when ``accelerator.scaler`` is ``None`` (bf16/no mixed precision, the common
    case for this codebase -- see docs/MUP.md on why fp16 is not the default).
    """
    if accelerator.scaler is None:
        return
    torch.save(accelerator.scaler.state_dict(), tmp_dir / _SCALER_SIDECAR_NAME)


def _restore_scaler_sidecar(checkpoint_dir: Path, accelerator: Accelerator) -> None:
    """Restore the fp16 ``GradScaler``'s state from the ``scaler.pt`` sidecar, every rank.

    Mirrors :func:`_write_scaler_sidecar`. Restored on every rank (not main-only) because
    every rank has its own ``accelerator.scaler`` instance driving its own ``backward()``
    calls, exactly as Accelerate's own ``load_accelerator_state`` does.

    A checkpoint/run mismatch (one has a scaler, the other doesn't) is logged and
    skipped rather than raised -- e.g. resuming a bf16 run from a checkpoint saved under
    fp16, or vice versa, is a config change the operator made deliberately.
    """
    sidecar_path = checkpoint_dir / _SCALER_SIDECAR_NAME
    has_sidecar = sidecar_path.is_file()
    has_scaler = accelerator.scaler is not None

    if has_sidecar and has_scaler:
        scaler_state: dict[str, Any] = torch.load(sidecar_path, weights_only=False)
        accelerator.scaler.load_state_dict(scaler_state)
    elif has_sidecar and not has_scaler:
        logger.warning(
            "Checkpoint %s has fp16 GradScaler state but the current run has no scaler "
            "(mixed_precision is not fp16); scaler state not restored.",
            sidecar_path,
        )
    elif has_scaler and not has_sidecar:
        logger.warning(
            "Current run has an fp16 GradScaler but checkpoint %s has no scaler.pt "
            "sidecar; scaler starting from its default state.",
            checkpoint_dir,
        )


def save_checkpoint(
    accelerator: Accelerator,
    model: torch.nn.Module,
    optimizers: Sequence[Any],
    schedulers: Sequence[Any],
    cfg: OplmConfig,
    output_dir: str,
    *,
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
    itself. The main process additionally writes the fp16 ``GradScaler``'s state (when
    ``accelerator.scaler is not None``) to a ``scaler.pt`` sidecar (see
    :func:`_write_scaler_sidecar` -- this replaces what ``accelerator.save_state`` used
    to serialize for that state), ``trainer_state.json`` metadata, a re-loadable
    ``config.yaml``, and a HuggingFace export under ``hf/`` suitable for
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
        # fp16 GradScaler state (no-op when accelerator.scaler is None, e.g. bf16/no
        # mixed precision). See _write_scaler_sidecar docstring.
        _write_scaler_sidecar(tmp_dir, accelerator)

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


def validate_schedule_compat(
    checkpoint_dir: Path,
    cfg: OplmConfig,
    checkpoint_global_step: int | None = None,
) -> None:
    """Raise if the checkpoint's LR-schedule config disagrees with the live config.

    Resuming into a scheduler rebuilt with different ``warmup_steps``/``stable_steps``/
    ``scheduler``/``lr``/``min_lr`` silently reshapes the LR trajectory --
    ``torch.optim.lr_scheduler.LambdaLR.load_state_dict`` only restores ``last_epoch`` and
    a couple of counters, not the lambda itself, so a scheduler rebuilt with a different
    ``warmup_steps`` (say) resumes at the *same step index* but a *different point on the
    curve*, with no error. Comparing the checkpoint's own ``config.yaml`` against the live
    config before ``dcp.load`` even runs closes that hole with a loud, specific error
    instead of a silent LR discontinuity discovered days later in a wandb chart.
    ``_SCHEDULE_COMPAT_FIELDS`` (everything above) is checked for exact equality.

    ``max_steps`` gets its own **asymmetric** policy instead of exact equality, because
    unlike the fields above, deliberately increasing it across a resume -- to keep
    training past the original target, reshaping the *remaining* portion of the curve to
    reach the new, longer target -- is an existing, tested, desired workflow for both
    explicit ``resume_from`` and ``auto_resume`` (e.g.
    ``test_resume_restores_state_and_continues``,
    ``test_auto_resume_picks_up_the_newest_committed_checkpoint``):

      - ``live max_steps < checkpoint max_steps`` (any decrease): raises. A shrunk
        schedule on resume is essentially always accidental, and the checkpoint's own
        ``global_step`` can even already exceed the new, smaller total.
      - ``live max_steps <= checkpoint_global_step`` (resuming into a target already
        met or passed by the checkpoint's own recorded progress): raises, independent of
        whether ``max_steps`` moved -- there would be nothing left to train. Skipped
        when ``checkpoint_global_step`` is unknown (``None``).
      - ``live max_steps > checkpoint max_steps`` (increase, and not already caught by
        the ``global_step`` check above): allowed, but logged as a prominent warning --
        the decay portion of the schedule will differ from an uninterrupted run to the
        new target, which is expected for a deliberate run extension, not silently
        swallowed.
      - Equal: silent.

    ``max_epochs``-derived ``total_steps`` changes are not covered at all -- only the
    ``max_steps`` config field itself is compared (out of scope here -- see the task
    brief: "don't over-engineer").

    Args:
        checkpoint_dir: Path to the committed checkpoint directory (holds ``config.yaml``).
        cfg: The live, resolved config being trained/resumed with.
        checkpoint_global_step: The checkpoint's own recorded ``global_step`` (from its
            ``trainer_state.json``), used only for the ``max_steps``-vs-``global_step``
            check above. ``None`` skips that specific check (e.g. when the caller hasn't
            read ``trainer_state.json`` yet).

    Raises:
        ValueError: A field in ``_SCHEDULE_COMPAT_FIELDS`` differs, ``max_steps``
            decreased, or ``max_steps`` no longer leaves any training to do; the message
            names every differing/invalid field with both values.
    """
    config_path = checkpoint_dir / "config.yaml"
    if not config_path.is_file():
        # No config.yaml to compare against (e.g. a pre-Task-1 checkpoint, or one
        # written by a caller that skipped it). Nothing to validate against; proceed
        # rather than block resume on missing provenance.
        logger.warning(
            "Checkpoint %s has no config.yaml; skipping schedule-compatibility validation.",
            checkpoint_dir,
        )
        return

    from oplm.config import load_config

    checkpoint_cfg = load_config(["--config", str(config_path)])

    mismatches = [
        f"{field} (checkpoint={getattr(checkpoint_cfg.train, field)!r}, "
        f"live={getattr(cfg.train, field)!r})"
        for field in _SCHEDULE_COMPAT_FIELDS
        if getattr(checkpoint_cfg.train, field) != getattr(cfg.train, field)
    ]

    checkpoint_max_steps = checkpoint_cfg.train.max_steps
    live_max_steps = cfg.train.max_steps
    if live_max_steps < checkpoint_max_steps:
        mismatches.append(
            f"max_steps (checkpoint={checkpoint_max_steps!r}, live={live_max_steps!r}) "
            "-- live max_steps is smaller than the checkpoint's; a shrunk schedule on "
            "resume is essentially always accidental"
        )
    elif checkpoint_global_step is not None and live_max_steps <= checkpoint_global_step:
        mismatches.append(
            f"max_steps (checkpoint={checkpoint_max_steps!r}, live={live_max_steps!r}) "
            f"<= the checkpoint's own global_step ({checkpoint_global_step!r}) -- "
            "resuming would already be past the end of training"
        )
    elif live_max_steps > checkpoint_max_steps:
        logger.warning(
            "Checkpoint %s: resuming with a larger max_steps (checkpoint=%d, live=%d). "
            "The decay portion of the LR schedule will differ from an uninterrupted run "
            "to the new, longer target -- expected for a deliberate run extension.",
            checkpoint_dir,
            checkpoint_max_steps,
            live_max_steps,
        )

    if mismatches:
        raise ValueError(
            f"Checkpoint {checkpoint_dir} is not schedule-compatible with the current "
            f"config -- resuming would silently reshape the LR schedule. Mismatched "
            f"field(s): {'; '.join(mismatches)}"
        )


def validate_checkpoint_for_resume(checkpoint_dir: Path, cfg: OplmConfig) -> None:
    """Cheaply validate a checkpoint before it is used as a (broadcast) auto-resume target.

    ``dcp.load`` is a collective: every rank must call it together, and a corrupt shard
    can raise on all ranks (a torn ``.metadata``, read during the collective's own
    metadata exchange) or in principle just one, which is the exact hang scenario
    rank-synchronization discipline exists to avoid. Rather than let the auto-resume path
    call ``dcp.load`` speculatively and retry a different candidate on failure -- which
    would need a *second* rank-agreed decision mid-recovery -- this function runs
    everything that is cheap to check *without* the collective, on the main process only,
    before the (single) resume target is broadcast to every rank (see
    ``oplm.training.trainer._resolve_resume_target``). A torn/invalid candidate is
    therefore rejected identically on every rank, and the fallback candidate chosen in its
    place is what gets broadcast -- never a candidate resolved after ranks disagree.

    Checks, in order (first failure wins):
      - ``trainer_state.json`` exists and parses as JSON.
      - ``.metadata`` exists and is readable as DCP metadata (catches a truncated file
        left by a kill during ``dcp.save``'s own write, distinct from the tmp-dir commit
        rename ``save_checkpoint`` otherwise protects against).
      - The checkpoint is schedule-compatible with ``cfg`` (see
        :func:`validate_schedule_compat`).

    A genuine mid-``dcp.load`` failure *after* this validation has passed (e.g. corruption
    that these cheap checks don't catch) is intentionally left fatal -- see
    ``oplm.training.trainer._resolve_resume_target``'s docstring for why.

    Args:
        checkpoint_dir: Path to a committed ``checkpoint-<step>/`` directory.
        cfg: The live, resolved config being trained/resumed with.

    Raises:
        FileNotFoundError: ``trainer_state.json`` or ``.metadata`` is missing.
        json.JSONDecodeError: ``trainer_state.json`` exists but is not valid JSON.
        ValueError: The checkpoint is not schedule-compatible with ``cfg``.
        Exception: Any other error ``FileSystemReader.read_metadata`` raises for a
            corrupted/truncated ``.metadata`` file (the exact type is an internal detail
            of DCP's metadata deserialization, not part of this function's contract).
    """
    state_path = checkpoint_dir / "trainer_state.json"
    state: dict[str, Any] = json.loads(state_path.read_text())

    FileSystemReader(str(checkpoint_dir)).read_metadata()

    validate_schedule_compat(checkpoint_dir, cfg, checkpoint_global_step=state.get("global_step"))


def load_checkpoint(
    accelerator: Accelerator,
    model: torch.nn.Module,
    optimizers: Sequence[Any],
    schedulers: Sequence[Any],
    checkpoint_dir: str,
    cfg: OplmConfig,
) -> dict[str, Any]:
    """Load a training checkpoint and return trainer state metadata.

    Validates that the checkpoint is schedule-compatible with ``cfg`` (see
    :func:`validate_schedule_compat`) before touching any state, then restores model and
    optimizer state via ``torch.distributed.checkpoint`` (DCP; see :class:`_ModelOptState`),
    restores each scheduler's state, restores this rank's RNG state from its sidecar (a
    missing sidecar is a hard error -- see :func:`_restore_rng_sidecar`), and restores the
    fp16 ``GradScaler``'s state (see :func:`_restore_scaler_sidecar`) when both the
    checkpoint and the current run have one. Reads and returns the trainer state dict.

    Args:
        accelerator: The HuggingFace Accelerator instance.
        model: The (possibly wrapped) model to restore state into.
        optimizers: All optimizers to restore, in the exact order passed to the
            :func:`save_checkpoint` call that produced ``checkpoint_dir`` (may be
            Accelerate-wrapped or bare).
        schedulers: All LR schedulers to restore, in the same order as ``optimizers``
            (may be Accelerate-wrapped or bare).
        checkpoint_dir: Path to the checkpoint directory.
        cfg: The live, resolved config being resumed with; compared against the
            checkpoint's own ``config.yaml`` for schedule compatibility.

    Returns:
        Dict with keys ``global_step``, ``epoch``, ``tokens_seen``, and
        optionally ``samples_seen`` for backward compatibility.

    Raises:
        FileNotFoundError: If the checkpoint directory or state file is missing.
        ValueError: If the checkpoint's LR schedule config disagrees with ``cfg`` (see
            :func:`validate_schedule_compat`).
        RuntimeError: If this rank's RNG sidecar is missing and
            ``OPLM_ALLOW_MISSING_RNG=1`` is not set (see :func:`_restore_rng_sidecar`).
    """
    ckpt_path = Path(checkpoint_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    # Read trainer_state.json up front (rather than after dcp.load, as in Task 2.1's
    # minimal conversion): validate_schedule_compat's max_steps-vs-global_step check
    # needs the checkpoint's own global_step, and failing fast on a missing/corrupt
    # state file before touching model/optimizer state is strictly better ordering
    # anyway. Reused for the return value below instead of re-reading.
    state_path = ckpt_path / "trainer_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"trainer_state.json not found in {checkpoint_dir}")
    state: dict[str, Any] = json.loads(state_path.read_text())

    validate_schedule_compat(ckpt_path, cfg, checkpoint_global_step=state.get("global_step"))

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
    _restore_scaler_sidecar(ckpt_path, accelerator)

    return state


def list_committed_checkpoints(output_dir: Path) -> list[Path]:
    """Return every *committed* checkpoint under ``output_dir``, newest step first.

    Checkpoints are named ``checkpoint-<global_step>`` (see :func:`save_checkpoint`), so
    ordering is numeric on the suffix — lexicographic ordering would rank
    ``checkpoint-9000`` above ``checkpoint-10000``. In-flight ``checkpoint-<step>.tmp``
    staging directories and ``checkpoint-<step>.old`` replace-in-progress directories are
    never committed and are always excluded (the numeric-suffix check alone excludes
    both, since neither ``<step>.tmp`` nor ``<step>.old`` is all-digit), so a process
    killed mid-save or mid-replace can never produce a resume candidate.

    Shared discovery machinery behind :func:`latest_checkpoint` (index 0 of this list)
    and :func:`nth_latest_checkpoint` (auto-resume's committed-fallback chain).

    Args:
        output_dir: The training output directory (may not exist yet).

    Returns:
        Committed checkpoint directories sorted by step, newest first. Empty if
        ``output_dir`` does not exist or holds no well-formed checkpoint directory.
    """
    if not output_dir.is_dir():
        return []
    candidates: list[tuple[int, Path]] = []
    for path in output_dir.glob(f"{_CHECKPOINT_PREFIX}*"):
        if not path.is_dir() or path.name.endswith(_TMP_SUFFIX):
            continue
        suffix = path.name.removeprefix(_CHECKPOINT_PREFIX)
        if suffix.isdigit():
            candidates.append((int(suffix), path))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [path for _, path in candidates]


def latest_checkpoint(output_dir: Path) -> Path | None:
    """Return the highest-step *committed* checkpoint under ``output_dir``, or ``None``.

    Args:
        output_dir: The training output directory (may not exist yet).

    Returns:
        The path to the committed checkpoint directory with the highest numeric step, or
        ``None`` if ``output_dir`` does not exist or holds no well-formed checkpoint
        directory.
    """
    checkpoints = list_committed_checkpoints(output_dir)
    return checkpoints[0] if checkpoints else None


def nth_latest_checkpoint(output_dir: Path, n: int) -> Path | None:
    """Return the ``n``-th newest *committed* checkpoint under ``output_dir`` (0 = latest).

    Used by the trainer's auto-resume fallback (Task 2.2) to walk backward through
    committed checkpoints when the newest one fails pre-load validation (see
    ``oplm.training.checkpoint.validate_checkpoint_for_resume``): ``n=0`` is
    :func:`latest_checkpoint`, ``n=1`` is the next-newest, and so on.

    Args:
        output_dir: The training output directory (may not exist yet).
        n: Zero-based rank from the newest checkpoint (0 = newest).

    Returns:
        The ``n``-th newest committed checkpoint directory, or ``None`` if fewer than
        ``n + 1`` committed checkpoints exist.
    """
    checkpoints = list_committed_checkpoints(output_dir)
    return checkpoints[n] if 0 <= n < len(checkpoints) else None


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
