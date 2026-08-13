"""Checkpoint saving and loading for training resumption."""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    from accelerate import Accelerator

    from oplm.config import OplmConfig

logger = logging.getLogger(__name__)

_CHECKPOINT_PREFIX = "checkpoint-"
_TMP_SUFFIX = ".tmp"
_OLD_SUFFIX = ".old"
_LATEST_POINTER_NAME = "latest"
_KEEP_MARKER_NAME = "KEEP"


def save_checkpoint(
    accelerator: Accelerator,
    model: torch.nn.Module,
    cfg: OplmConfig,
    output_dir: str,
    global_step: int,
    epoch: int,
    samples_seen: int,
    tokens_seen: int,
    save_total_limit: int = 3,
    keep_every_n_steps: int | None = None,
    extra_state: dict[str, Any] | None = None,
) -> None:
    """Save a training checkpoint atomically via a tmp-dir + rename commit.

    Every rank writes into a staging directory named ``checkpoint-<step>.tmp/`` (via
    ``accelerator.save_state()`` for the resumable model, optimizer, scheduler, and RNG
    states; the main process additionally writes ``trainer_state.json`` metadata, a
    re-loadable ``config.yaml``, and a HuggingFace export under ``hf/`` suitable for
    ``OplmForMaskedLM.from_pretrained``). Only after every rank has finished writing does the
    main process rename the staging directory to the final ``checkpoint-<step>/`` name,
    rewrite the ``latest`` pointer file, and rotate old checkpoints. This guarantees a process
    killed mid-save leaves behind only an invisible ``.tmp`` directory — never a resume
    candidate.

    Re-saving at the same step (e.g. a requeue that saves before reaching a later step)
    never deletes the previously committed dir before the new one exists: the old
    ``checkpoint-<step>/`` is first moved aside to ``checkpoint-<step>.old/``, the ``.tmp``
    dir is renamed onto the now-free ``checkpoint-<step>/`` name, and only then is the
    ``.old`` dir removed. A kill between those two renames leaves a recoverable
    ``checkpoint-<step>.old/`` dir (restored by :func:`clean_stale_checkpoint_dirs` at next
    trainer start) rather than a gap where neither the old nor the new checkpoint survives.

    Args:
        accelerator: The HuggingFace Accelerator instance.
        model: The (possibly wrapped) model to export under ``hf/``.
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
            and ``last_time_keep_index``). Merged on top of the base keys below, so
            callers must not collide with ``global_step``, ``epoch``, ``samples_seen``,
            or ``tokens_seen``.
    """
    from oplm.config import serialize_config
    from oplm.data import get_tokenizer

    tmp_dir = Path(output_dir) / f"{_CHECKPOINT_PREFIX}{global_step}{_TMP_SUFFIX}"
    accelerator.save_state(str(tmp_dir))

    if accelerator.is_main_process:
        # Save trainer state
        state = {
            "global_step": global_step,
            "epoch": epoch,
            "samples_seen": samples_seen,
            "tokens_seen": tokens_seen,
        }
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
        final_dir = tmp_dir.with_name(f"{_CHECKPOINT_PREFIX}{global_step}")
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


def load_checkpoint(
    accelerator: Accelerator,
    checkpoint_dir: str,
) -> dict[str, Any]:
    """Load a training checkpoint and return trainer state metadata.

    Calls ``accelerator.load_state()`` to restore model, optimizer, scheduler,
    and RNG states. Reads and returns the trainer state dict.

    Args:
        accelerator: The HuggingFace Accelerator instance.
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

    accelerator.load_state(str(ckpt_path))

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
