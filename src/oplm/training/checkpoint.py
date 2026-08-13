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
_LATEST_POINTER_NAME = "latest"


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

    Args:
        accelerator: The HuggingFace Accelerator instance.
        model: The (possibly wrapped) model to export under ``hf/``.
        cfg: Full OPLM configuration (serialized for reproducibility).
        output_dir: Base output directory (checkpoints saved under subdirs).
        global_step: Current global training step.
        epoch: Current epoch number.
        samples_seen: Cumulative training samples processed globally.
        tokens_seen: Cumulative training tokens processed.
        save_total_limit: Maximum number of checkpoints to keep.
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
        if final_dir.exists():
            # Re-saving at the same step after a requeue: replace atomically-enough —
            # the old committed dir is only removed once the new one is fully staged.
            shutil.rmtree(final_dir)
        tmp_dir.rename(final_dir)
        _write_latest_pointer(Path(output_dir), final_dir.name)
        _rotate_checkpoints(Path(output_dir), save_total_limit)

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
    staging directories are never committed and are always ignored, so a process killed
    mid-save can never produce a resume candidate.

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


def clean_stale_tmp_checkpoints(output_dir: Path) -> None:
    """Delete any ``checkpoint-*.tmp`` staging dirs left behind by a killed-mid-save process.

    By definition, a ``.tmp`` directory that survives past process exit was never committed
    (the commit rename removes the ``.tmp`` name in the same step that creates the final
    dir), so it is torn and unusable. Call once at trainer start, on the main process only,
    before any resume logic runs.

    Args:
        output_dir: The training output directory (may not exist yet).
    """
    if not output_dir.is_dir():
        return
    for path in output_dir.glob(f"{_CHECKPOINT_PREFIX}*{_TMP_SUFFIX}"):
        if path.is_dir():
            logger.info("Removing stale checkpoint staging dir: %s", path)
            shutil.rmtree(path)


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


def _rotate_checkpoints(output_dir: Path, save_total_limit: int) -> None:
    """Delete oldest committed checkpoints to keep at most ``save_total_limit``.

    ``.tmp`` staging directories are ignored entirely: they neither count toward the limit
    nor are they ever deleted here (a killed-mid-save ``.tmp`` dir is cleaned up separately
    by :func:`clean_stale_tmp_checkpoints`).
    """
    if save_total_limit <= 0:
        return

    def _is_committed_checkpoint(d: Path) -> bool:
        if not d.is_dir() or d.name.endswith(_TMP_SUFFIX):
            return False
        return (
            d.name.startswith(_CHECKPOINT_PREFIX)
            and d.name.removeprefix(_CHECKPOINT_PREFIX).isdigit()
        )

    checkpoint_dirs = sorted(
        (d for d in output_dir.iterdir() if _is_committed_checkpoint(d)),
        key=lambda d: int(d.name.removeprefix(_CHECKPOINT_PREFIX)),
    )

    while len(checkpoint_dirs) > save_total_limit:
        oldest = checkpoint_dirs.pop(0)
        logger.info("Removing old checkpoint: %s", oldest)
        shutil.rmtree(oldest)
