"""Checkpoint saving and loading for training resumption."""

from __future__ import annotations

import json
import logging
import shutil
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omegaconf import OmegaConf

if TYPE_CHECKING:
    from accelerate import Accelerator

    from oplm.config import OplmConfig

logger = logging.getLogger(__name__)


def save_checkpoint(
    accelerator: Accelerator,
    cfg: OplmConfig,
    output_dir: str,
    global_step: int,
    epoch: int,
    samples_seen: int,
    tokens_seen: int,
    save_total_limit: int = 3,
) -> None:
    """Save a training checkpoint.

    Uses ``accelerator.save_state()`` for model, optimizer, scheduler, and RNG
    states. Writes ``trainer_state.json`` and ``config.yaml`` alongside.
    Rotates old checkpoints to respect ``save_total_limit``.

    Args:
        accelerator: The HuggingFace Accelerator instance.
        cfg: Full OPLM configuration (frozen copy saved for reproducibility).
        output_dir: Base output directory (checkpoints saved under subdirs).
        global_step: Current global training step.
        epoch: Current epoch number.
        samples_seen: Cumulative training samples processed globally.
        tokens_seen: Cumulative training tokens processed.
        save_total_limit: Maximum number of checkpoints to keep.
    """
    checkpoint_dir = Path(output_dir) / f"checkpoint-{global_step}"
    accelerator.save_state(str(checkpoint_dir))

    if accelerator.is_main_process:
        # Save trainer state
        state = {
            "global_step": global_step,
            "epoch": epoch,
            "samples_seen": samples_seen,
            "tokens_seen": tokens_seen,
        }
        state_path = checkpoint_dir / "trainer_state.json"
        state_path.write_text(json.dumps(state, indent=2))

        # Save frozen config
        cfg_config = OmegaConf.to_container(OmegaConf.structured(deepcopy(cfg)), resolve=True)
        config_path = checkpoint_dir / "config.yaml"
        config_path.write_text(OmegaConf.to_yaml(OmegaConf.create(cfg_config)))

        # Rotate old checkpoints
        _rotate_checkpoints(Path(output_dir), save_total_limit)

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
