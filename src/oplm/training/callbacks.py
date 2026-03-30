"""Trainer callback interface for stable training-event observation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from oplm.training.trainer import Trainer


class TrainerCallback:
    """Minimal main-process callback surface exposed by :class:`Trainer`."""

    def on_train_start(self, trainer: Trainer) -> None:
        """Called once before the training loop begins."""

    def on_log(self, trainer: Trainer, metrics: dict[str, float], step: int) -> None:
        """Called whenever the trainer logs metrics through ``accelerator.log``."""

    def on_eval_end(self, trainer: Trainer, metrics: dict[str, float], step: int) -> None:
        """Called after an evaluation pass emits its aggregated metrics."""

    def on_checkpoint_saved(self, trainer: Trainer, checkpoint_dir: Path, step: int) -> None:
        """Called after a checkpoint directory has been written."""

    def on_train_end(self, trainer: Trainer) -> None:
        """Called once when the training loop finishes or exits early."""
