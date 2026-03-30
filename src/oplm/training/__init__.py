"""Training infrastructure for OPLM."""

from __future__ import annotations

from oplm.training.callbacks import TrainerCallback
from oplm.training.trainer import Trainer

__all__ = ["Trainer", "TrainerCallback"]
