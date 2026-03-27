"""Optimizer and learning rate scheduler construction."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch.optim.lr_scheduler import LambdaLR

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import nn

    from oplm.config import TrainConfig


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.AdamW:
    """Build AdamW optimizer with weight decay parameter grouping.

    Parameters that should not receive weight decay:
    - 1D tensors (norms, gates, biases)
    - Embedding weights

    Args:
        model: The model whose parameters to optimize.
        cfg: Training configuration.

    Returns:
        Configured AdamW optimizer.
    """
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or "embed" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(
        param_groups,
        lr=cfg.lr,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_eps,
    )


def get_schedule_fn(
    scheduler_name: str,
    warmup_steps: int,
    total_steps: int,
    min_ratio: float = 0.0,
    stable_fraction: float = 0.0,
) -> Callable[[int], float]:
    """Return a step -> lr_multiplier function for LambdaLR.

    All schedule variants follow a three-phase structure:
    warmup (linear 0->1) -> stable (1.0) -> decay (1.0 -> min_ratio).

    For non-WSD schedules, the stable phase has zero length.

    Args:
        scheduler_name: One of ``warmup_linear``, ``warmup_cosine``,
            ``wsd_linear``, ``wsd_cosine``.
        warmup_steps: Number of linear warmup steps.
        total_steps: Total training steps.
        min_ratio: Minimum LR as a fraction of peak LR (min_lr / lr).
        stable_fraction: Fraction of total steps at peak LR after warmup
            (only meaningful for WSD schedules).

    Returns:
        A callable mapping step number to LR multiplier.
    """
    is_wsd = scheduler_name.startswith("wsd_")
    is_cosine = scheduler_name.endswith("_cosine")

    stable_steps = int(total_steps * stable_fraction) if is_wsd else 0
    decay_steps = max(1, total_steps - warmup_steps - stable_steps)

    def schedule_fn(current_step: int) -> float:
        # Warmup phase: linear ramp from 0 to 1
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)

        # Stable phase (WSD only): hold at peak
        if current_step < warmup_steps + stable_steps:
            return 1.0

        # Decay phase
        progress = (current_step - warmup_steps - stable_steps) / decay_steps
        progress = min(progress, 1.0)

        decay = 0.5 * (1.0 + math.cos(math.pi * progress)) if is_cosine else 1.0 - progress

        return min_ratio + (1.0 - min_ratio) * decay

    return schedule_fn


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    total_steps: int,
) -> LambdaLR:
    """Build a learning rate scheduler from config.

    Args:
        optimizer: The optimizer to schedule.
        cfg: Training configuration.
        total_steps: Total number of training steps.

    Returns:
        A LambdaLR scheduler.
    """
    min_ratio = cfg.min_lr / cfg.lr if cfg.lr > 0 else 0.0

    schedule_fn = get_schedule_fn(
        scheduler_name=cfg.scheduler,
        warmup_steps=cfg.warmup_steps,
        total_steps=total_steps,
        min_ratio=min_ratio,
        stable_fraction=cfg.stable_fraction,
    )

    return LambdaLR(optimizer, schedule_fn)
