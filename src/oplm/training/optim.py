"""Optimizer and learning rate scheduler construction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch.optim.lr_scheduler import LambdaLR

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from torch import nn

    from oplm.config import TrainConfig


@dataclass(frozen=True)
class OptimizerParamGroups:
    """Partitioned model parameters for AdamW or Muon training."""

    muon_params: list[torch.nn.Parameter]
    adamw_decay_params: list[torch.nn.Parameter]
    adamw_no_decay_params: list[torch.nn.Parameter]


def _uses_no_weight_decay(name: str, param: torch.nn.Parameter) -> bool:
    """Return whether a parameter should follow the no-decay AdamW rules."""
    return param.ndim <= 1 or "embed" in name


def partition_optimizer_params(model: nn.Module, cfg: TrainConfig) -> OptimizerParamGroups:
    """Split model parameters into Muon and AdamW groups.

    The AdamW path preserves the existing OPLM grouping rules. The Muon path
    follows PyTorch's recommendation to use Muon only for eligible 2D hidden
    weights and keep embeddings, biases, norms, classifier weights, and other
    non-2D parameters on AdamW.

    Args:
        model: The model whose parameters should be partitioned.
        cfg: Training configuration.

    Returns:
        Partitioned optimizer parameter groups.

    Raises:
        ValueError: If ``cfg.optimizer="muon"`` but no eligible Muon parameters
            are found.
        RuntimeError: If the partitioning misses or duplicates parameters.
    """
    muon_params: list[torch.nn.Parameter] = []
    adamw_decay_params: list[torch.nn.Parameter] = []
    adamw_no_decay_params: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if _uses_no_weight_decay(name, param):
            adamw_no_decay_params.append(param)
            continue

        if cfg.optimizer == "muon" and param.ndim == 2 and not name.startswith("mlm_head."):
            muon_params.append(param)
            continue

        adamw_decay_params.append(param)

    grouped_ids = [
        id(param) for param in [*muon_params, *adamw_decay_params, *adamw_no_decay_params]
    ]
    model_ids = [id(param) for param in model.parameters() if param.requires_grad]
    if len(grouped_ids) != len(set(grouped_ids)):
        raise RuntimeError("Optimizer parameter partition duplicated one or more parameters")
    if set(grouped_ids) != set(model_ids):
        raise RuntimeError("Optimizer parameter partition did not cover all trainable parameters")
    if cfg.optimizer == "muon" and not muon_params:
        raise ValueError("Muon optimizer requires at least one eligible 2D hidden weight")

    return OptimizerParamGroups(
        muon_params=muon_params,
        adamw_decay_params=adamw_decay_params,
        adamw_no_decay_params=adamw_no_decay_params,
    )


def _build_adamw_optimizer(
    decay_params: Sequence[torch.nn.Parameter],
    no_decay_params: Sequence[torch.nn.Parameter],
    cfg: TrainConfig,
) -> torch.optim.AdamW:
    """Build an AdamW optimizer using OPLM's decay grouping rules."""
    param_groups = [
        {"params": list(decay_params), "weight_decay": cfg.weight_decay},
        {"params": list(no_decay_params), "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        param_groups,
        lr=cfg.lr,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_eps,
    )


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    """Build the primary optimizer for the configured training mode.

    For ``optimizer="adamw"``, this is the single AdamW optimizer. For
    ``optimizer="muon"``, this returns the primary Muon optimizer; use
    :func:`build_optimizers` when the caller needs the complete optimizer set.

    Args:
        model: The model whose parameters to optimize.
        cfg: Training configuration.

    Returns:
        The primary optimizer instance.
    """
    return build_optimizers(model, cfg)[0]


def build_optimizers(model: nn.Module, cfg: TrainConfig) -> list[torch.optim.Optimizer]:
    """Build all optimizers required by the configured training mode.

    Args:
        model: The model whose parameters to optimize.
        cfg: Training configuration.

    Returns:
        A list containing one AdamW optimizer for the default path, or a Muon
        optimizer plus an auxiliary AdamW optimizer for ``optimizer="muon"``.
    """
    param_groups = partition_optimizer_params(model, cfg)
    if cfg.optimizer == "adamw":
        return [
            _build_adamw_optimizer(
                param_groups.adamw_decay_params,
                param_groups.adamw_no_decay_params,
                cfg,
            )
        ]

    muon_optimizer = torch.optim.Muon(
        param_groups.muon_params,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        momentum=cfg.muon_momentum,
        nesterov=cfg.muon_nesterov,
        ns_steps=cfg.muon_ns_steps,
        adjust_lr_fn=cfg.muon_adjust_lr_fn,
    )
    auxiliary_adamw = _build_adamw_optimizer(
        param_groups.adamw_decay_params,
        param_groups.adamw_no_decay_params,
        cfg,
    )
    return [muon_optimizer, auxiliary_adamw]


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


def build_schedulers(
    optimizers: Sequence[torch.optim.Optimizer],
    cfg: TrainConfig,
    total_steps: int,
) -> list[LambdaLR]:
    """Build one scheduler per optimizer.

    Args:
        optimizers: Optimizers to schedule.
        cfg: Training configuration.
        total_steps: Total number of training steps.

    Returns:
        A list of LambdaLR schedulers in the same order as ``optimizers``.
    """
    return [build_scheduler(optimizer, cfg, total_steps) for optimizer in optimizers]
