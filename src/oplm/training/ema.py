"""Exponential moving average of trainable weights (``train.ema_decay``; off by default)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

if TYPE_CHECKING:
    from torch import nn

__all__ = ["EMA_HF_DIRNAME", "EMA_SIDECAR_NAME", "build_ema", "sync_ema_buffers"]

EMA_SIDECAR_NAME = "ema.pt"
EMA_HF_DIRNAME = "hf_ema"


def build_ema(model: nn.Module, decay: float) -> AveragedModel:
    """Deep-copy ``model`` into an EMA tracker driven by ``update_parameters``.

    The first ``update_parameters`` call copies the live weights; every later call
    applies ``ema = decay * ema + (1 - decay) * live``. ``n_averaged`` counts the
    updates and is part of the tracker's ``state_dict``. Buffers are not averaged;
    see :func:`sync_ema_buffers`.

    Args:
        model: The unwrapped (no DDP/compile wrapper) live model.
        decay: EMA decay in ``(0, 1)``.
    """
    return AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay), use_buffers=False)


@torch.no_grad()
def sync_ema_buffers(ema: AveragedModel, model: nn.Module) -> None:
    """Copy the live model's buffers into the EMA copy before exporting it.

    OPLM's buffers (RoPE caches, residual ``alpha``) are constants, so this is a
    no-op in practice; it keeps an exported ``hf_ema/`` self-consistent for any
    buffer that does change during training.
    """
    for (name, ema_buffer), (live_name, live_buffer) in zip(
        ema.module.named_buffers(), model.named_buffers(), strict=True
    ):
        assert name == live_name, f"buffer order mismatch: {name} vs {live_name}"
        ema_buffer.copy_(live_buffer)
