"""Training-task seam: what the Trainer needs from a model/data/step triple.

The Trainer owns acceleration, optimization, gradient accumulation, evaluation
cadence, fault tolerance, and checkpointing. A :class:`TrainTask` owns the three
things that differ between objectives: how the model is built, how the training
dataloader is built, and how one micro-batch becomes a loss. :class:`MLMTask` is
the default and reproduces the pre-seam masked-language-model behaviour exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path

    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    from oplm.config import OplmConfig

__all__ = ["MLMTask", "StepResult", "TrainTask"]

# Metric keys the Trainer itself emits under ``train/``; a task may not reuse them.
_RESERVED_METRIC_KEYS = frozenset(
    {
        "loss",
        "loss_mean",
        "lr",
        "lr_adamw",
        "epoch",
        "samples",
        "tokens",
        "flops",
        "global_step",
        "grad_norm",
        "grad_norm_max",
        "clip_frac",
        "mean_seq_len",
        "tokens_per_sec",
        "step_time_s",
        "achieved_tflops",
        "mfu",
    }
)


@dataclass
class StepResult:
    """What one micro-batch produced.

    Attributes:
        loss: Differentiable scalar the Trainer backpropagates. The Trainer averages
            the logged value across the accumulation window; return the plain loss.
        tokens: Valid (non-padding) tokens in this rank's micro-batch.
        samples: Examples in this rank's micro-batch.
        metrics: Detached scalars, logged as ``train/<key>`` (mean over the log window).
    """

    loss: torch.Tensor
    tokens: int
    samples: int
    metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        clash = _RESERVED_METRIC_KEYS & self.metrics.keys()
        if clash:
            raise ValueError(f"StepResult.metrics may not use reserved keys {sorted(clash)}")


@runtime_checkable
class TrainTask(Protocol):
    """The objective-specific third of a training run."""

    def build_model(self, cfg: OplmConfig, initialization_source: Path | None) -> nn.Module:
        """Construct the trainable model, loading weights from ``initialization_source`` if set."""
        ...

    def build_dataloader(self, cfg: OplmConfig) -> DataLoader[Any]:
        """Construct the rank/worker-striped training dataloader."""
        ...

    def step(self, model: nn.Module, batch: dict[str, Any]) -> StepResult:
        """Run one micro-batch forward and return its loss and accounting."""
        ...

    def flops_per_token(self, cfg: OplmConfig) -> int | None:
        """Training FLOPs per token, or ``None`` when no honest estimate exists."""
        ...


class MLMTask:
    """Default task: masked-language-model pretraining with :class:`OplmForMaskedLM`."""

    def build_model(self, cfg: OplmConfig, initialization_source: Path | None) -> nn.Module:
        from oplm.model import OplmForMaskedLM

        # transformers strips ``config.gradient_checkpointing`` during
        # ``PreTrainedModel.__init__``, so read it before constructing the model.
        gradient_checkpointing = getattr(cfg.model, "gradient_checkpointing", False)
        if initialization_source is None:
            model = OplmForMaskedLM(cfg.model)
        else:
            from oplm.training.initialization import load_initial_model

            model = load_initial_model(initialization_source, cfg.model)
            model.train()  # HF pretrained loading returns an evaluation-mode model.
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()  # propagates to every OplmBlock
        return model

    def build_dataloader(self, cfg: OplmConfig) -> DataLoader[Any]:
        from oplm.data import build_train_dataloader

        return build_train_dataloader(cfg)

    def step(self, model: nn.Module, batch: dict[str, Any]) -> StepResult:
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return StepResult(
            loss=outputs["loss"],
            tokens=int(batch["attention_mask"].sum().item()),
            samples=len(batch["input_ids"]),
        )

    def flops_per_token(self, cfg: OplmConfig) -> int | None:
        from oplm.training.flops import estimate_flops_per_token

        return estimate_flops_per_token(cfg.model)
