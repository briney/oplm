"""Abstract base class for evaluation tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from accelerate import Accelerator

    from oplm.config import EvalDatasetEntry, OplmConfig
    from oplm.model.transformer import OplmForMLM


class EvalTask(ABC):
    """Abstract base class for evaluation tasks.

    Each subclass handles one eval dataset type: loading data, computing metrics,
    and returning results. Subclasses are registered via @register_eval_task.
    """

    # Class-level defaults. Subclasses override these.
    default_metrics: ClassVar[list[str]] = []

    def __init__(self, entry: EvalDatasetEntry, cfg: OplmConfig) -> None:
        self.name = entry.name
        self.path = entry.path
        self.eval_every: int = entry.eval_every or cfg.train.eval_every
        self.metrics = entry.metrics or self.default_metrics
        self.cfg = cfg

    @abstractmethod
    def evaluate(
        self,
        model: OplmForMLM,
        accelerator: Accelerator,
    ) -> dict[str, float]:
        """Run evaluation and return metrics.

        Keys should be bare metric names (e.g. "loss", "perplexity").
        The Evaluator prefixes them with eval/{self.name}/.

        Args:
            model: The unwrapped model in eval mode.
            accelerator: The Accelerator instance (for distributed ops).

        Returns:
            Mapping of metric name to scalar value.
        """
        ...
