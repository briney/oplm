"""Abstract base class for evaluation tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar

from oplm.eval.schedule import build_schedule

if TYPE_CHECKING:
    from accelerate import Accelerator
    from torch import nn

    from oplm.config import EvalDatasetEntry, OplmConfig
    from oplm.eval.schedule import Schedule

ModelT = TypeVar("ModelT", bound="nn.Module")


class EvalTask(ABC, Generic[ModelT]):
    """Abstract base class for evaluation tasks.

    Each subclass handles one eval dataset type: loading data, computing metrics,
    and returning results. Subclasses are registered via @register_eval_task.
    ``ModelT`` is the model class a task scores (spec §6.4: the annotation generalizes to the
    common module interface): the MLM tasks bind ``OplmForMaskedLM``, ``fold_monomer`` binds
    ``OplmForFolding``.
    """

    # Class-level defaults. Subclasses override these.
    default_metrics: ClassVar[list[str]] = []

    def __init__(self, entry: EvalDatasetEntry, cfg: OplmConfig) -> None:
        self.name = entry.name
        self.path = entry.path
        self.metrics = entry.metrics or self.default_metrics
        self.schedule: Schedule = build_schedule(entry.schedule)
        self.cfg = cfg

    @abstractmethod
    def evaluate(
        self,
        model: ModelT,
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
