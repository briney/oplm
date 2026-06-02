"""EVEREST benchmark for priority virus variant effect prediction (stub)."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from oplm.eval.registry import register_eval_task
from oplm.eval.tasks.base import EvalTask

if TYPE_CHECKING:
    from oplm.eval.context import DistContext
    from oplm.model import OplmForMaskedLM


@register_eval_task("everest")
class EverestEvalTask(EvalTask):
    """EVEREST benchmark for priority virus variant effect prediction.

    Evaluates zero-shot variant effect prediction on clinically relevant
    viral protein datasets.

    Data format: EVEREST benchmark CSV files.
    See https://github.com/debbiemarkslab/priority-viruses for details.
    """

    default_metrics: ClassVar[list[str]] = ["spearman", "auroc"]

    def evaluate(
        self,
        model: OplmForMaskedLM,
        dist_ctx: DistContext,
    ) -> dict[str, float]:
        raise NotImplementedError(
            "EVEREST evaluation is not yet implemented. "
            "See this class docstring for the planned evaluation protocol."
        )
