"""ProteinGlue benchmark evaluation (stub)."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from oplm.eval.registry import register_eval_task
from oplm.eval.tasks.base import EvalTask

if TYPE_CHECKING:
    from accelerate import Accelerator

    from oplm.model.transformer import OplmForMLM


@register_eval_task("proteinglue")
class ProteinGlueEvalTask(EvalTask):
    """ProteinGlue benchmark evaluation.

    Multi-task protein understanding benchmark covering fold classification,
    enzyme reaction classification, gene ontology prediction, and more.

    Data format: ProteinGlue benchmark files.
    See https://github.com/ibivu/protein-glue for format details.
    """

    default_metrics: ClassVar[list[str]] = [
        "fold_accuracy",
        "enzyme_accuracy",
        "go_fmax",
    ]

    def evaluate(
        self,
        model: OplmForMLM,
        accelerator: Accelerator,
    ) -> dict[str, float]:
        raise NotImplementedError(
            "ProteinGlue evaluation is not yet implemented. "
            "See this class docstring for the planned evaluation protocol."
        )
