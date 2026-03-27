"""TAPE benchmark evaluation (stub)."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from oplm.eval.registry import register_eval_task
from oplm.eval.tasks.base import EvalTask

if TYPE_CHECKING:
    from accelerate import Accelerator

    from oplm.model.transformer import OplmForMLM


@register_eval_task("tape")
class TapeEvalTask(EvalTask):
    """TAPE benchmark evaluation.

    Tasks: secondary structure prediction (Q3/Q8), contact prediction,
    remote homology detection, fluorescence prediction, stability prediction.

    Data format: TAPE benchmark LMDB files or converted parquet/CSV.
    See https://github.com/songlab-cal/tape for format details.

    Evaluation protocol:
        Each sub-task trains a lightweight head (linear or small MLP) on
        frozen embeddings, then evaluates on the test split.
    """

    default_metrics: ClassVar[list[str]] = [
        "ss3_accuracy",
        "ss8_accuracy",
        "contact_precision",
        "homology_accuracy",
        "fluorescence_spearman",
        "stability_spearman",
    ]

    def evaluate(
        self,
        model: OplmForMLM,
        accelerator: Accelerator,
    ) -> dict[str, float]:
        raise NotImplementedError(
            "TAPE evaluation is not yet implemented. "
            "See this class docstring for the planned evaluation protocol."
        )
