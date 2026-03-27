"""ProteinGym zero-shot variant effect prediction (stub)."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from oplm.eval.registry import register_eval_task
from oplm.eval.tasks.base import EvalTask

if TYPE_CHECKING:
    from accelerate import Accelerator

    from oplm.model.transformer import OplmForMLM


@register_eval_task("proteingym")
class ProteinGymEvalTask(EvalTask):
    """Zero-shot variant effect prediction using ProteinGym.

    Evaluates the model's ability to predict the functional effects of
    protein mutations using pseudo-log-likelihood scoring (masked marginals).

    Data format: directory of CSV files from the ProteinGym substitution
    benchmark. Each CSV has columns: mutant (e.g. "A42T"), DMS_score (float),
    and the wild-type sequence in the metadata.

    Evaluation protocol:
        For each assay:
        1. Encode wild-type sequence.
        2. For each mutant, compute the log-likelihood ratio:
           score = log P(mutant_aa | context) - log P(wt_aa | context)
           using masked marginal scoring (mask each mutant position, score).
        3. Compute Spearman correlation between model scores and DMS_score.
        4. Report mean Spearman across all assays.

    Task-specific config:
        max_assays: int | None     -- limit number of assays (for cost control)
        scoring: str               -- "masked_marginals" (default) or "wild_type_marginals"
    """

    default_metrics: ClassVar[list[str]] = ["spearman", "ndcg"]

    def evaluate(
        self,
        model: OplmForMLM,
        accelerator: Accelerator,
    ) -> dict[str, float]:
        raise NotImplementedError(
            "ProteinGym evaluation is not yet implemented. "
            "See this class docstring for the planned evaluation protocol."
        )
