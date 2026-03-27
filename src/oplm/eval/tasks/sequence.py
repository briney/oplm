"""Sequence evaluation task — MLM loss, accuracy, and perplexity on held-out data."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from oplm.eval.data.sequence_loader import build_sequence_eval_dataloader
from oplm.eval.metrics.mlm import compute_mlm_metrics
from oplm.eval.registry import register_eval_task
from oplm.eval.tasks.base import EvalTask

if TYPE_CHECKING:
    from accelerate import Accelerator
    from torch.utils.data import DataLoader

    from oplm.config import EvalDatasetEntry, OplmConfig
    from oplm.model.transformer import OplmForMLM


@register_eval_task("sequence")
class SequenceEvalTask(EvalTask):
    """Evaluate MLM performance on held-out protein sequences.

    Data format: single parquet file or directory of sharded parquet files
    with columns ``(sequence_id, sequence)`` — same as training data.

    Default metrics:
        - ``loss``: Cross-entropy loss on masked positions.
        - ``accuracy``: Fraction of masked positions predicted correctly.
        - ``perplexity``: exp(loss), capped at 1000.
    """

    default_metrics: ClassVar[list[str]] = ["loss", "accuracy", "perplexity"]

    def __init__(self, entry: EvalDatasetEntry, cfg: OplmConfig) -> None:
        super().__init__(entry, cfg)
        self._dataloader: DataLoader | None = None

    def evaluate(
        self,
        model: OplmForMLM,
        accelerator: Accelerator,
    ) -> dict[str, float]:
        """Run MLM evaluation on held-out sequences.

        The DataLoader is lazily initialized on the first call to avoid
        loading eval data before training starts and to ensure the
        accelerator is fully set up.

        Args:
            model: The unwrapped model (already in eval mode).
            accelerator: The Accelerator instance.

        Returns:
            Dict of metric name to scalar value, filtered to requested metrics.
        """
        if self._dataloader is None:
            self._dataloader = build_sequence_eval_dataloader(self.path, self.cfg)

        all_metrics = compute_mlm_metrics(model, self._dataloader, accelerator)

        # Filter to only the metrics requested by the config
        return {k: v for k, v in all_metrics.items() if k in self.metrics}
