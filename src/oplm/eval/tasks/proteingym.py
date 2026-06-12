"""ProteinGym DMS zero-shot variant-effect prediction.

Zero-shot fitness prediction on ProteinGym's deep-mutational-scanning (DMS)
substitution benchmark. Each CSV is one assay; the model scores every variant by
a per-position log-likelihood ratio (see :mod:`oplm.eval.tasks.variant_scoring`)
and the leaderboard metrics — Spearman, AUROC, and top-fraction precision — are
computed per assay and averaged across assays.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from oplm.data import get_tokenizer, load_variant_assays
from oplm.eval.metrics.classification import roc_auc_score
from oplm.eval.metrics.ranking import spearman_corr, top_k_precision
from oplm.eval.registry import register_eval_task
from oplm.eval.tasks.base import EvalTask
from oplm.eval.tasks.variant_scoring import (
    VALID_SCORING,
    compute_variant_llrs,
    gather_objects,
    wildtype_fits,
)

if TYPE_CHECKING:
    import torch
    from accelerate import Accelerator

    from oplm.config import EvalDatasetEntry, OplmConfig
    from oplm.data import VariantAssay
    from oplm.model import OplmForMaskedLM, OplmTokenizerFast

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProteinGymTaskConfig:
    """Typed ProteinGym DMS eval knobs, parsed from ``EvalDatasetEntry.extra``.

    Attributes:
        scoring: Variant-scoring method, ``"masked_marginals"`` (mask each
            mutated position; more accurate) or ``"wt_marginals"`` (one wild-type
            forward pass; cheaper).
        max_assays: Optional cap on the number of assays (CSV files) loaded.
        mask_batch_size: Masked sequences per forward pass (``masked_marginals``
            only).
        top_k_fraction: Top fraction (by experimental score) for the
            ``top_k_precision`` metric. Default ``0.1`` (top 10%).
    """

    scoring: str = "masked_marginals"
    max_assays: int | None = None
    mask_batch_size: int = 64
    top_k_fraction: float = 0.1

    @classmethod
    def from_extra(cls, extra: dict[str, Any]) -> ProteinGymTaskConfig:
        """Build a config from an eval entry's ``extra`` block, validating values."""
        cfg = cls(
            scoring=str(extra.get("scoring", "masked_marginals")),
            max_assays=int(extra["max_assays"]) if "max_assays" in extra else None,
            mask_batch_size=int(extra.get("mask_batch_size", 64)),
            top_k_fraction=float(extra.get("top_k_fraction", 0.1)),
        )
        if cfg.scoring not in VALID_SCORING:
            raise ValueError(f"scoring must be one of {VALID_SCORING}, got {cfg.scoring!r}")
        if cfg.max_assays is not None and cfg.max_assays < 0:
            raise ValueError("max_assays must be >= 0 when provided")
        if not 1 <= cfg.mask_batch_size <= 1024:
            raise ValueError("mask_batch_size must be in [1, 1024]")
        if not 0.0 < cfg.top_k_fraction <= 1.0:
            raise ValueError("top_k_fraction must be in (0, 1]")
        return cfg


@register_eval_task("proteingym")
class ProteinGymEvalTask(EvalTask):
    """Zero-shot DMS variant-effect prediction on the ProteinGym substitution benchmark.

    Data format: a directory of CSV files from the ProteinGym DMS substitution
    benchmark. Each CSV has a ``mutant`` column (e.g. ``"A42T"``), a ``DMS_score``
    column (continuous fitness), and a ``mutated_sequence`` column from which the
    wild-type is reconstructed. An optional ``DMS_score_bin`` column (binarized
    fitness, 0/1) enables AUROC.

    Evaluation protocol, per assay:
        1. Score each variant by the summed per-position log-likelihood ratio
           ``LLR = log P(mut_aa | context) - log P(wt_aa | context)``, using
           ``masked_marginals`` or ``wt_marginals``. A higher LLR predicts higher
           fitness, so scores align with ``DMS_score`` directly (no sign flip).
        2. Compute ``spearman`` (vs ``DMS_score``), ``auroc`` (vs ``DMS_score_bin``,
           when present), and ``top_k_precision`` (overlap of the top
           ``top_k_fraction`` of variants by ``DMS_score`` and by model score).
    Report each metric averaged across the assays where it is defined (an assay is
    skipped for a metric whose inputs are degenerate, e.g. a constant score column
    or a missing/single-class ``DMS_score_bin``).

    Task-specific config (extra keys on the eval dataset entry):
        scoring (str): ``masked_marginals`` (default) or ``wt_marginals``.
        max_assays (int | None): Cap on assays loaded. Default None (all).
        mask_batch_size (int): Masked sequences per forward pass. Default 64.
        top_k_fraction (float): Top fraction for ``top_k_precision``. Default 0.1.
    """

    default_metrics: ClassVar[list[str]] = ["spearman", "auroc", "top_k_precision"]

    def __init__(self, entry: EvalDatasetEntry, cfg: OplmConfig) -> None:
        super().__init__(entry, cfg)
        self.tcfg = ProteinGymTaskConfig.from_extra(entry.extra)

        # Cached data (lazily loaded on first evaluate()).
        self._assays: list[VariantAssay] | None = None
        self._tokenizer: OplmTokenizerFast | None = None

    def evaluate(
        self,
        model: OplmForMaskedLM,
        accelerator: Accelerator,
    ) -> dict[str, float]:
        """Score DMS variants and return each requested metric averaged across assays.

        Args:
            model: The unwrapped model (already in eval mode).
            accelerator: The Accelerator instance (for distributed gather).

        Returns:
            Dict of metric name to scalar value, restricted to requested metrics.
        """
        if self._assays is None:
            self._assays = load_variant_assays(self.path, max_assays=self.tcfg.max_assays)
        if self._tokenizer is None:
            self._tokenizer = get_tokenizer()

        if not self._assays:
            logger.warning("No DMS variant assays loaded from %s", self.path)
            return {m: 0.0 for m in self.metrics}

        # Shard whole assays across ranks (every metric is computed per assay).
        rank = accelerator.process_index
        world_size = accelerator.num_processes
        rank_assays = self._assays[rank::world_size]

        device = accelerator.device
        local_results = [
            result
            for assay in rank_assays
            if (result := self._score_assay(assay, model, device)) is not None
        ]

        all_results = gather_objects(local_results, accelerator)
        if not all_results:
            logger.warning("No scorable DMS assays in %s", self.path)
            return {m: 0.0 for m in self.metrics}

        return {m: self._mean_metric(all_results, m) for m in self.metrics}

    def _mean_metric(self, results: list[dict[str, float]], metric: str) -> float:
        """Average ``metric`` over the assays that produced it (0.0 if none did)."""
        values = [r[metric] for r in results if metric in r]
        if not values:
            logger.warning("DMS metric %r had no scorable assays", metric)
            return 0.0
        return sum(values) / len(values)

    def _score_assay(
        self,
        assay: VariantAssay,
        model: OplmForMaskedLM,
        device: torch.device,
    ) -> dict[str, float] | None:
        """Compute requested metrics for one assay.

        Returns a dict with the metrics that are defined for this assay (a metric
        is omitted when its inputs are degenerate), or None when the assay cannot
        be scored at all (wild-type too long, or an unknown amino acid).
        """
        assert self._tokenizer is not None

        if not wildtype_fits(assay.wildtype, self.cfg.model.max_position_embeddings):
            logger.debug(
                "Skipping %s: wild-type length %d exceeds max context %d",
                assay.name,
                len(assay.wildtype),
                self.cfg.model.max_position_embeddings - 2,
            )
            return None

        try:
            llr = compute_variant_llrs(
                assay,
                model,
                self._tokenizer,
                device,
                scoring=self.tcfg.scoring,
                mask_batch_size=self.tcfg.mask_batch_size,
            )
        except ValueError as err:
            logger.warning("[%s] skipping: %s", assay.name, err)
            return None

        dms_score = np.asarray(assay.labels, dtype=np.float64)
        result: dict[str, float] = {}

        # Higher LLR predicts higher fitness, so model score aligns with DMS_score
        # and with DMS_score_bin (1 = functional) directly — no sign flip.
        if "spearman" in self.metrics:
            self._maybe_set(result, "spearman", lambda: spearman_corr(llr, dms_score), assay.name)
        if "top_k_precision" in self.metrics:
            self._maybe_set(
                result,
                "top_k_precision",
                lambda: top_k_precision(dms_score, llr, fraction=self.tcfg.top_k_fraction),
                assay.name,
            )
        if "auroc" in self.metrics and assay.bin_labels is not None:
            bin_labels = np.asarray(assay.bin_labels, dtype=np.float64)
            self._maybe_set(result, "auroc", lambda: roc_auc_score(bin_labels, llr), assay.name)

        return result

    @staticmethod
    def _maybe_set(
        result: dict[str, float],
        metric: str,
        compute: Any,
        assay_name: str,
    ) -> None:
        """Store ``compute()`` under ``metric``, skipping it on a degenerate-input error."""
        try:
            result[metric] = compute()
        except ValueError as err:
            logger.debug("[%s] %s undefined: %s", assay_name, metric, err)
