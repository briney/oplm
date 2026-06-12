"""ProteinGym clinical-variant eval — Pathogenic-vs-Benign AUROC.

Zero-shot pathogenicity prediction on ProteinGym's clinical-substitution
benchmark. Each CSV is one protein; variants are labelled ``Pathogenic`` /
``Benign``. The model scores every variant by a per-position log-likelihood
ratio (see :mod:`oplm.eval.tasks.variant_scoring`); AUROC is computed per protein
and averaged across proteins.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from oplm.data import get_tokenizer, load_clinical_variant_assays
from oplm.eval.metrics.classification import roc_auc_score
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
class ProteinGymClinicalTaskConfig:
    """Typed clinical-variant eval knobs, parsed from ``EvalDatasetEntry.extra``.

    Attributes:
        scoring: Variant-scoring method, ``"masked_marginals"`` (mask each
            mutated position; more accurate) or ``"wt_marginals"`` (one wild-type
            forward pass; cheaper).
        max_assays: Optional cap on the number of assays (CSV files) loaded.
        mask_batch_size: Masked sequences per forward pass (``masked_marginals``
            only).
    """

    scoring: str = "masked_marginals"
    max_assays: int | None = None
    mask_batch_size: int = 64

    @classmethod
    def from_extra(cls, extra: dict[str, Any]) -> ProteinGymClinicalTaskConfig:
        """Build a config from an eval entry's ``extra`` block, validating values."""
        cfg = cls(
            scoring=str(extra.get("scoring", "masked_marginals")),
            max_assays=int(extra["max_assays"]) if "max_assays" in extra else None,
            mask_batch_size=int(extra.get("mask_batch_size", 64)),
        )
        if cfg.scoring not in VALID_SCORING:
            raise ValueError(f"scoring must be one of {VALID_SCORING}, got {cfg.scoring!r}")
        if cfg.max_assays is not None and cfg.max_assays < 0:
            raise ValueError("max_assays must be >= 0 when provided")
        if not 1 <= cfg.mask_batch_size <= 1024:
            raise ValueError("mask_batch_size must be in [1, 1024]")
        return cfg


@register_eval_task("proteingym_clinical")
class ProteinGymClinicalEvalTask(EvalTask):
    """Zero-shot clinical-variant pathogenicity prediction (Pathogenic vs Benign).

    Data format: a directory of CSV files from the ProteinGym clinical-substitution
    benchmark, one protein per file, with columns ``mutant`` (e.g. ``"R46W"``),
    ``protein_sequence`` (the constant wild-type), and ``DMS_bin_score`` (the
    categorical label ``Pathogenic`` / ``Benign``).

    Evaluation protocol, per assay:
        1. Score each variant by the summed per-position log-likelihood ratio
           ``LLR = log P(mut_aa | context) - log P(wt_aa | context)``, using
           ``masked_marginals`` or ``wt_marginals``.
        2. A higher LLR means the substitution is more tolerated (Benign-like).
           Pathogenic is the positive class, so the **pathogenicity score is
           ``-LLR``**: a good model ranks Pathogenic variants higher and yields
           AUROC > 0.5.
        3. Compute AUROC over (label, pathogenicity-score) pairs. Assays without
           both classes (or fewer than two rows) are skipped.
    Report the mean AUROC across all scorable assays.

    Task-specific config (extra keys on the eval dataset entry):
        scoring (str): ``masked_marginals`` (default) or ``wt_marginals``.
        max_assays (int | None): Cap on assays loaded. Default None (all).
        mask_batch_size (int): Masked sequences per forward pass. Default 64.
    """

    default_metrics: ClassVar[list[str]] = ["auroc"]

    def __init__(self, entry: EvalDatasetEntry, cfg: OplmConfig) -> None:
        super().__init__(entry, cfg)
        self.tcfg = ProteinGymClinicalTaskConfig.from_extra(entry.extra)

        # Cached data (lazily loaded on first evaluate()).
        self._assays: list[VariantAssay] | None = None
        self._tokenizer: OplmTokenizerFast | None = None

    def evaluate(
        self,
        model: OplmForMaskedLM,
        accelerator: Accelerator,
    ) -> dict[str, float]:
        """Score clinical variants and return mean per-assay AUROC.

        Args:
            model: The unwrapped model (already in eval mode).
            accelerator: The Accelerator instance (for distributed gather).

        Returns:
            Dict of metric name to scalar value, filtered to requested metrics.
        """
        if self._assays is None:
            self._assays = load_clinical_variant_assays(self.path, max_assays=self.tcfg.max_assays)
        if self._tokenizer is None:
            self._tokenizer = get_tokenizer()

        if not self._assays:
            logger.warning("No clinical variant assays loaded from %s", self.path)
            return {m: 0.0 for m in self.metrics}

        # Shard whole assays across ranks (AUROC is computed per assay).
        rank = accelerator.process_index
        world_size = accelerator.num_processes
        rank_assays = self._assays[rank::world_size]

        device = accelerator.device
        local_aurocs = [
            auroc
            for assay in rank_assays
            if (auroc := self._score_assay(assay, model, device)) is not None
        ]

        all_aurocs = gather_objects(local_aurocs, accelerator)
        if not all_aurocs:
            logger.warning("No scorable clinical assays (need both classes present)")
            return {m: 0.0 for m in self.metrics}

        results = {"auroc": sum(all_aurocs) / len(all_aurocs)}
        return {k: v for k, v in results.items() if k in self.metrics}

    def _score_assay(
        self,
        assay: VariantAssay,
        model: OplmForMaskedLM,
        device: torch.device,
    ) -> float | None:
        """Compute AUROC for one assay, or None if it cannot be scored.

        Returns None when the wild-type exceeds the model's context length, an
        amino acid is unknown to the tokenizer, or AUROC is undefined (a single
        class present).
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
            labels = np.asarray(assay.labels, dtype=np.float64)
            # Pathogenic (label 1) should score high → use -LLR (tolerated = low).
            return roc_auc_score(labels, -llr)
        except ValueError as err:
            logger.warning("[%s] skipping: %s", assay.name, err)
            return None
