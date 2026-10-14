"""``fold_monomer``: held-out monomer accuracy as an eval task (addendum §6, spec §8).

In training the Trainer passes the live model (never the EMA copy) every ``every`` steps;
``oplm fold eval`` scores saved EMA weights offline with the same scoring code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, ClassVar

from oplm.eval.registry import register_eval_task
from oplm.eval.tasks.base import EvalTask
from oplm.eval.tasks.variant_scoring import gather_objects
from oplm.fold.eval.monomer import METRICS, SamplingSettings, score_target, summarize
from oplm.fold.eval.targets import load_targets

if TYPE_CHECKING:
    from accelerate import Accelerator

    from oplm.config import EvalDatasetEntry, OplmConfig
    from oplm.fold.eval.targets import EvalTarget
    from oplm.fold.modeling_fold import OplmForFolding

logger = logging.getLogger(__name__)

__all__ = ["FoldMonomerTask", "FoldMonomerTaskConfig"]

_POSITIVE = ("max_targets", "max_length", "num_loops", "num_steps", "num_samples")


@dataclass(frozen=True)
class FoldMonomerTaskConfig:
    """The extra keys of a ``type: fold_monomer`` entry under ``data.eval``."""

    target_set: str = "valid"
    max_targets: int | None = None
    max_length: int | None = None  # None: the model's lm_max_chain_length
    num_loops: int | None = None  # None: the checkpoint's inference default
    num_steps: int | None = None
    num_samples: int = 1
    seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.target_set, str) or not self.target_set:
            raise ValueError(f"fold_monomer target_set must be a name, got {self.target_set!r}")
        for name in _POSITIVE:
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"fold_monomer {name} must be a positive integer, got {value!r}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError(f"fold_monomer seed must be an integer, got {self.seed!r}")

    @classmethod
    def from_extra(cls, extra: dict[str, Any]) -> FoldMonomerTaskConfig:
        """Build from an eval entry's ``extra`` mapping.

        Raises:
            ValueError: An unknown key, or a value of the wrong type or range.
        """
        known = sorted(f.name for f in fields(cls))
        unknown = sorted(set(extra) - set(known))
        if unknown:
            raise ValueError(f"unknown fold_monomer option(s) {unknown}; known: {known}")
        return cls(**extra)

    @property
    def sampling(self) -> SamplingSettings:
        """The fold settings for every target."""
        return SamplingSettings(
            num_loops=self.num_loops,
            num_steps=self.num_steps,
            num_samples=self.num_samples,
            seed=self.seed,
        )


@register_eval_task("fold_monomer")
class FoldMonomerTask(EvalTask["OplmForFolding"]):
    """Fold every target of a set, sharded across ranks, and report the means (spec §8).

    Metrics: TM-score, CA and all-atom lDDT, mean pLDDT (0-1) and pTM over targets scored ``ok``,
    plus the ``targets`` and ``failed`` counts. The entry's ``path`` is a preprocessed dataset.
    """

    default_metrics: ClassVar[list[str]] = [*METRICS, "targets", "failed"]

    def __init__(self, entry: EvalDatasetEntry, cfg: OplmConfig) -> None:
        super().__init__(entry, cfg)
        self.tcfg = FoldMonomerTaskConfig.from_extra(entry.extra)
        unknown = sorted(set(self.metrics) - set(self.default_metrics))
        if unknown:
            raise ValueError(
                f"fold_monomer cannot report {unknown}; available: {self.default_metrics}"
            )
        self._targets: list[EvalTarget] | None = None

    def evaluate(self, model: OplmForFolding, accelerator: Accelerator) -> dict[str, float]:
        """Score this rank's share of the targets, gather every row, return the means.

        Rank 0 logs a warning for each target that failed, with its error.

        Args:
            model: The live folding model (never the EMA copy), in eval mode.
            accelerator: The Accelerator; ranks take every ``num_processes``-th target.

        Returns:
            The metrics named by the entry (``tm_score``, ``targets``, ``failed``, ...).
        """
        if self._targets is None:
            max_length = self.tcfg.max_length or model.config.lm_max_chain_length
            self._targets, _ = load_targets(
                self.path, self.tcfg.target_set, max_length=max_length, limit=self.tcfg.max_targets
            )
        share = self._targets[accelerator.process_index :: accelerator.num_processes]
        local = [score_target(model, target, self.tcfg.sampling) for target in share]
        rows = gather_objects(local, accelerator)
        if accelerator.process_index == 0:
            for row in rows:
                if row.status != "ok":
                    logger.warning(
                        "fold_monomer %s: %s/%s_%s failed: %s",
                        self.name,
                        row.target_set,
                        row.entry_id,
                        row.chain_id,
                        row.error,
                    )
        summary = summarize(rows)
        return {k: v for k, v in summary.items() if k in self.metrics}
