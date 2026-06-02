"""Evaluator orchestrator — the single integration point between Trainer and tasks."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from oplm.config import DEFAULT_EVAL_CADENCE, parse_schedule_block
from oplm.data.config import parse_eval_configs
from oplm.eval.registry import get_eval_task_class
from oplm.eval.schedule import EveryNSteps, EveryNTokens

if TYPE_CHECKING:
    import torch.nn as nn
    from accelerate import Accelerator

    from oplm.config import OplmConfig
    from oplm.eval.context import EvalContext
    from oplm.eval.tasks.base import EvalTask
    from oplm.model import OplmForMaskedLM

logger = logging.getLogger(__name__)


class Evaluator:
    """Builds eval tasks from config and runs the ones due at the current step."""

    def __init__(self, cfg: OplmConfig) -> None:
        import oplm.eval.tasks  # noqa: F401  -- triggers task registration

        # eval_every defaults to None (clean override merge); coalesce here.
        raw_cadence = cfg.train.eval_every
        if raw_cadence is None:
            raw_cadence = DEFAULT_EVAL_CADENCE
        default_schedule = parse_schedule_block(raw_cadence, "train.eval_every")
        entries = parse_eval_configs(cfg.data.eval, default_schedule)
        self.tasks: list[EvalTask] = []
        for entry in entries:
            cls = get_eval_task_class(entry.type)
            self.tasks.append(cls(entry, cfg))
            logger.info(
                "Registered eval task %r (type=%s, schedule=%r)",
                entry.name,
                entry.type,
                entry.schedule,
            )
        self._warn_unreachable(cfg)

    def _warn_unreachable(self, cfg: OplmConfig) -> None:
        """Warn about step schedules that can provably never fire (step-bounded runs)."""
        if cfg.train.max_epochs is not None:
            return  # total steps not known here for epoch-bounded runs
        total_steps = cfg.train.max_steps
        for task in self.tasks:
            sched = task.schedule
            # A schedule with at_start=True fires on the first eval call regardless of
            # cadence, and at_end=True fires on the final step — so it is only provably
            # unreachable when BOTH are off and the cadence exceeds the run length.
            if (
                isinstance(sched, EveryNSteps)
                and not sched.at_start
                and not sched.at_end
                and sched.n > total_steps
            ):
                logger.warning(
                    "Eval task %r: step cadence n=%d exceeds max_steps=%d with at_start "
                    "and at_end both false; it will never run.",
                    task.name,
                    sched.n,
                    total_steps,
                )

    def run_due(
        self, ctx: EvalContext, model: nn.Module, accelerator: Accelerator
    ) -> dict[str, float]:
        """Run every task due at ``ctx`` and return merged ``eval/<name>/<metric>`` metrics.

        Returns an empty dict (and does no unwrap / no eval-mode toggle) when nothing
        is due. ``model`` is the WRAPPED model; it is unwrapped here only when needed.
        """
        due = [t for t in self.tasks if t.schedule.is_due(ctx)]
        if not due:
            return {}
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.eval()
        metrics: dict[str, float] = {}
        try:
            for task in due:
                for key, value in task.evaluate(unwrapped, accelerator).items():
                    metrics[f"eval/{task.name}/{key}"] = value
        finally:
            unwrapped.train()
        return metrics

    @property
    def has_tasks(self) -> bool:
        """Whether any eval tasks are configured."""
        return len(self.tasks) > 0

    @property
    def needs_token_count(self) -> bool:
        """Whether any task uses a token schedule (so the trainer must reduce tokens)."""
        return any(isinstance(t.schedule, EveryNTokens) for t in self.tasks)
