"""Evaluator orchestrator — single integration point between Trainer and eval tasks."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from oplm.config import parse_eval_configs
from oplm.eval.registry import get_eval_task_class

if TYPE_CHECKING:
    from accelerate import Accelerator

    from oplm.config import OplmConfig
    from oplm.eval.tasks.base import EvalTask
    from oplm.model.transformer import OplmForMLM

logger = logging.getLogger(__name__)


class Evaluator:
    """Orchestrates evaluation across multiple datasets and schedules.

    Instantiated once at training start. Called every optimizer step by the
    Trainer. Internally determines which tasks are due at the current step
    and runs only those.
    """

    def __init__(self, cfg: OplmConfig) -> None:
        # Trigger task registration by importing the tasks subpackage
        import oplm.eval.tasks  # noqa: F401

        entries = parse_eval_configs(cfg.data.eval, cfg.train.eval_every)
        self.tasks: list[EvalTask] = []
        for entry in entries:
            cls = get_eval_task_class(entry.type)
            self.tasks.append(cls(entry, cfg))
            logger.info(
                "Registered eval task %r (type=%s, eval_every=%d)",
                entry.name,
                entry.type,
                entry.eval_every,
            )

    def __call__(
        self,
        model: OplmForMLM,
        accelerator: Accelerator,
        global_step: int,
    ) -> dict[str, float]:
        """Run all due evaluations for the current step.

        Args:
            model: Unwrapped model (Evaluator handles eval/train mode toggle).
            accelerator: Accelerator instance.
            global_step: Current optimizer step.

        Returns:
            Merged metrics dict with eval/{name}/{metric} keys.
            Empty dict if no evals are due.
        """
        due_tasks = [t for t in self.tasks if global_step % t.eval_every == 0]
        if not due_tasks:
            return {}

        model.eval()
        all_metrics: dict[str, float] = {}
        try:
            for task in due_tasks:
                raw = task.evaluate(model, accelerator)
                for key, value in raw.items():
                    all_metrics[f"eval/{task.name}/{key}"] = value
        finally:
            model.train()

        return all_metrics

    @property
    def has_tasks(self) -> bool:
        """Whether any eval tasks are configured."""
        return len(self.tasks) > 0
