"""OPLM evaluation harness. See docs/EVAL_HARNESS.md for the design."""

from __future__ import annotations

from oplm.eval.context import EvalContext
from oplm.eval.evaluator import Evaluator
from oplm.eval.registry import register_eval_task
from oplm.eval.schedule import EveryNSteps, EveryNTokens, Schedule
from oplm.eval.tasks.base import EvalTask

__all__ = [
    "EvalContext",
    "EvalTask",
    "Evaluator",
    "EveryNSteps",
    "EveryNTokens",
    "Schedule",
    "register_eval_task",
]
