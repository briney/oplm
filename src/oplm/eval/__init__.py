"""Evaluation harness for OPLM."""

from __future__ import annotations

from oplm.eval.evaluator import Evaluator
from oplm.eval.registry import register_eval_task
from oplm.eval.tasks.base import EvalTask

__all__ = ["EvalTask", "Evaluator", "register_eval_task"]
