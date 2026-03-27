"""Tests for the eval task registry."""

from __future__ import annotations

import pytest

from oplm.eval.registry import EVAL_TASK_REGISTRY, get_eval_task_class, register_eval_task
from oplm.eval.tasks.base import EvalTask


class TestRegistry:
    """Test register_eval_task decorator and get_eval_task_class lookup."""

    def test_register_and_lookup(self) -> None:
        @register_eval_task("_test_reg")
        class _TestTask(EvalTask):
            def evaluate(self, model, accelerator):  # type: ignore[override]
                return {}

        try:
            assert get_eval_task_class("_test_reg") is _TestTask
        finally:
            EVAL_TASK_REGISTRY.pop("_test_reg", None)

    def test_duplicate_registration_raises(self) -> None:
        @register_eval_task("_test_dup")
        class _Task1(EvalTask):
            def evaluate(self, model, accelerator):  # type: ignore[override]
                return {}

        try:
            with pytest.raises(ValueError, match="already registered"):

                @register_eval_task("_test_dup")
                class _Task2(EvalTask):
                    def evaluate(self, model, accelerator):  # type: ignore[override]
                        return {}
        finally:
            EVAL_TASK_REGISTRY.pop("_test_dup", None)

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown eval type"):
            get_eval_task_class("nonexistent_type_xyz")
