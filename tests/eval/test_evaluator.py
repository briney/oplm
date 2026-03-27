"""Tests for the Evaluator orchestrator."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from oplm.config import EvalDatasetEntry, OplmConfig, TrainConfig
from oplm.eval.evaluator import Evaluator
from oplm.eval.registry import EVAL_TASK_REGISTRY
from oplm.eval.tasks.base import EvalTask


class _StubTask(EvalTask):
    """Concrete EvalTask for testing."""

    default_metrics = ["loss", "accuracy"]

    def __init__(self, entry: EvalDatasetEntry, cfg: OplmConfig) -> None:
        super().__init__(entry, cfg)
        self.call_count = 0

    def evaluate(self, model: object, accelerator: object) -> dict[str, float]:
        self.call_count += 1
        return {"loss": 0.5, "accuracy": 0.8}


@pytest.fixture(autouse=True)
def _register_stub():
    """Register and clean up the _stub task type."""
    EVAL_TASK_REGISTRY["_stub"] = _StubTask
    yield
    EVAL_TASK_REGISTRY.pop("_stub", None)


def _make_cfg(eval_config: dict | None = None, eval_every: int = 100) -> OplmConfig:
    cfg = OplmConfig(train=TrainConfig(eval_every=eval_every))
    cfg.data.eval = eval_config
    return cfg


class TestEvaluator:
    """Test Evaluator initialization and __call__."""

    def test_no_eval_config(self) -> None:
        evaluator = Evaluator(_make_cfg(eval_config=None))
        assert not evaluator.has_tasks

    def test_empty_eval_config(self) -> None:
        evaluator = Evaluator(_make_cfg(eval_config={}))
        assert not evaluator.has_tasks

    def test_single_task_created(self) -> None:
        cfg = _make_cfg({"test_ds": {"path": "/data", "type": "_stub"}})
        evaluator = Evaluator(cfg)
        assert evaluator.has_tasks
        assert len(evaluator.tasks) == 1
        assert evaluator.tasks[0].name == "test_ds"
        assert evaluator.tasks[0].eval_every == 100

    def test_multiple_tasks(self) -> None:
        cfg = _make_cfg(
            {
                "ds1": {"path": "/data1", "type": "_stub"},
                "ds2": {"path": "/data2", "type": "_stub", "eval_every": 500},
            }
        )
        evaluator = Evaluator(cfg)
        assert len(evaluator.tasks) == 2

    def test_call_runs_due_tasks(self) -> None:
        cfg = _make_cfg({"ds": {"path": "/data", "type": "_stub"}}, eval_every=10)
        evaluator = Evaluator(cfg)
        model = MagicMock()
        accelerator = MagicMock()

        # Step 10 is due (10 % 10 == 0)
        metrics = evaluator(model, accelerator, global_step=10)
        assert "eval/ds/loss" in metrics
        assert "eval/ds/accuracy" in metrics
        assert metrics["eval/ds/loss"] == 0.5
        model.eval.assert_called_once()
        model.train.assert_called_once()

    def test_call_skips_non_due_step(self) -> None:
        cfg = _make_cfg({"ds": {"path": "/data", "type": "_stub"}}, eval_every=10)
        evaluator = Evaluator(cfg)
        model = MagicMock()
        accelerator = MagicMock()

        metrics = evaluator(model, accelerator, global_step=7)
        assert metrics == {}
        model.eval.assert_not_called()

    def test_mixed_schedules(self) -> None:
        cfg = _make_cfg(
            {
                "fast": {"path": "/f", "type": "_stub", "eval_every": 10},
                "slow": {"path": "/s", "type": "_stub", "eval_every": 50},
            }
        )
        evaluator = Evaluator(cfg)
        model = MagicMock()
        accelerator = MagicMock()

        # Step 10: only fast is due
        metrics = evaluator(model, accelerator, global_step=10)
        assert "eval/fast/loss" in metrics
        assert "eval/slow/loss" not in metrics

        model.reset_mock()

        # Step 50: both are due
        metrics = evaluator(model, accelerator, global_step=50)
        assert "eval/fast/loss" in metrics
        assert "eval/slow/loss" in metrics

    def test_model_restored_on_error(self) -> None:
        """Ensure model.train() is called even if a task raises."""

        class _FailingTask(EvalTask):
            default_metrics = ["loss"]

            def evaluate(self, model, accelerator):  # type: ignore[override]
                raise RuntimeError("eval failed")

        EVAL_TASK_REGISTRY["_fail"] = _FailingTask
        try:
            cfg = _make_cfg({"bad": {"path": "/x", "type": "_fail"}}, eval_every=1)
            evaluator = Evaluator(cfg)
            model = MagicMock()

            with pytest.raises(RuntimeError, match="eval failed"):
                evaluator(model, MagicMock(), global_step=1)

            # model.train() should still be called via finally
            model.train.assert_called_once()
        finally:
            EVAL_TASK_REGISTRY.pop("_fail", None)

    def test_unknown_type_raises(self) -> None:
        cfg = _make_cfg({"bad": {"path": "/x", "type": "nonexistent_xyz"}})
        with pytest.raises(ValueError, match="Unknown eval type"):
            Evaluator(cfg)
