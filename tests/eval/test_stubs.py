"""Tests for labeled benchmark stub eval tasks (ProteinGym, TAPE, ProteinGlue, EVEREST)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

from oplm.config import EvalDatasetEntry, OplmConfig, TrainConfig
from oplm.eval.evaluator import Evaluator
from oplm.eval.registry import EVAL_TASK_REGISTRY
from oplm.eval.tasks.everest import EverestEvalTask

if TYPE_CHECKING:
    from oplm.eval.tasks.base import EvalTask
from oplm.eval.tasks.proteinglue import ProteinGlueEvalTask
from oplm.eval.tasks.proteingym import ProteinGymEvalTask
from oplm.eval.tasks.tape import TapeEvalTask


def _make_entry(
    name: str = "test_ds",
    path: str = "/fake/path",
    task_type: str = "proteingym",
    eval_every: int | None = None,
    metrics: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> EvalDatasetEntry:
    return EvalDatasetEntry(
        name=name,
        path=path,
        type=task_type,
        eval_every=eval_every,
        metrics=metrics,
        extra=extra or {},
    )


def _make_cfg(eval_every: int = 1000) -> OplmConfig:
    return OplmConfig(train=TrainConfig(eval_every=eval_every))


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestStubRegistration:
    """All four stub task types should be registered."""

    def test_proteingym_registered(self) -> None:
        assert "proteingym" in EVAL_TASK_REGISTRY
        assert EVAL_TASK_REGISTRY["proteingym"] is ProteinGymEvalTask

    def test_tape_registered(self) -> None:
        assert "tape" in EVAL_TASK_REGISTRY
        assert EVAL_TASK_REGISTRY["tape"] is TapeEvalTask

    def test_proteinglue_registered(self) -> None:
        assert "proteinglue" in EVAL_TASK_REGISTRY
        assert EVAL_TASK_REGISTRY["proteinglue"] is ProteinGlueEvalTask

    def test_everest_registered(self) -> None:
        assert "everest" in EVAL_TASK_REGISTRY
        assert EVAL_TASK_REGISTRY["everest"] is EverestEvalTask


# ---------------------------------------------------------------------------
# Default metrics
# ---------------------------------------------------------------------------


class TestStubDefaultMetrics:
    """Each stub should declare its expected default metrics."""

    def test_proteingym_defaults(self) -> None:
        assert ProteinGymEvalTask.default_metrics == ["spearman", "ndcg"]

    def test_tape_defaults(self) -> None:
        assert TapeEvalTask.default_metrics == [
            "ss3_accuracy",
            "ss8_accuracy",
            "contact_precision",
            "homology_accuracy",
            "fluorescence_spearman",
            "stability_spearman",
        ]

    def test_proteinglue_defaults(self) -> None:
        assert ProteinGlueEvalTask.default_metrics == [
            "fold_accuracy",
            "enzyme_accuracy",
            "go_fmax",
        ]

    def test_everest_defaults(self) -> None:
        assert EverestEvalTask.default_metrics == ["spearman", "auroc"]


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestStubInit:
    """Stub tasks should initialize correctly from config."""

    @pytest.mark.parametrize(
        "task_type,cls",
        [
            ("proteingym", ProteinGymEvalTask),
            ("tape", TapeEvalTask),
            ("proteinglue", ProteinGlueEvalTask),
            ("everest", EverestEvalTask),
        ],
    )
    def test_basic_init(self, task_type: str, cls: type[EvalTask]) -> None:
        entry = _make_entry(task_type=task_type, name="ds1", path="/data/ds1")
        task = cls(entry, _make_cfg())
        assert task.name == "ds1"
        assert task.path == "/data/ds1"
        assert task.eval_every == 1000
        assert task.metrics == cls.default_metrics

    @pytest.mark.parametrize("task_type", ["proteingym", "tape", "proteinglue", "everest"])
    def test_custom_eval_every(self, task_type: str) -> None:
        cls = EVAL_TASK_REGISTRY[task_type]
        entry = _make_entry(task_type=task_type, eval_every=5000)
        task = cls(entry, _make_cfg())
        assert task.eval_every == 5000

    @pytest.mark.parametrize("task_type", ["proteingym", "tape", "proteinglue", "everest"])
    def test_custom_metrics(self, task_type: str) -> None:
        cls = EVAL_TASK_REGISTRY[task_type]
        entry = _make_entry(task_type=task_type, metrics=["spearman"])
        task = cls(entry, _make_cfg())
        assert task.metrics == ["spearman"]

    @pytest.mark.parametrize("task_type", ["proteingym", "tape", "proteinglue", "everest"])
    def test_falls_back_to_train_eval_every(self, task_type: str) -> None:
        cls = EVAL_TASK_REGISTRY[task_type]
        entry = _make_entry(task_type=task_type, eval_every=None)
        task = cls(entry, _make_cfg(eval_every=2000))
        assert task.eval_every == 2000


# ---------------------------------------------------------------------------
# evaluate() raises NotImplementedError
# ---------------------------------------------------------------------------


class TestStubEvaluateRaises:
    """All stubs should raise NotImplementedError with a descriptive message."""

    @pytest.mark.parametrize("task_type", ["proteingym", "tape", "proteinglue", "everest"])
    def test_evaluate_raises(self, task_type: str) -> None:
        cls = EVAL_TASK_REGISTRY[task_type]
        entry = _make_entry(task_type=task_type)
        task = cls(entry, _make_cfg())

        with pytest.raises(NotImplementedError, match="not yet implemented"):
            task.evaluate(MagicMock(), MagicMock())


# ---------------------------------------------------------------------------
# Evaluator integration
# ---------------------------------------------------------------------------


class TestStubEvaluatorIntegration:
    """Evaluator should correctly instantiate stub tasks from config."""

    @pytest.mark.parametrize("task_type", ["proteingym", "tape", "proteinglue", "everest"])
    def test_evaluator_creates_stub_task(self, task_type: str) -> None:
        cfg = _make_cfg()
        cfg.data.eval = {"bench": {"path": "/data/bench", "type": task_type}}
        evaluator = Evaluator(cfg)

        assert len(evaluator.tasks) == 1
        assert evaluator.tasks[0].name == "bench"
        assert type(evaluator.tasks[0]) is EVAL_TASK_REGISTRY[task_type]

    @pytest.mark.parametrize("task_type", ["proteingym", "tape", "proteinglue", "everest"])
    def test_evaluator_propagates_error(self, task_type: str) -> None:
        """Evaluator should propagate NotImplementedError from stubs."""
        cfg = _make_cfg(eval_every=1)
        cfg.data.eval = {"bench": {"path": "/data/bench", "type": task_type}}
        evaluator = Evaluator(cfg)

        model = MagicMock()
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            evaluator(model, MagicMock(), global_step=1)

        # model.train() should still be called (finally block)
        model.train.assert_called_once()

    def test_mixed_stub_and_real_config(self) -> None:
        """Evaluator should handle a mix of real and stub task types."""
        cfg = _make_cfg(eval_every=100)
        cfg.data.eval = {
            "heldout": {"path": "/data/heldout", "type": "sequence"},
            "pg": {"path": "/data/pg", "type": "proteingym", "eval_every": 5000},
            "tape": {"path": "/data/tape", "type": "tape", "eval_every": 10000},
        }
        evaluator = Evaluator(cfg)

        assert len(evaluator.tasks) == 3
        types = {t.name: type(t).__name__ for t in evaluator.tasks}
        assert types["heldout"] == "SequenceEvalTask"
        assert types["pg"] == "ProteinGymEvalTask"
        assert types["tape"] == "TapeEvalTask"
