"""Tests for SequenceEvalTask."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

if TYPE_CHECKING:
    from pathlib import Path

from oplm.config import (
    DataConfig,
    EvalDatasetEntry,
    ModelConfig,
    OplmConfig,
    TrainConfig,
)
from oplm.eval.registry import EVAL_TASK_REGISTRY
from oplm.eval.tasks.sequence import SequenceEvalTask
from oplm.model.transformer import OplmForMLM


def _make_cfg(max_length: int = 64, batch_size: int = 4) -> OplmConfig:
    return OplmConfig(
        model=ModelConfig(
            hidden_dim=32,
            num_layers=1,
            num_heads=2,
            num_kv_heads=2,
            max_seq_len=max_length,
        ),
        train=TrainConfig(batch_size=batch_size, eval_every=100),
        data=DataConfig(max_length=max_length, num_workers=0),
    )


def _make_entry(
    name: str = "test_eval",
    path: str = "/fake/path",
    eval_every: int | None = None,
    metrics: list[str] | None = None,
) -> EvalDatasetEntry:
    return EvalDatasetEntry(
        name=name,
        path=path,
        type="sequence",
        eval_every=eval_every,
        metrics=metrics,
    )


class TestSequenceEvalTaskRegistration:
    """Test that SequenceEvalTask is properly registered."""

    def test_registered_as_sequence(self) -> None:
        """The 'sequence' type should be registered in the eval task registry."""
        assert "sequence" in EVAL_TASK_REGISTRY
        assert EVAL_TASK_REGISTRY["sequence"] is SequenceEvalTask

    def test_default_metrics(self) -> None:
        """Default metrics should be loss, accuracy, perplexity."""
        assert SequenceEvalTask.default_metrics == ["loss", "accuracy", "perplexity"]


class TestSequenceEvalTaskInit:
    """Test SequenceEvalTask initialization."""

    def test_basic_init(self) -> None:
        cfg = _make_cfg()
        entry = _make_entry()
        task = SequenceEvalTask(entry, cfg)

        assert task.name == "test_eval"
        assert task.path == "/fake/path"
        assert task.eval_every == 100  # from train config
        assert task.metrics == ["loss", "accuracy", "perplexity"]

    def test_custom_eval_every(self) -> None:
        cfg = _make_cfg()
        entry = _make_entry(eval_every=500)
        task = SequenceEvalTask(entry, cfg)

        assert task.eval_every == 500

    def test_custom_metrics(self) -> None:
        cfg = _make_cfg()
        entry = _make_entry(metrics=["loss"])
        task = SequenceEvalTask(entry, cfg)

        assert task.metrics == ["loss"]

    def test_dataloader_not_initialized_at_construction(self) -> None:
        """DataLoader should be lazily initialized, not at construction time."""
        cfg = _make_cfg()
        entry = _make_entry()
        task = SequenceEvalTask(entry, cfg)

        assert task._dataloader is None


class TestSequenceEvalTaskEvaluate:
    """Test SequenceEvalTask.evaluate with real data."""

    def test_evaluate_returns_metrics(self, training_parquet: Path) -> None:
        """evaluate() should return all default metrics with valid values."""
        cfg = _make_cfg(max_length=64, batch_size=4)
        entry = _make_entry(path=str(training_parquet))
        task = SequenceEvalTask(entry, cfg)

        model = OplmForMLM(cfg.model)
        model.eval()

        acc = MagicMock()
        acc.device = __import__("torch").device("cpu")
        acc.reduce = lambda tensor, reduction: tensor

        metrics = task.evaluate(model, acc)

        assert "loss" in metrics
        assert "accuracy" in metrics
        assert "perplexity" in metrics
        assert metrics["loss"] > 0.0
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert metrics["perplexity"] > 1.0

    def test_evaluate_filters_to_requested_metrics(self, training_parquet: Path) -> None:
        """When custom metrics are specified, only those should be returned."""
        cfg = _make_cfg(max_length=64, batch_size=4)
        entry = _make_entry(path=str(training_parquet), metrics=["loss"])
        task = SequenceEvalTask(entry, cfg)

        model = OplmForMLM(cfg.model)
        model.eval()

        acc = MagicMock()
        acc.device = __import__("torch").device("cpu")
        acc.reduce = lambda tensor, reduction: tensor

        metrics = task.evaluate(model, acc)

        assert set(metrics.keys()) == {"loss"}
        assert "accuracy" not in metrics
        assert "perplexity" not in metrics

    def test_dataloader_lazily_initialized(self, training_parquet: Path) -> None:
        """DataLoader should be created on first evaluate() call."""
        cfg = _make_cfg(max_length=64, batch_size=4)
        entry = _make_entry(path=str(training_parquet))
        task = SequenceEvalTask(entry, cfg)

        assert task._dataloader is None

        model = OplmForMLM(cfg.model)
        model.eval()

        acc = MagicMock()
        acc.device = __import__("torch").device("cpu")
        acc.reduce = lambda tensor, reduction: tensor

        task.evaluate(model, acc)

        assert task._dataloader is not None

    def test_dataloader_reused_across_calls(self, training_parquet: Path) -> None:
        """The same DataLoader should be reused on subsequent evaluate() calls."""
        cfg = _make_cfg(max_length=64, batch_size=4)
        entry = _make_entry(path=str(training_parquet))
        task = SequenceEvalTask(entry, cfg)

        model = OplmForMLM(cfg.model)
        model.eval()

        acc = MagicMock()
        acc.device = __import__("torch").device("cpu")
        acc.reduce = lambda tensor, reduction: tensor

        task.evaluate(model, acc)
        dl_first = task._dataloader

        task.evaluate(model, acc)
        dl_second = task._dataloader

        assert dl_first is dl_second


class TestSequenceEvalTaskIntegration:
    """Integration test with the Evaluator."""

    def test_evaluator_creates_sequence_task(self, training_parquet: Path) -> None:
        """Evaluator should create a SequenceEvalTask from config."""
        from oplm.eval.evaluator import Evaluator

        cfg = OplmConfig(
            model=ModelConfig(
                hidden_dim=32,
                num_layers=1,
                num_heads=2,
                num_kv_heads=2,
                max_seq_len=64,
            ),
            train=TrainConfig(batch_size=4, eval_every=10),
            data=DataConfig(
                max_length=64,
                num_workers=0,
                eval={
                    "heldout": {
                        "path": str(training_parquet),
                        "type": "sequence",
                    }
                },
            ),
        )

        evaluator = Evaluator(cfg)
        assert len(evaluator.tasks) == 1
        assert isinstance(evaluator.tasks[0], SequenceEvalTask)
        assert evaluator.tasks[0].name == "heldout"

    def test_evaluator_runs_sequence_task(self, training_parquet: Path) -> None:
        """Evaluator should run SequenceEvalTask and return prefixed metrics."""
        from oplm.eval.evaluator import Evaluator

        cfg = OplmConfig(
            model=ModelConfig(
                hidden_dim=32,
                num_layers=1,
                num_heads=2,
                num_kv_heads=2,
                max_seq_len=64,
            ),
            train=TrainConfig(batch_size=4, eval_every=10),
            data=DataConfig(
                max_length=64,
                num_workers=0,
                eval={
                    "heldout": {
                        "path": str(training_parquet),
                        "type": "sequence",
                    }
                },
            ),
        )

        evaluator = Evaluator(cfg)
        model = OplmForMLM(cfg.model)

        acc = MagicMock()
        acc.device = __import__("torch").device("cpu")
        acc.reduce = lambda tensor, reduction: tensor

        metrics = evaluator(model, acc, global_step=10)

        assert "eval/heldout/loss" in metrics
        assert "eval/heldout/accuracy" in metrics
        assert "eval/heldout/perplexity" in metrics
        assert metrics["eval/heldout/loss"] > 0.0
