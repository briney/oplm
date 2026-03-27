"""Tests for StructureEvalTask."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
import torch

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
from oplm.eval.tasks.structure import StructureEvalTask
from oplm.model.transformer import OplmForMLM


def _make_cfg(max_length: int = 128) -> OplmConfig:
    return OplmConfig(
        model=ModelConfig(
            hidden_dim=32,
            num_layers=2,
            num_heads=2,
            num_kv_heads=2,
            max_seq_len=max_length,
        ),
        train=TrainConfig(batch_size=1, eval_every=100),
        data=DataConfig(max_length=max_length, num_workers=0),
    )


def _make_entry(
    name: str = "test_structures",
    path: str = "/fake/path",
    eval_every: int | None = None,
    metrics: list[str] | None = None,
    extra: dict | None = None,
) -> EvalDatasetEntry:
    return EvalDatasetEntry(
        name=name,
        path=path,
        type="structure",
        eval_every=eval_every,
        metrics=metrics,
        extra=extra or {},
    )


class TestStructureEvalTaskRegistration:
    """Test that StructureEvalTask is properly registered."""

    def test_registered_as_structure(self) -> None:
        assert "structure" in EVAL_TASK_REGISTRY
        assert EVAL_TASK_REGISTRY["structure"] is StructureEvalTask

    def test_default_metrics(self) -> None:
        assert StructureEvalTask.default_metrics == ["precision_at_L"]


class TestStructureEvalTaskInit:
    """Test StructureEvalTask initialization."""

    def test_basic_init(self) -> None:
        cfg = _make_cfg()
        entry = _make_entry()
        task = StructureEvalTask(entry, cfg)

        assert task.name == "test_structures"
        assert task.eval_every == 100
        assert task.metrics == ["precision_at_L"]
        assert task.contact_threshold == 8.0
        assert task.min_seq_sep == 6
        assert task.l_divisor == 1
        assert task.use_cbeta is True
        assert task.use_logistic_regression is True
        assert task.logreg_n_train == 20
        assert task.logreg_n_iterations == 5
        assert task.logreg_c == 0.15
        assert task.max_structures is None

    def test_extra_config_parsed(self) -> None:
        cfg = _make_cfg()
        entry = _make_entry(
            extra={
                "contact_threshold": 10.0,
                "l_divisor": 2,
                "use_cbeta": False,
                "logreg_c": 0.5,
                "max_structures": 50,
            }
        )
        task = StructureEvalTask(entry, cfg)

        assert task.contact_threshold == 10.0
        assert task.l_divisor == 2
        assert task.use_cbeta is False
        assert task.logreg_c == 0.5
        assert task.max_structures == 50

    def test_lazy_loading(self) -> None:
        cfg = _make_cfg()
        entry = _make_entry()
        task = StructureEvalTask(entry, cfg)

        assert task._structures is None
        assert task._tokenizer is None


class TestStructureEvalTaskEvaluate:
    """Test StructureEvalTask.evaluate with real data."""

    @pytest.mark.slow
    def test_evaluate_returns_precision(self, structure_fixtures_dir: Path) -> None:
        """evaluate() should return precision_at_L in [0, 1]."""
        cfg = _make_cfg(max_length=128)
        entry = _make_entry(path=str(structure_fixtures_dir))
        task = StructureEvalTask(entry, cfg)

        model = OplmForMLM(cfg.model)
        model.eval()

        acc = MagicMock()
        acc.device = torch.device("cpu")
        acc.process_index = 0
        acc.num_processes = 1

        metrics = task.evaluate(model, acc)

        assert "precision_at_L" in metrics
        assert 0.0 <= metrics["precision_at_L"] <= 1.0

    @pytest.mark.slow
    def test_evaluate_filters_metrics(self, structure_fixtures_dir: Path) -> None:
        """Only requested metrics should be returned."""
        cfg = _make_cfg(max_length=128)
        entry = _make_entry(
            path=str(structure_fixtures_dir),
            metrics=["precision_at_L"],
        )
        task = StructureEvalTask(entry, cfg)

        model = OplmForMLM(cfg.model)
        model.eval()

        acc = MagicMock()
        acc.device = torch.device("cpu")
        acc.process_index = 0
        acc.num_processes = 1

        metrics = task.evaluate(model, acc)
        assert set(metrics.keys()) == {"precision_at_L"}

    @pytest.mark.slow
    def test_structures_cached(self, structure_fixtures_dir: Path) -> None:
        """Structures should be loaded once and reused."""
        cfg = _make_cfg(max_length=128)
        entry = _make_entry(path=str(structure_fixtures_dir))
        task = StructureEvalTask(entry, cfg)

        model = OplmForMLM(cfg.model)
        model.eval()

        acc = MagicMock()
        acc.device = torch.device("cpu")
        acc.process_index = 0
        acc.num_processes = 1

        task.evaluate(model, acc)
        structures_first = task._structures

        task.evaluate(model, acc)
        structures_second = task._structures

        assert structures_first is structures_second


class TestStructureEvalTaskIntegration:
    """Integration test with the Evaluator."""

    def test_evaluator_creates_structure_task(self, structure_fixtures_dir: Path) -> None:
        from oplm.eval.evaluator import Evaluator

        cfg = OplmConfig(
            model=ModelConfig(
                hidden_dim=32,
                num_layers=2,
                num_heads=2,
                num_kv_heads=2,
                max_seq_len=128,
            ),
            train=TrainConfig(batch_size=1, eval_every=10),
            data=DataConfig(
                max_length=128,
                num_workers=0,
                eval={
                    "pdb": {
                        "path": str(structure_fixtures_dir),
                        "type": "structure",
                        "contact_threshold": 8.0,
                    }
                },
            ),
        )

        evaluator = Evaluator(cfg)
        assert len(evaluator.tasks) == 1
        assert isinstance(evaluator.tasks[0], StructureEvalTask)
        assert evaluator.tasks[0].name == "pdb"
        assert evaluator.tasks[0].contact_threshold == 8.0

    @pytest.mark.slow
    def test_evaluator_runs_structure_task(self, structure_fixtures_dir: Path) -> None:
        """End-to-end: Evaluator runs StructureEvalTask and returns prefixed metrics."""
        from oplm.eval.evaluator import Evaluator

        cfg = OplmConfig(
            model=ModelConfig(
                hidden_dim=32,
                num_layers=2,
                num_heads=2,
                num_kv_heads=2,
                max_seq_len=128,
            ),
            train=TrainConfig(batch_size=1, eval_every=10),
            data=DataConfig(
                max_length=128,
                num_workers=0,
                eval={
                    "pdb": {
                        "path": str(structure_fixtures_dir),
                        "type": "structure",
                    }
                },
            ),
        )

        evaluator = Evaluator(cfg)
        model = OplmForMLM(cfg.model)

        acc = MagicMock()
        acc.device = torch.device("cpu")
        acc.process_index = 0
        acc.num_processes = 1

        metrics = evaluator(model, acc, global_step=10)

        assert "eval/pdb/precision_at_L" in metrics
        assert 0.0 <= metrics["eval/pdb/precision_at_L"] <= 1.0
