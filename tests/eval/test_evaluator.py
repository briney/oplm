"""Evaluator orchestration tests — no real model; a dummy task stands in."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from oplm.config import DataConfig, OplmConfig, TrainConfig
from oplm.eval import Evaluator

if TYPE_CHECKING:
    from collections.abc import Callable

    from oplm.eval import EvalContext


def _cfg(eval_map: dict[str, Any], *, max_steps: int = 50_000) -> OplmConfig:
    """Build a root config carrying the given ``data.eval`` map."""
    return OplmConfig(
        train=TrainConfig(max_steps=max_steps, wandb_enabled=False),
        data=DataConfig(eval=eval_map),
    )


class _StubModel:
    """Records eval()/train() toggles so we can assert the unwrap path ran."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def eval(self) -> _StubModel:
        self.calls.append("eval")
        return self

    def train(self) -> _StubModel:
        self.calls.append("train")
        return self


class _FakeAccelerator:
    """Minimal accelerator: ``unwrap_model`` is the identity."""

    def unwrap_model(self, model: Any) -> Any:
        return model


def test_run_due_returns_empty_when_nothing_due(
    ctx: Callable[..., EvalContext], dummy_task_type: str
) -> None:
    """Not-due path returns ``{}`` without unwrapping or touching the model."""
    cfg = _cfg({"d": {"path": "x", "type": dummy_task_type, "every": {"steps": 10}}})
    evaluator = Evaluator(cfg)
    # model/accelerator are None: reaching them would raise, proving the early return.
    assert evaluator.run_due(ctx(step=9, steps_delta=1), None, None) == {}  # type: ignore[arg-type]


def test_run_due_namespaces_metrics(ctx: Callable[..., EvalContext], dummy_task_type: str) -> None:
    """Due path runs the task and namespaces its metrics as ``eval/<name>/<metric>``."""
    cfg = _cfg({"d": {"path": "x", "type": dummy_task_type, "every": {"steps": 10}}})
    evaluator = Evaluator(cfg)
    model = _StubModel()
    metrics = evaluator.run_due(ctx(step=10, steps_delta=1), model, _FakeAccelerator())  # type: ignore[arg-type]
    assert metrics == {"eval/d/score": 1.0}
    # The model is toggled to eval for the run and restored to train afterward.
    assert model.calls == ["eval", "train"]


def test_needs_token_count_true_for_token_cadence(dummy_task_type: str) -> None:
    """A token-cadence task flags that the trainer must rank-reduce tokens."""
    cfg = _cfg({"d": {"path": "x", "type": dummy_task_type, "every": {"tokens": 1_000_000}}})
    assert Evaluator(cfg).needs_token_count is True


def test_needs_token_count_false_for_step_cadence(dummy_task_type: str) -> None:
    """A purely step-cadence configuration does not require token counting."""
    cfg = _cfg({"d": {"path": "x", "type": dummy_task_type, "every": {"steps": 10}}})
    assert Evaluator(cfg).needs_token_count is False


def test_warns_on_unreachable_step_schedule(dummy_task_type: str, caplog: Any) -> None:
    """A step cadence beyond ``max_steps`` with at_end off logs an unreachable warning."""
    cfg = _cfg(
        {"d": {"path": "x", "type": dummy_task_type, "every": {"steps": 10**9, "at_end": False}}},
        max_steps=100,
    )
    with caplog.at_level(logging.WARNING, logger="oplm.eval.evaluator"):
        Evaluator(cfg)
    assert any("never run" in rec.message for rec in caplog.records)
