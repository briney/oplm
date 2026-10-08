"""TrainTask seam: MLMTask reproduces the pre-seam step; a custom task's metrics and
FLOP policy flow through the Trainer (docs/TRAIN.md §17)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import pytest
import torch

from oplm.training.flops import estimate_flops_per_token
from oplm.training.task import MLMTask, StepResult, TrainTask
from tests.training.conftest import FullRecordingCallback, tiny_train_cfg

if TYPE_CHECKING:
    from pathlib import Path

    from torch import nn


def test_step_result_rejects_reserved_metric_keys() -> None:
    with pytest.raises(ValueError, match="reserved"):
        StepResult(loss=torch.tensor(0.0), tokens=1, samples=1, metrics={"loss": 1.0})


def test_mlm_task_step_matches_direct_model_call(tmp_path: Path, training_parquet: Path) -> None:
    """MLMTask.step returns the model's own loss plus exact token/sample accounting."""
    cfg = tiny_train_cfg(tmp_path, training_parquet)
    task = MLMTask()
    assert isinstance(task, TrainTask)
    model = task.build_model(cfg, None)
    model.eval()  # no dropout: the two forward passes below must agree exactly
    batch = next(iter(task.build_dataloader(cfg)))

    result = task.step(model, batch)
    expected = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )["loss"]
    assert torch.equal(result.loss, expected)
    assert result.tokens == int(batch["attention_mask"].sum())
    assert result.samples == len(batch["input_ids"])
    assert result.metrics == {}
    assert task.flops_per_token(cfg) == estimate_flops_per_token(cfg.model)


class _AuxMetricTask(MLMTask):
    """MLMTask that also reports a per-micro-batch metric and declines to estimate FLOPs."""

    def step(self, model: nn.Module, batch: dict[str, Any]) -> StepResult:
        result = super().step(model, batch)
        result.metrics["aux"] = float(result.tokens)
        return result

    def flops_per_token(self, cfg: Any) -> int | None:
        return None


@pytest.mark.slow
def test_custom_task_metrics_and_flop_policy_flow_through_trainer(
    tmp_path: Path, training_parquet: Path
) -> None:
    from oplm.training.trainer import Trainer

    cfg = tiny_train_cfg(tmp_path, training_parquet, max_steps=4, log_every=2, batch_size=4)
    callback = FullRecordingCallback()
    Trainer(cfg, callbacks=[callback], task=_AuxMetricTask()).train()

    assert callback.train_log_steps == [2, 4]
    for _step, metrics in callback.train_logs:
        assert math.isfinite(metrics["train/aux"])
        assert "train/flops" not in metrics
        assert "train/achieved_tflops" not in metrics
        assert "train/mfu" not in metrics
    # train/aux is the window mean of a per-micro-batch token count: two optimizer
    # steps per log window at accumulation 1, so 2 * mean == the window's token delta.
    (_, first), (_, second) = callback.train_logs
    assert second["train/aux"] * 2 == pytest.approx(second["train/tokens"] - first["train/tokens"])
