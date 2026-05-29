"""End-to-end sequence-eval task test on real held-out sequences (slow)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from oplm.config import (
    DataConfig,
    EvalDatasetEntry,
    ModelConfig,
    OplmConfig,
    ScheduleSpec,
    TrainConfig,
)
from oplm.eval.tasks.sequence import SequenceEvalTask

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from oplm.model import OplmForMaskedLM

pytestmark = pytest.mark.slow

_MAX_SEQ_LEN = 64
_BATCH_SIZE = 8


def _root_cfg() -> OplmConfig:
    """Root config feeding the eval dataloader (data/train) and length bound.

    Only ``data`` / ``train`` / ``model.max_seq_len`` are consulted here — the
    model itself is built separately from the HF config (see ``make_model``).
    """
    return OplmConfig(
        model=ModelConfig(max_seq_len=_MAX_SEQ_LEN),
        train=TrainConfig(batch_size=_BATCH_SIZE, wandb_enabled=False),
        data=DataConfig(num_workers=0, pin_memory=False),
    )


def test_sequence_eval_returns_finite_metrics(
    training_parquet: Path, make_model: Callable[..., OplmForMaskedLM]
) -> None:
    """``SequenceEvalTask.evaluate`` yields finite loss/accuracy/perplexity."""
    from accelerate import Accelerator

    cfg = _root_cfg()
    model = make_model(max_position_embeddings=_MAX_SEQ_LEN)
    entry = EvalDatasetEntry(
        name="hd",
        path=str(training_parquet),
        type="sequence",
        schedule=ScheduleSpec("steps", 1),
    )
    task = SequenceEvalTask(entry, cfg)
    accelerator = Accelerator(cpu=True)

    metrics = task.evaluate(model, accelerator)

    assert set(metrics) == {"loss", "accuracy", "perplexity"}
    assert all(math.isfinite(v) for v in metrics.values())
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["perplexity"] >= 1.0
