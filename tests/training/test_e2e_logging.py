"""G1 — logging cadence & train-metric contract (docs/TESTING_E2E.md §5).

Drives a real ~12-step :class:`~oplm.training.trainer.Trainer` run with
``log_every=3`` across the CPU/CUDA device matrix (single-rank, unsharded) and
asserts the observable logging contract: train metrics are emitted exactly on the
cadence, every payload carries the full ``train/*`` key set with finite values,
the sample/token counters advance by their hand-computed amounts, and the FLOP
accounting is internally consistent with ``estimate_flops_per_token``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from oplm.training.flops import estimate_flops_per_token
from tests.training.conftest import (
    DEVICE_PARAMS,
    FullRecordingCallback,
    force_device,
    tiny_train_cfg,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.slow

_TRAIN_KEYS = (
    "train/loss",
    "train/lr",
    "train/epoch",
    "train/samples",
    "train/tokens",
    "train/flops",
)
_MAX_STEPS = 12
_LOG_EVERY = 3
_BATCH_SIZE = 4


@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_logging_cadence_and_metric_contract(
    device: str,
    training_parquet: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Train logs land on multiples of ``log_every`` with a complete, finite payload."""
    from oplm.training.trainer import Trainer

    force_device(device, monkeypatch)
    cfg = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=_MAX_STEPS,
        batch_size=_BATCH_SIZE,
        log_every=_LOG_EVERY,
    )
    callback = FullRecordingCallback()
    Trainer(cfg, callbacks=[callback]).train()

    # Lifecycle brackets fire exactly once each.
    assert callback.train_start_count == 1
    assert callback.train_end_count == 1

    # Train metrics are emitted exactly on multiples of log_every (12 steps / 3).
    assert callback.train_log_steps == [3, 6, 9, 12]

    flops_per_token = estimate_flops_per_token(cfg.model)
    prev_tokens = 0
    prev_samples = 0
    for step, metrics in callback.train_logs:
        # Every train payload carries the full key set, all finite.
        for key in _TRAIN_KEYS:
            assert key in metrics, f"missing {key} at step {step}"
            assert math.isfinite(metrics[key]), f"{key} not finite at step {step}"

        # samples is deterministic: one full batch per opt step (single epoch, no
        # accumulation), counted once per micro-batch.
        assert metrics["train/samples"] == _BATCH_SIZE * step

        # Counters are monotonically non-decreasing.
        assert metrics["train/tokens"] >= prev_tokens
        assert metrics["train/samples"] >= prev_samples
        prev_tokens = metrics["train/tokens"]
        prev_samples = metrics["train/samples"]

        # Real data means real (positive) token counts, and FLOPs are exactly the
        # per-token estimate times the cumulative token count.
        assert metrics["train/tokens"] > 0
        assert metrics["train/flops"] == flops_per_token * metrics["train/tokens"]
