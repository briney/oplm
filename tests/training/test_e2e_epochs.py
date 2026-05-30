"""G8 — epoch-bounded runs through the real Trainer (docs/TESTING_E2E.md §5).

Runs ``max_epochs=2`` over the 16-row tiny parquet with ``batch_size=4`` so an
epoch is exactly four optimizer steps and the second epoch actually crosses the
``StopIteration -> epoch++ -> set_epoch`` boundary. Asserts the total-steps epoch
formula, that the epoch counter advanced, and that the fractional ``train/epoch``
metric is monotone and lands exactly on 2.0.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from tests.training.conftest import FullRecordingCallback, tiny_train_cfg

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.slow

_BATCH_SIZE = 4
_MAX_EPOCHS = 2


def test_epoch_bounded_run_crosses_boundary(tiny_training_parquet: Path, tmp_path: Path) -> None:
    """A 2-epoch run computes total steps by the epoch formula and crosses the boundary."""
    from oplm.training.trainer import Trainer

    cfg = tiny_train_cfg(
        tmp_path,
        tiny_training_parquet,
        max_epochs=_MAX_EPOCHS,
        batch_size=_BATCH_SIZE,
        log_every=1,
    )
    callback = FullRecordingCallback()
    trainer = Trainer(cfg, callbacks=[callback])

    # The 16-row fixture drives the epoch formula: ceil(16/4) * 2 = 8 steps.
    assert trainer._dataset_size == 16
    expected_total = math.ceil(trainer._dataset_size / _BATCH_SIZE) * _MAX_EPOCHS
    assert trainer.total_steps == expected_total == 8

    trainer.train()
    assert trainer.global_step == 8

    # The epoch counter advanced exactly once (epoch 0 -> 1 at the StopIteration boundary).
    assert trainer.epoch == 1

    # Fractional train/epoch is monotone non-decreasing, crosses 1.0, and lands on 2.0.
    epoch_series = [m["train/epoch"] for _, m in callback.train_logs]
    assert len(epoch_series) == 8
    assert epoch_series == sorted(epoch_series)
    assert max(epoch_series) > 1.0  # the second epoch was actually entered
    assert epoch_series[-1] == pytest.approx(2.0)
