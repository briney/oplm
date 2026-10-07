"""``train.branch_from``: full-state branch under the live LR / weight decay / schedule."""

from __future__ import annotations

import logging
import shutil
from typing import TYPE_CHECKING

import pytest

from oplm.training.checkpoint import _unwrap_optimizer
from oplm.training.optim import get_schedule_fn
from tests.training.conftest import FullRecordingCallback, tiny_train_cfg

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.slow

_BATCH_SIZE = 4
_COMMON = dict(batch_size=_BATCH_SIZE, log_every=1, scheduler="wsd_linear", warmup_steps=0)


def _train_main(training_parquet: Path, tmp_path: Path) -> Path:
    """Train the 'main' run to step 4 at lr 1e-3 and return its checkpoint."""
    from oplm.training.trainer import Trainer

    cfg = tiny_train_cfg(
        tmp_path / "main",
        training_parquet,
        max_steps=4,
        save_every=4,
        lr=1e-3,
        stable_steps=4,
        weight_decay=0.01,
        **_COMMON,
    )
    Trainer(cfg, callbacks=[]).train()
    return tmp_path / "main" / "checkpoint-4"


def test_branch_keeps_state_but_takes_live_hyperparameters(
    training_parquet: Path, tmp_path: Path
) -> None:
    """A branch resumes counters + data position, yet runs the live lr/wd/schedule."""
    from oplm.training.trainer import Trainer

    ckpt = _train_main(training_parquet, tmp_path)

    lr, wd = 1e-4, 0.1
    cfg = tiny_train_cfg(
        tmp_path / "branch",
        training_parquet,
        max_steps=8,
        lr=lr,
        stable_steps=4,  # decay over steps 4 -> 8
        weight_decay=wd,
        branch_from=str(ckpt),
        **_COMMON,
    )
    cb = FullRecordingCallback()
    trainer = Trainer(cfg, callbacks=[cb])

    # Counters restored; the very first branch step already uses the live LR at λ(4).
    assert trainer.global_step == 4
    schedule_fn = get_schedule_fn("wsd_linear", warmup_steps=0, total_steps=8, stable_steps=4)
    inner = _unwrap_optimizer(trainer.optimizers[0])
    assert inner.param_groups[0]["lr"] == pytest.approx(lr * schedule_fn(4))
    assert sorted({g["weight_decay"] for g in inner.param_groups}) == [0.0, wd]

    trainer.train()
    post = dict(cb.train_logs)
    assert sorted(post) == [5, 6, 7, 8]
    # Live schedule, not the checkpoint's: step-5 LR is the branch's own decay curve.
    assert post[5]["train/lr"] == pytest.approx(lr * schedule_fn(5), rel=1e-5)
    assert post[8]["train/lr"] == pytest.approx(0.0, abs=1e-12)

    # Data position carried over: tokens match an uninterrupted 0 -> 8 control run.
    control_cfg = tiny_train_cfg(
        tmp_path / "control",
        training_parquet,
        max_steps=8,
        lr=1e-3,
        stable_steps=4,
        weight_decay=0.01,
        **_COMMON,
    )
    control = Trainer(control_cfg, callbacks=[])
    control.train()
    assert trainer.tokens_seen == control.tokens_seen


def test_branch_with_different_data_drops_the_cursor(
    training_parquet: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from oplm.training.trainer import Trainer

    ckpt = _train_main(training_parquet, tmp_path)
    other_data = tmp_path / "other.parquet"
    shutil.copy(training_parquet, other_data)

    cfg = tiny_train_cfg(
        tmp_path / "branch",
        other_data,
        max_steps=8,
        stable_steps=4,
        branch_from=str(ckpt),
        **_COMMON,
    )
    with caplog.at_level(logging.WARNING):
        trainer = Trainer(cfg, callbacks=[])
    assert "dropping its data cursor" in caplog.text
    assert trainer._batches_in_epoch == 0
    assert trainer.global_step == 4


def test_branch_refuses_a_finished_budget(training_parquet: Path, tmp_path: Path) -> None:
    from oplm.training.trainer import Trainer

    ckpt = _train_main(training_parquet, tmp_path)
    cfg = tiny_train_cfg(
        tmp_path / "branch",
        training_parquet,
        max_steps=4,
        stable_steps=4,
        branch_from=str(ckpt),
        **_COMMON,
    )
    with pytest.raises(ValueError, match="nothing left to train"):
        Trainer(cfg, callbacks=[])
