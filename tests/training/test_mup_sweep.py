"""Slow test for the μP LR-sweep metric utilities (Phase 5).

Drives two tiny real-data training runs at one width across a 2-point LR grid,
each with a :class:`~oplm.training.mup.SweepMetricsCallback`, then checks the
sweep summary path: every run writes ``metrics.json``, and
:func:`~oplm.training.mup.best_lr_per_width` selects the lower-loss LR and
populates the transfer verdict. A clearly-too-small LR (barely trains) loses to a
working LR, so the argmin is deterministic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from oplm.training.mup import SweepMetricsCallback, best_lr_per_width, summarize_sweep
from tests.training.conftest import tiny_train_cfg

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.slow

# A frozen LR barely moves the loss in 30 steps; the working LR reduces it. The
# working LR is therefore the unambiguous argmin (see test_e2e_optim's learn test).
_FROZEN_LR = 1e-6
_WORKING_LR = 1e-3


def test_sweep_selects_lower_loss_lr(training_parquet: Path, tmp_path: Path) -> None:
    """Each grid point writes metrics.json; best_lr picks the working LR; verdict set."""
    from oplm.training.trainer import Trainer

    run_dirs: list[Path] = []
    width: int | None = None
    for lr in (_FROZEN_LR, _WORKING_LR):
        out = tmp_path / f"lr_{lr:.0e}"
        cfg = tiny_train_cfg(out, training_parquet, max_steps=30, batch_size=8, lr=lr, log_every=1)
        # Enable μP as a faithful no-op (base width == this width, so m = 1): the
        # sweep utilities are width-keyed and optimizer-agnostic, and m = 1 keeps
        # per-group LRs at cfg.lr so the loss signal is clean.
        cfg.model.mup_enable = True
        cfg.model.mup_base_width = cfg.model.hidden_size
        width = cfg.model.hidden_size

        Trainer(cfg, callbacks=[SweepMetricsCallback(out / "metrics.json")]).train()
        assert (out / "metrics.json").exists()
        run_dirs.append(out)

    df = summarize_sweep(run_dirs)
    assert len(df) == 2
    assert df["final_train_loss"].notna().all()

    result = best_lr_per_width(df)
    assert result.best_lr[width] == pytest.approx(_WORKING_LR)
    # One width was swept, so transfer is undecidable — but the verdict is populated.
    assert isinstance(result.transferred, bool)
    assert result.transferred is False
