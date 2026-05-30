"""G9 — gradient checkpointing through the real Trainer (docs/TESTING_E2E.md §5).

``gradient_checkpointing=True`` trades compute for memory by recomputing block
activations in the backward pass; it must be mathematically transparent. We run
two CPU trainers with the same seed and data — checkpointing on vs off — and
assert the on-run is finite and its per-step loss trajectory matches the off-run.
CPU (manual attention, dropout off) keeps recomputation deterministic, and the
training mask stream is unaffected because the forward consumes no global RNG.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from tests.training.conftest import (
    FullRecordingCallback,
    configure_accelerator_device,
    tiny_train_cfg,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.slow


def _run_losses(cfg, callback) -> dict[int, float]:
    from oplm.training.trainer import Trainer

    Trainer(cfg, callbacks=[callback]).train()
    return {step: m["train/loss"] for step, m in callback.train_logs}


def test_gradient_checkpointing_matches_plain_run(
    training_parquet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Checkpointed and non-checkpointed runs produce the same loss trajectory."""
    configure_accelerator_device("cpu", monkeypatch)

    on_cb = FullRecordingCallback()
    on_losses = _run_losses(
        tiny_train_cfg(
            tmp_path / "on",
            training_parquet,
            max_steps=5,
            batch_size=4,
            log_every=1,
            gradient_checkpointing=True,
        ),
        on_cb,
    )

    off_cb = FullRecordingCallback()
    off_losses = _run_losses(
        tiny_train_cfg(
            tmp_path / "off",
            training_parquet,
            max_steps=5,
            batch_size=4,
            log_every=1,
            gradient_checkpointing=False,
        ),
        off_cb,
    )

    assert sorted(on_losses) == sorted(off_losses) == [1, 2, 3, 4, 5]
    assert all(math.isfinite(v) for v in on_losses.values())

    # Step 1 (pre-any-update) is identical; later steps match within tolerance.
    assert on_losses[1] == pytest.approx(off_losses[1], abs=1e-6)
    for step in range(1, 6):
        assert on_losses[step] == pytest.approx(off_losses[step], rel=1e-4, abs=1e-4)
