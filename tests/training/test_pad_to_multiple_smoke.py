"""Smoke test for ``data.pad_to_multiple_of`` — runs in the FAST (non-slow) suite.

This test is intentionally compile-off and CPU-fast (~1 s).  It lives in its own
module (rather than ``test_pilot_train.py``) so it is NOT covered by the
module-level ``pytestmark = pytest.mark.slow`` there and therefore always runs in
the default ``python -m pytest -m "not slow"`` pass.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from tests.training.conftest import FullRecordingCallback, tiny_train_cfg

if TYPE_CHECKING:
    from pathlib import Path


def test_pad_to_multiple_smoke(training_parquet: Path, tmp_path: Path) -> None:
    """pad_to_multiple_of=16 + throughput_warmup_steps=0 smoke: completes, evals, logs tput.

    Asserts:
    - Training completes the full step budget with no shape mismatch.
    - The eval pass fires at least once and every eval metric is finite.
    - ``train/tokens_per_sec`` is present in at least one logged metric payload
      (throughput_warmup_steps=0 ensures warmup exclusion is bypassed for this
      short run).

    ``compile`` is left OFF so no real compilation occurs — this test is intentionally
    CPU-fast for the non-GPU CI tier.
    """
    from oplm.training.trainer import Trainer

    cfg = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=4,
        log_every=1,
        pad_to_multiple_of=16,
        throughput_warmup_steps=0,
        eval={
            "hd": {
                "path": str(training_parquet),
                "type": "sequence",
                "every": {"steps": 2},
            }
        },
        compile=False,
    )
    callback = FullRecordingCallback()
    Trainer(cfg, callbacks=[callback]).train()

    # (a) Training completes with no shape mismatch (no exception raised above).
    assert callback.train_end_count == 1

    # (b) The eval pass ran and all metrics are finite.
    assert callback.evals, "eval never fired"
    for _step, eval_metrics in callback.evals:
        for key, val in eval_metrics.items():
            assert math.isfinite(val), f"eval metric {key!r} is not finite"

    # (c) train/tokens_per_sec was emitted in at least one log payload.
    tput_steps = [step for step, m in callback.train_logs if "train/tokens_per_sec" in m]
    assert tput_steps, "train/tokens_per_sec was never logged (throughput_warmup_steps=0 should enable it)"
