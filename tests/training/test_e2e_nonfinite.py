"""E2E non-finite loss guard test (Task 1.8): a NaN loss aborts training immediately and
leaves the last good checkpoint intact.

Unlike the drain test (`test_e2e_drain.py`), this scenario needs no real signal delivery,
so it drives the real `Trainer` in-process (same pattern as
`test_e2e_accumulation.py::test_gradient_accumulation_step_loss_and_tokens`): wrap the
prepared model's `forward` to inject a NaN loss on exactly one micro-batch, then assert
the trainer raises `RuntimeError` naming the step it happened at, and that the checkpoint
committed just before it is untouched.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from tests.training.conftest import configure_accelerator_device, tiny_train_cfg

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.slow

_NAN_AT_STEP = 3
_LAST_GOOD_STEP = 2


class _NanAtCall:
    """Wrap a model `forward` to inject a NaN loss on exactly one call."""

    def __init__(self, original, nan_at_call: int) -> None:
        self._original = original
        self._nan_at_call = nan_at_call
        self._calls = 0

    def __call__(self, *args: object, **kwargs: object):
        self._calls += 1
        outputs = self._original(*args, **kwargs)
        if self._calls == self._nan_at_call:
            outputs["loss"] = outputs["loss"] * float("nan")
        return outputs


def test_nonfinite_loss_aborts_training_and_preserves_last_checkpoint(
    tmp_path: Path,
    training_parquet: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A NaN loss at step 3 raises `RuntimeError` mentioning step 3; checkpoint-2 survives."""
    from oplm.training.trainer import Trainer

    configure_accelerator_device("cpu", monkeypatch)
    run_dir = tmp_path / "run"

    cfg = tiny_train_cfg(
        run_dir,
        training_parquet,
        max_steps=10,
        save_every=_LAST_GOOD_STEP,
        save_final=False,
        log_every=1,
    )
    trainer = Trainer(cfg)
    trainer.model.forward = _NanAtCall(  # type: ignore[method-assign]
        trainer.model.forward, nan_at_call=_NAN_AT_STEP
    )

    with pytest.raises(RuntimeError, match=f"non-finite training loss at step {_NAN_AT_STEP}"):
        trainer.train()

    checkpoint_dir = run_dir / f"checkpoint-{_LAST_GOOD_STEP}"
    assert checkpoint_dir.is_dir(), "the step-2 checkpoint must have been committed"
    state = json.loads((checkpoint_dir / "trainer_state.json").read_text())
    assert state["global_step"] == _LAST_GOOD_STEP

    # No torn staging/replace dirs, and nothing was ever written for the poisoned step.
    assert not any(run_dir.glob(f"checkpoint-{_LAST_GOOD_STEP}.tmp"))
    assert not any(run_dir.glob(f"checkpoint-{_LAST_GOOD_STEP}.old"))
    assert not (run_dir / f"checkpoint-{_NAN_AT_STEP}").exists()
