"""G4 — gradient accumulation through the real Trainer (docs/TESTING_E2E.md §5).

With ``gradient_accumulation_steps=4`` the trainer must take exactly one optimizer
step per four micro-batches, log the mean loss across those micro-batches (not the
last one), and reduce tokens once per optimizer step. We verify all three by
wrapping the prepared model's ``forward`` to capture each micro-batch's loss and
token count, then reconciling them against the per-step logged metrics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.training.conftest import FullRecordingCallback, tiny_train_cfg

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.slow

_BATCH_SIZE = 4
_ACCUM = 4
_MAX_STEPS = 3


def test_gradient_accumulation_step_loss_and_tokens(training_parquet: Path, tmp_path: Path) -> None:
    """One opt step per 4 micro-batches; logged loss is their mean; tokens reduce once/step."""
    from oplm.training.trainer import Trainer

    cfg = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=_MAX_STEPS,
        batch_size=_BATCH_SIZE,
        gradient_accumulation_steps=_ACCUM,
        log_every=1,
    )
    callback = FullRecordingCallback()
    trainer = Trainer(cfg, callbacks=[callback])

    # Capture each micro-batch's loss + token count by wrapping the prepared model's
    # forward (nn.Module.__call__ dispatches through the instance's `forward`).
    micro: list[tuple[float, int]] = []
    orig_forward = trainer.model.forward

    def recording_forward(*args: object, **kwargs: object):  # noqa: ANN202 - passthrough
        out = orig_forward(*args, **kwargs)
        loss = out["loss"].detach().item()
        tokens = int(kwargs["attention_mask"].sum().item())
        micro.append((loss, tokens))
        return out

    trainer.model.forward = recording_forward  # type: ignore[method-assign]
    trainer.train()

    # Exactly one optimizer step per `_ACCUM` micro-batches.
    assert trainer.global_step == _MAX_STEPS
    assert callback.train_log_steps == [1, 2, 3]
    assert len(micro) == _MAX_STEPS * _ACCUM  # 12 micro-batches, 3 opt steps

    micro_losses = [loss for loss, _ in micro]
    micro_tokens = [tokens for _, tokens in micro]
    logs = dict(callback.train_logs)

    for step in range(1, _MAX_STEPS + 1):
        window = slice((step - 1) * _ACCUM, step * _ACCUM)

        # Logged loss is the mean across the optimizer step's micro-batches.
        expected_loss = sum(micro_losses[window]) / _ACCUM
        assert logs[step]["train/loss"] == pytest.approx(expected_loss, abs=1e-6)

        # samples advance by one full effective batch per opt step (counted per micro-batch).
        assert logs[step]["train/samples"] == _BATCH_SIZE * _ACCUM * step

        # tokens are accumulated across the 4 micro-batches and reduced once per opt step.
        assert logs[step]["train/tokens"] == sum(micro_tokens[: step * _ACCUM])
