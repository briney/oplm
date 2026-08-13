"""Phase 3 acceptance gate — data-exact resume (docs/TESTING_E2E.md; TODOS Task 3.4).

Tasks 3.1-3.3 made resume *data-exact*: a checkpoint records a :class:`~oplm.data.DataCursor`
(epoch, batches-in-epoch, layout) and, on a layout-compatible resume, the top-level dataset's
``set_resume_skip`` arms an index-arithmetic skip so the resumed stream is the exact suffix of
what an uninterrupted run would have produced — no row lost, none re-seen, none reordered
(modulo the worker-cycle phase caveat Task 3.3's save-alignment deferral exists to close: a
periodic/drain save is deferred, at most ``num_workers - 1`` extra steps, until
``batches_in_epoch % num_workers == 0``, so a checkpoint is only ever taken on a boundary where
the *global* interleaving order — not just each worker's own substream — matches the control).

This module is the property's acceptance test: for both ``num_workers=0`` (no ``DataLoader``
worker processes) and ``num_workers=2`` (the round-robin/alignment machinery actually
exercised), an INTERRUPTED run (train to step 6, checkpoint, fresh ``Trainer`` with
``auto_resume=true`` finishing to step 12) must reproduce a CONTROL run's (uninterrupted,
straight to step 12) per-step ``sequence_ids`` stream *exactly* when the interrupted run's two
halves are concatenated, and the final ``tokens_seen`` must match exactly too. Two fixture
sizes are used so the checkpoint lands mid-epoch in one variant and inside epoch 1 (having
already crossed one epoch boundary) in the other — proving ``epoch`` and ``batches_in_epoch``
compose correctly across the rollover, not just within a single epoch.

Masking is untouched by this tier: masks are dynamic (redrawn every call, docs/DATA_TOOLING.md
§4.5) and may differ bit-for-bit between the control and the interrupted-then-resumed run even
when the underlying rows match exactly. The assertion is on ``sequence_ids`` (which rows, in
which order) and ``tokens_seen`` (how much real content), never on ``input_ids``/``labels``.

Recorder mechanism: ``MLMCollator`` already supports ``keep_sequence_ids=True`` (Task 0.1), but
``build_train_dataloader`` never sets it. Rather than adding a production config knob for a
test-only need, ``_force_keep_sequence_ids`` monkeypatches the ``MLMCollator`` name inside
``oplm.data.sequence.loaders`` (the only place ``build_train_dataloader`` looks it up) to a
thin subclass that always forces the flag on; every other kwarg passes through unchanged, and
no production file changes. Batches themselves are captured by ``_SequenceIdRecorder``, which
wraps ``Trainer.dataloader`` (a :class:`~oplm.data.DeviceDataLoader`) *after* construction and
delegates ``dataset``/``set_epoch``/``__len__`` — the trainer never sees the difference, and no
``TrainerCallback`` hook is needed (callbacks never see the batch itself).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.training.conftest import FullRecordingCallback, tiny_train_cfg

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from torch import Tensor

    from oplm.data import DeviceDataLoader
    from oplm.training.trainer import Trainer

pytestmark = pytest.mark.slow

_BATCH_SIZE = 4
_TOTAL_STEPS = 12
_STOP_STEP = 6


class _SequenceIdRecorder:
    """Wraps a :class:`~oplm.data.DeviceDataLoader`, recording each yielded batch's ids.

    Delegates every attribute :class:`Trainer` actually uses on ``self.dataloader``
    (``dataset``, ``set_epoch``, ``__len__``) to the wrapped loader, so swapping
    ``trainer.dataloader`` for one of these after construction is invisible to the
    trainer — ``train()`` keeps calling ``iter(self.dataloader)`` exactly as before,
    once per epoch, and each call is recorded.
    """

    def __init__(self, inner: DeviceDataLoader) -> None:
        self._inner = inner
        self.batches: list[list[str]] = []

    def __iter__(self) -> Iterator[dict[str, Tensor | list[str]]]:
        for batch in self._inner:
            self.batches.append(list(batch["sequence_ids"]))
            yield batch

    def __len__(self) -> int:
        return len(self._inner)

    @property
    def dataset(self) -> object:
        return self._inner.dataset

    def set_epoch(self, epoch: int) -> None:
        self._inner.set_epoch(epoch)

    def flat(self) -> list[str]:
        """Return every recorded batch's ids, concatenated in yield order."""
        return [sid for batch in self.batches for sid in batch]


def _force_keep_sequence_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force every ``MLMCollator`` built by ``build_train_dataloader`` to keep ids.

    ``build_train_dataloader`` (``oplm.data.sequence.loaders``) constructs its own
    ``MLMCollator`` with no ``cfg`` knob for ``keep_sequence_ids`` — training never
    needs it, only this test does. Monkeypatches the ``MLMCollator`` name bound
    inside the ``loaders`` module (the only place ``build_train_dataloader`` looks
    it up) to a thin subclass that always passes ``keep_sequence_ids=True`` through;
    no other behavior changes, and no production file is touched.
    """
    from oplm.data.sequence import loaders as loaders_mod
    from oplm.data.sequence.collate import MLMCollator

    class _RecordingCollator(MLMCollator):
        def __init__(self, *args: object, **kwargs: object) -> None:
            kwargs["keep_sequence_ids"] = True
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(loaders_mod, "MLMCollator", _RecordingCollator)


def _wrap_recorder(trainer: Trainer) -> _SequenceIdRecorder:
    """Swap ``trainer.dataloader`` for a recording wrapper around the same loader."""
    recorder = _SequenceIdRecorder(trainer.dataloader)
    trainer.dataloader = recorder  # type: ignore[assignment]
    return recorder


def _assert_data_exact_resume(train_data: Path, tmp_path: Path, num_workers: int) -> None:
    """Control (0->12) vs. interrupted (0->6, checkpoint) + resumed (auto_resume->12).

    Asserts the concatenated ``sequence_ids`` stream of the interrupted run's two
    halves equals the control's stream exactly, and that final ``tokens_seen``
    matches exactly. Does not assume which step the checkpoint that resume actually
    uses lands on (Task 3.3's worker-cycle save-alignment deferral can push a
    ``save_every=5`` trigger to a later, worker-aligned step) — it is read back from
    the recording callback and checked, never hardcoded.
    """
    from oplm.training.trainer import Trainer

    common = dict(batch_size=_BATCH_SIZE, num_workers=num_workers, log_every=1)

    # Control: a single uninterrupted run straight to the final target. No checkpoint
    # needed, so periodic/final saving is disabled entirely to keep this cheap.
    control_cfg = tiny_train_cfg(
        tmp_path / "control",
        train_data,
        max_steps=_TOTAL_STEPS,
        save_every=10_000,
        save_final=False,
        **common,
    )
    control = Trainer(control_cfg, callbacks=[])
    control_recorder = _wrap_recorder(control)
    control.train()

    # Interrupted run, phase 1: train only to step 6. save_every=5 exercises the
    # worker-cycle alignment machinery for num_workers=2 (the trigger at step 5 is
    # deferred to the next aligned batch count); save_final backstops a checkpoint
    # at the actual stop step regardless of where the periodic trigger landed.
    interrupt_dir = tmp_path / "interrupt"
    cfg1 = tiny_train_cfg(
        interrupt_dir,
        train_data,
        max_steps=_STOP_STEP,
        save_every=5,
        **common,
    )
    callback = FullRecordingCallback()
    run1 = Trainer(cfg1, callbacks=[callback])
    run1_recorder = _wrap_recorder(run1)
    run1.train()

    assert callback.checkpoint_steps, "run1 must have saved at least one checkpoint"
    checkpoint_step = callback.checkpoint_steps[-1]
    # The checkpoint resume actually uses reflects everything run1 consumed: nothing
    # is re-derived or duplicated across the boundary.
    assert checkpoint_step == run1.global_step == _STOP_STEP

    # Interrupted run, phase 2: a fresh Trainer, auto_resume finds the newest
    # committed checkpoint in interrupt_dir and finishes the rest of the schedule.
    cfg2 = tiny_train_cfg(
        interrupt_dir,
        train_data,
        max_steps=_TOTAL_STEPS,
        save_every=10_000,
        save_final=False,
        auto_resume=True,
        **common,
    )
    run2 = Trainer(cfg2, callbacks=[])
    run2_recorder = _wrap_recorder(run2)
    run2.train()

    combined = run1_recorder.flat() + run2_recorder.flat()
    control_stream = control_recorder.flat()
    assert len(combined) == len(control_stream) == _TOTAL_STEPS * _BATCH_SIZE
    assert combined == control_stream
    assert run2.global_step == control.global_step == _TOTAL_STEPS
    assert run2.tokens_seen == control.tokens_seen


@pytest.mark.parametrize("num_workers", [0, 2])
def test_resume_matches_control_exactly_mid_epoch(
    training_parquet: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    num_workers: int,
) -> None:
    """256-row fixture: 64 batches/epoch keeps all 12 steps inside epoch 0.

    The checkpoint lands mid-epoch (never near a boundary), isolating the
    within-epoch skip arithmetic from any epoch-rollover interaction.
    """
    _force_keep_sequence_ids(monkeypatch)
    _assert_data_exact_resume(training_parquet, tmp_path, num_workers)


@pytest.mark.parametrize("num_workers", [0, 2])
def test_resume_matches_control_exactly_across_epoch_boundary(
    tiny_training_parquet: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    num_workers: int,
) -> None:
    """16-row fixture: 4 batches/epoch, so the checkpoint (~step 5-6) sits in epoch 1.

    Proves ``epoch`` + ``batches_in_epoch`` compose across the rollover: the
    checkpoint used for resume was taken after one full epoch already elapsed, not
    within epoch 0.
    """
    _force_keep_sequence_ids(monkeypatch)
    _assert_data_exact_resume(tiny_training_parquet, tmp_path, num_workers)
