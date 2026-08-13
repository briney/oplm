"""G3 — checkpointing, rotation & resume equivalence (docs/TESTING_E2E.md §5).

The first test drives a real run with ``save_every`` + ``save_total_limit`` and
asserts exactly the expected ``checkpoint-*`` set survives rotation, each carrying
the resumable state, a reloadable ``config.yaml``, and a ``from_pretrained``-ready
``hf/`` export. The second test checks resume *equivalence*: a fresh trainer
resumed from a mid-run checkpoint restores the training counters, the LR schedule
position, reaches the new target, and continues with a finite, non-discontinuous
loss.

Note on scope: OPLM's training stream is an ``IterableDataset`` that restarts at
the top of the epoch when a resumed trainer rebuilds its iterator, so the exact
sequence of post-resume batches (and therefore ``tokens_seen``) does not match an
uninterrupted run. We assert counter *restoration* and trajectory *continuity*
rather than bit-exact equivalence against a non-resumed control.
"""

from __future__ import annotations

import json
import logging
import math
import time as _real_time
from typing import TYPE_CHECKING

import pytest

from oplm.training.optim import get_schedule_fn
from tests.training.conftest import FullRecordingCallback, tiny_train_cfg

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

pytestmark = pytest.mark.slow

_BATCH_SIZE = 4


class _FakeClock:
    """A fake monotonic/wall clock that advances by a fixed step on every call."""

    def __init__(self, step: float = 30.0, start: float = 0.0) -> None:
        self._value = start
        self._step = step

    def __call__(self) -> float:
        value = self._value
        self._value += self._step
        return value


class _FakeTimeModule:
    """Stand-in for the ``time`` module-level name bound inside ``trainer.py``.

    Installed via ``monkeypatch.setattr("oplm.training.trainer.time", ...)``, which
    replaces trainer.py's own ``time`` *name binding* rather than mutating an attribute
    on the real, shared ``time`` module. That distinction matters: since Task 2.1,
    ``save_checkpoint`` calls ``torch.distributed.checkpoint`` (DCP), which instruments
    its own save/load durations with real ``time.time()`` calls internally. Patching the
    global ``time.time`` slot (as this file used to do) would let those unrelated calls
    silently consume "ticks" from a fake clock shared process-wide, desynchronizing it
    from what the elapsed-time math below expects. Scoping the patch to trainer.py's own
    ``time`` reference means only trainer.py's ``time.time()``/``time.monotonic()``
    calls ever see the fake clock; DCP's own ``import time`` elsewhere is untouched.
    """

    def __init__(
        self,
        *,
        time_fn: Callable[[], float] | None = None,
        monotonic_fn: Callable[[], float] | None = None,
    ) -> None:
        self.time = time_fn if time_fn is not None else _real_time.time
        self.monotonic = monotonic_fn if monotonic_fn is not None else _real_time.monotonic
        # trainer.py also calls time.perf_counter() for throughput timing; no test in
        # this file fakes it, so it always passes through to the real clock.
        self.perf_counter = _real_time.perf_counter


def _checkpoint_names(output_dir: Path) -> list[str]:
    return sorted(d.name for d in output_dir.iterdir() if d.name.startswith("checkpoint-"))


def test_rotation_and_checkpoint_artifacts(training_parquet: Path, tmp_path: Path) -> None:
    """Rotation keeps the newest ``save_total_limit`` dirs, each fully populated."""
    from oplm.training.trainer import Trainer

    cfg = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=6,
        batch_size=_BATCH_SIZE,
        save_every=2,
        save_total_limit=2,
    )
    callback = FullRecordingCallback()
    Trainer(cfg, callbacks=[callback]).train()

    # Saves fire on the cadence (2, 4, 6); the final save is de-duplicated because
    # the last step (6) already triggered a periodic save.
    assert callback.checkpoint_steps == [2, 4, 6]

    # The fully resolved config is dropped at the top level of the run directory.
    assert (tmp_path / "config.yaml").exists()

    # Rotation keeps only the two newest checkpoint directories.
    assert _checkpoint_names(tmp_path) == ["checkpoint-4", "checkpoint-6"]

    for name in ("checkpoint-4", "checkpoint-6"):
        ckpt = tmp_path / name
        assert (ckpt / "trainer_state.json").exists()
        assert (ckpt / "config.yaml").exists()
        hf = ckpt / "hf"
        assert (hf / "config.json").exists()
        assert (hf / "model.safetensors").exists()
        # Tokenizer round-trip files for from_pretrained.
        assert (hf / "tokenizer_config.json").exists()
        assert (hf / "tokenizer.json").exists()


def test_save_final_writes_unaligned_final_checkpoint(
    training_parquet: Path, tmp_path: Path
) -> None:
    """``save_final`` guarantees a final checkpoint even when the last step is off-cadence."""
    from oplm.training.trainer import Trainer

    # save_every=4 over max_steps=6 -> periodic save only at step 4. The final
    # checkpoint at step 6 comes solely from the save_final path.
    cfg = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=6,
        batch_size=_BATCH_SIZE,
        save_every=4,
    )
    callback = FullRecordingCallback()
    Trainer(cfg, callbacks=[callback]).train()
    assert callback.checkpoint_steps == [4, 6]


def test_save_final_disabled_skips_final_checkpoint(training_parquet: Path, tmp_path: Path) -> None:
    """``save_final=False`` leaves only the periodic saves; no off-cadence final save."""
    from oplm.training.trainer import Trainer

    cfg = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=6,
        batch_size=_BATCH_SIZE,
        save_every=4,
        save_final=False,
    )
    callback = FullRecordingCallback()
    Trainer(cfg, callbacks=[callback]).train()
    assert callback.checkpoint_steps == [4]


def test_save_every_minutes_triggers_time_based_checkpoint(
    training_parquet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A save_every_minutes timer fires a checkpoint independent of save_every.

    The fake monotonic clock advances 30s per call; ``Trainer.__init__`` consumes
    the first tick (anchoring ``_last_save_at`` at t=0), so the per-step due-check
    sees t=30 at step 1 (not due) and t=60 at step 2 — exactly the 60s
    (save_every_minutes=1) threshold, so the checkpoint appears at step 2.
    """
    from oplm.training.trainer import Trainer

    monkeypatch.setattr(
        "oplm.training.trainer.time", _FakeTimeModule(monotonic_fn=_FakeClock(step=30.0))
    )

    cfg = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=3,
        batch_size=_BATCH_SIZE,
        save_every=0,
        save_every_minutes=1,
        save_final=False,
    )
    callback = FullRecordingCallback()
    Trainer(cfg, callbacks=[callback]).train()

    assert callback.checkpoint_steps == [2]


def test_keep_every_n_hours_marks_crossing_checkpoint_permanent(
    training_parquet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Crossing a keep_every_n_hours boundary marks that checkpoint permanent.

    The fake wall clock advances 2000s per checkpoint; with keep_every_n_hours=1
    (3600s), the anchor is set at the step-1 checkpoint (t=0) and the boundary is
    crossed at the step-3 checkpoint (elapsed=4000s >= 3600s). save_total_limit=1
    means only the newest rolling checkpoint would normally survive — the KEEP
    marker on checkpoint-3 exempts it, and it is the newest anyway.
    """
    from oplm.training.trainer import Trainer

    monkeypatch.setattr(
        "oplm.training.trainer.time", _FakeTimeModule(time_fn=_FakeClock(step=2000.0))
    )

    cfg = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=3,
        batch_size=_BATCH_SIZE,
        save_every=1,
        save_total_limit=1,
        keep_every_n_hours=1.0,
    )
    callback = FullRecordingCallback()
    Trainer(cfg, callbacks=[callback]).train()

    assert callback.checkpoint_steps == [1, 2, 3]
    assert _checkpoint_names(tmp_path) == ["checkpoint-3"]
    assert (tmp_path / "checkpoint-3" / "KEEP").exists()

    state = json.loads((tmp_path / "checkpoint-3" / "trainer_state.json").read_text())
    assert state["first_checkpoint_unix"] == 0.0
    assert state["last_time_keep_index"] == 1


def test_first_checkpoint_unix_anchor_survives_resume(
    training_parquet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The keep_every_n_hours anchor is restored from trainer_state.json on resume.

    A requeue must not reset ``first_checkpoint_unix`` to the resumed trainer's
    own start time — otherwise a long-running job that requeues every few hours
    would never accumulate enough wall-clock time to cross a keep_every_n_hours
    boundary.
    """
    from oplm.training.trainer import Trainer

    monkeypatch.setattr(
        "oplm.training.trainer.time", _FakeTimeModule(time_fn=_FakeClock(step=100.0))
    )

    cfg1 = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=1,
        batch_size=_BATCH_SIZE,
        save_every=1,
        keep_every_n_hours=1.0,
    )
    Trainer(cfg1, callbacks=[]).train()

    state1 = json.loads((tmp_path / "checkpoint-1" / "trainer_state.json").read_text())
    assert state1["first_checkpoint_unix"] == 0.0

    # A fresh trainer resumes; even though its own wall clock keeps advancing,
    # the anchor loaded from trainer_state.json must be reused, not reset.
    cfg2 = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=2,
        batch_size=_BATCH_SIZE,
        save_every=1,
        keep_every_n_hours=1.0,
        resume_from=str(tmp_path / "checkpoint-1"),
    )
    resumed = Trainer(cfg2, callbacks=[])
    assert resumed._first_checkpoint_at == 0.0


def test_resume_restores_state_and_continues(
    training_parquet: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Resuming restores counters + LR position, reaches the new target, stays finite.

    Also covers the Task 2.2 fix round: extending max_steps (4 -> 8) across an explicit
    resume_from is a supported workflow, not a schedule-compat error, but it does log a
    prominent warning (see ``validate_schedule_compat``'s asymmetric max_steps policy).
    """
    from oplm.training.trainer import Trainer

    lr = 1e-3
    min_lr = 1e-4
    scheduler = "warmup_linear"
    common = dict(
        batch_size=_BATCH_SIZE,
        log_every=1,
        lr=lr,
        min_lr=min_lr,
        scheduler=scheduler,
        warmup_steps=0,
    )

    # Phase 1: train to step 4 and checkpoint.
    cfg1 = tiny_train_cfg(tmp_path, training_parquet, max_steps=4, save_every=4, **common)
    cb1 = FullRecordingCallback()
    Trainer(cfg1, callbacks=[cb1]).train()

    ckpt = tmp_path / "checkpoint-4"
    saved_state = json.loads((ckpt / "trainer_state.json").read_text())

    # Phase 2: a fresh trainer resumes from the checkpoint and targets step 8.
    cfg2 = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=8,
        save_every=8,
        resume_from=str(ckpt),
        **common,
    )
    cb2 = FullRecordingCallback()
    with caplog.at_level(logging.WARNING):
        resumed = Trainer(cfg2, callbacks=[cb2])

    # max_steps increased (4 -> 8): allowed, but warned about loudly.
    assert "max_steps" in caplog.text

    # Counters are restored from the checkpoint at construction time.
    assert resumed.global_step == saved_state["global_step"] == 4
    assert resumed.tokens_seen == saved_state["tokens_seen"]
    assert resumed._samples_seen == saved_state["samples_seen"]
    assert resumed.epoch == saved_state["epoch"]

    resumed.train()
    assert resumed.global_step == 8

    # The post-resume train logs cover exactly steps 5..8 and are all finite.
    post_resume = dict(cb2.train_logs)
    assert sorted(post_resume) == [5, 6, 7, 8]
    assert all(math.isfinite(m["train/loss"]) for m in post_resume.values())

    # The LR schedule resumes at its restored position: the step-5 LR matches the
    # total_steps=8 schedule at step 5, not a scheduler restarted from zero.
    schedule_fn = get_schedule_fn(scheduler, warmup_steps=0, total_steps=8, min_ratio=min_lr / lr)
    assert post_resume[5]["train/lr"] == pytest.approx(lr * schedule_fn(5), rel=1e-5)

    # No gross discontinuity at the resume boundary (catches lost optimizer state):
    # the first post-resume loss stays within a generous band of the last pre-resume loss.
    last_pre = dict(cb1.train_logs)[4]["train/loss"]
    first_post = post_resume[5]["train/loss"]
    assert 0.2 < first_post / last_pre < 5.0


# --- auto_resume (Task 1.4) -------------------------------------------------------------


def test_auto_resume_picks_up_the_newest_committed_checkpoint(
    training_parquet: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """``auto_resume=true`` with no explicit ``resume_from`` finds and resumes the checkpoint.

    Mirrors ``test_resume_restores_state_and_continues`` but exercises the scanning path
    (``Trainer`` discovers ``checkpoint-4`` under ``output_dir`` itself) instead of an
    operator-pinned ``resume_from`` -- the requeue scenario auto_resume exists for. Also
    covers the Task 2.2 fix round: extending max_steps (4 -> 8) through the auto_resume
    pre-validation + load path warns (not errors) too.
    """
    from oplm.training.trainer import Trainer

    # Phase 1: train to step 4 and checkpoint.
    cfg1 = tiny_train_cfg(tmp_path, training_parquet, max_steps=4, save_every=4, log_every=1)
    Trainer(cfg1, callbacks=[]).train()

    ckpt = tmp_path / "checkpoint-4"
    saved_state = json.loads((ckpt / "trainer_state.json").read_text())

    # Phase 2: a fresh trainer, resume_from unset, auto_resume=true, targets step 8 -- as a
    # requeued Slurm relaunch of the same output_dir would.
    cfg2 = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=8,
        save_every=8,
        auto_resume=True,
        log_every=1,
    )
    assert cfg2.train.resume_from is None
    with caplog.at_level(logging.WARNING):
        resumed = Trainer(cfg2, callbacks=[])

    # max_steps increased (4 -> 8): allowed, but warned about loudly.
    assert "max_steps" in caplog.text

    # Counters are restored from the discovered checkpoint at construction time -- the same
    # contract an explicit resume_from gives.
    assert resumed.global_step == saved_state["global_step"] == 4
    assert resumed.tokens_seen == saved_state["tokens_seen"]

    resumed.train()
    assert resumed.global_step == 8


def test_auto_resume_with_a_fresh_output_dir_starts_at_step_zero(
    training_parquet: Path, tmp_path: Path
) -> None:
    """``auto_resume=true`` against an empty output_dir is a no-op: training starts at step 0.

    A fresh output_dir holds no committed checkpoint, so the scan must not raise or otherwise
    surprise the caller -- it behaves exactly as if auto_resume were unset.
    """
    from oplm.training.trainer import Trainer

    cfg = tiny_train_cfg(tmp_path, training_parquet, max_steps=2, auto_resume=True, log_every=1)
    trainer = Trainer(cfg, callbacks=[])
    assert trainer.global_step == 0

    trainer.train()
    assert trainer.global_step == 2


# --- data cursor + layout guard (Task 3.3) ----------------------------------------------


def test_checkpoint_records_a_data_cursor(training_parquet: Path, tmp_path: Path) -> None:
    """A saved checkpoint's ``trainer_state.json`` carries a ``cursor`` matching the run layout."""
    from oplm.training.trainer import Trainer

    cfg = tiny_train_cfg(
        tmp_path, training_parquet, max_steps=3, batch_size=_BATCH_SIZE, save_every=3
    )
    Trainer(cfg, callbacks=[]).train()

    state = json.loads((tmp_path / "checkpoint-3" / "trainer_state.json").read_text())
    cursor = state["cursor"]
    assert cursor["epoch"] == 0
    assert cursor["batches_in_epoch"] == 3
    assert cursor["world_size"] == 1
    # cfg.data.num_workers defaults to 0 (no DataLoader worker processes); the cursor
    # records the dataset-striping-*effective* count (1), matching what
    # oplm.data.sequence.dataset's own arithmetic treats "no worker processes" as.
    assert cursor["num_workers"] == 1
    assert cursor["per_rank_batch"] == _BATCH_SIZE
    assert cursor["seed"] == cfg.train.seed


def test_resume_restores_batches_in_epoch_from_cursor(
    training_parquet: Path, tmp_path: Path
) -> None:
    """A layout-compatible resume restores ``_batches_in_epoch`` from the checkpoint's cursor."""
    from oplm.training.trainer import Trainer

    cfg1 = tiny_train_cfg(
        tmp_path, training_parquet, max_steps=3, batch_size=_BATCH_SIZE, save_every=3
    )
    Trainer(cfg1, callbacks=[]).train()

    cfg2 = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=6,
        batch_size=_BATCH_SIZE,
        save_every=6,
        resume_from=str(tmp_path / "checkpoint-3"),
    )
    resumed = Trainer(cfg2, callbacks=[])
    assert resumed._batches_in_epoch == 3

    # The skip must actually reach the top-level dataset (not just the trainer's own
    # counter): ShardedProteinDataset.set_resume_skip stores it as a plain instance
    # attribute (see its docstring), so this is directly inspectable.
    dataset = resumed.dataloader.dataset
    assert dataset._resume_batches_in_epoch == 3
    assert dataset._resume_per_rank_batch == _BATCH_SIZE
    assert dataset._resume_num_workers == 1  # effective count for num_workers=0


def test_resume_skip_is_cleared_on_the_next_epoch_rollover(
    tiny_training_parquet: Path, tmp_path: Path
) -> None:
    """The armed resume skip is disarmed once the resumed epoch actually rolls over.

    The 16-row fixture at batch_size=4 makes one epoch exactly 4 steps. Resuming with
    2 of those 4 batches already consumed and then training past the epoch boundary
    must clear the skip so the *second* epoch is not incorrectly skipped too (Task
    3.3: the StopIteration branch calls ``clear_resume_skip`` on the top-level dataset).
    """
    from oplm.training.trainer import Trainer

    cfg1 = tiny_train_cfg(
        tmp_path, tiny_training_parquet, max_steps=2, batch_size=_BATCH_SIZE, save_every=2
    )
    Trainer(cfg1, callbacks=[]).train()

    cfg2 = tiny_train_cfg(
        tmp_path,
        tiny_training_parquet,
        max_steps=6,  # 2 more steps to finish epoch 0, then 2 steps into epoch 1
        batch_size=_BATCH_SIZE,
        save_every=6,
        resume_from=str(tmp_path / "checkpoint-2"),
    )
    resumed = Trainer(cfg2, callbacks=[])
    dataset = resumed.dataloader.dataset
    assert dataset._resume_batches_in_epoch == 2  # armed at construction

    resumed.train()
    assert resumed.global_step == 6
    assert resumed.epoch == 1  # rolled over once (steps 3-4 finish epoch 0)
    assert dataset._resume_batches_in_epoch is None  # disarmed at the rollover


def test_resume_guard_raises_on_num_workers_mismatch(
    training_parquet: Path, tmp_path: Path
) -> None:
    """A ``num_workers`` change across a resume raises, naming the mismatched field."""
    from oplm.training.trainer import Trainer

    cfg1 = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=3,
        batch_size=_BATCH_SIZE,
        save_every=3,
        num_workers=0,
    )
    Trainer(cfg1, callbacks=[]).train()

    cfg2 = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=6,
        batch_size=_BATCH_SIZE,
        save_every=6,
        num_workers=2,
        resume_from=str(tmp_path / "checkpoint-3"),
    )
    with pytest.raises(ValueError, match="num_workers"):
        Trainer(cfg2, callbacks=[])


def test_resume_guard_mismatch_names_the_escape_hatch(
    training_parquet: Path, tmp_path: Path
) -> None:
    """The mismatch error names ``train.resume_data_position=false`` as the escape hatch."""
    from oplm.training.trainer import Trainer

    cfg1 = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=3,
        batch_size=_BATCH_SIZE,
        save_every=3,
        num_workers=0,
    )
    Trainer(cfg1, callbacks=[]).train()

    cfg2 = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=6,
        batch_size=_BATCH_SIZE,
        save_every=6,
        num_workers=2,
        resume_from=str(tmp_path / "checkpoint-3"),
    )
    with pytest.raises(ValueError, match="resume_data_position=false"):
        Trainer(cfg2, callbacks=[])


def test_resume_guard_bypassed_by_resume_data_position_false(
    training_parquet: Path, tmp_path: Path
) -> None:
    """``resume_data_position=false`` bypasses the guard even with a mismatched layout."""
    from oplm.training.trainer import Trainer

    cfg1 = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=3,
        batch_size=_BATCH_SIZE,
        save_every=3,
        num_workers=0,
    )
    Trainer(cfg1, callbacks=[]).train()

    cfg2 = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=6,
        batch_size=_BATCH_SIZE,
        save_every=6,
        num_workers=2,
        resume_from=str(tmp_path / "checkpoint-3"),
        resume_data_position=False,
    )
    resumed = Trainer(cfg2, callbacks=[])
    # The escape hatch restarts the epoch's data position at row 0 -- no skip armed.
    assert resumed._batches_in_epoch == 0
    resumed.train()
    assert resumed.global_step == 6


def test_resume_data_position_false_logs_reseen_warning(
    training_parquet: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """``resume_data_position=false`` (matching layout) still logs that data will be re-seen."""
    from oplm.training.trainer import Trainer

    cfg1 = tiny_train_cfg(
        tmp_path, training_parquet, max_steps=3, batch_size=_BATCH_SIZE, save_every=3
    )
    Trainer(cfg1, callbacks=[]).train()

    cfg2 = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=6,
        batch_size=_BATCH_SIZE,
        save_every=6,
        resume_from=str(tmp_path / "checkpoint-3"),
        resume_data_position=False,
    )
    with caplog.at_level(logging.WARNING):
        resumed = Trainer(cfg2, callbacks=[])

    assert resumed._batches_in_epoch == 0
    assert "re-seen" in caplog.text


def test_resume_with_no_cursor_logs_reseen_warning_and_restarts_at_zero(
    training_parquet: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A checkpoint with no cursor (e.g. pre-Task-3.3) restarts the epoch, with a warning."""
    from oplm.training.trainer import Trainer

    cfg1 = tiny_train_cfg(
        tmp_path, training_parquet, max_steps=3, batch_size=_BATCH_SIZE, save_every=3
    )
    Trainer(cfg1, callbacks=[]).train()

    # Simulate a pre-cursor checkpoint by stripping the "cursor" key back out.
    state_path = tmp_path / "checkpoint-3" / "trainer_state.json"
    state = json.loads(state_path.read_text())
    del state["cursor"]
    state_path.write_text(json.dumps(state))

    cfg2 = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=6,
        batch_size=_BATCH_SIZE,
        save_every=6,
        resume_from=str(tmp_path / "checkpoint-3"),
    )
    with caplog.at_level(logging.WARNING):
        resumed = Trainer(cfg2, callbacks=[])

    assert resumed._batches_in_epoch == 0
    assert "re-seen" in caplog.text


# --- worker-cycle save alignment deferral (Task 3.3 controller addition) ---------------


def test_periodic_save_deferred_until_worker_cycle_alignment(
    training_parquet: Path, tmp_path: Path
) -> None:
    """With ``num_workers=2``, a save due on an odd ``batches_in_epoch`` lands on the next even one.

    ``save_every=1`` fires the trigger every step; with ``gradient_accumulation_steps=1``,
    ``batches_in_epoch`` tracks ``global_step`` exactly. Steps 1 and 3 land on an odd
    (misaligned) batch count and must be deferred one step; steps 2 and 4 are already
    aligned, so the actually-committed checkpoints land at [2, 4], not [1, 2, 3, 4].
    """
    from oplm.training.trainer import Trainer

    cfg = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=4,
        batch_size=_BATCH_SIZE,
        save_every=1,
        save_total_limit=10,
        num_workers=2,
    )
    callback = FullRecordingCallback()
    Trainer(cfg, callbacks=[callback]).train()

    assert callback.checkpoint_steps == [2, 4]
