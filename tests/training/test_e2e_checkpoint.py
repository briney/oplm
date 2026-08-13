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
import math
import time as time_module
from typing import TYPE_CHECKING

import pytest

from oplm.training.optim import get_schedule_fn
from tests.training.conftest import FullRecordingCallback, tiny_train_cfg

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.slow

_BATCH_SIZE = 4


class _FakeClock:
    """A fake monotonic/wall clock that advances by a fixed step on every call.

    Installed via ``monkeypatch.setattr(time_module, "monotonic", ...)`` (or
    ``"time"``) so it stands in for the exact stdlib attribute the trainer calls —
    patching the module attribute (not a local ``from time import monotonic``
    binding) is what makes the fake take effect inside ``trainer.py``.
    """

    def __init__(self, step: float = 30.0, start: float = 0.0) -> None:
        self._value = start
        self._step = step

    def __call__(self) -> float:
        value = self._value
        self._value += self._step
        return value


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

    monkeypatch.setattr(time_module, "monotonic", _FakeClock(step=30.0))

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

    monkeypatch.setattr(time_module, "time", _FakeClock(step=2000.0))

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

    monkeypatch.setattr(time_module, "time", _FakeClock(step=100.0))

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


def test_resume_restores_state_and_continues(training_parquet: Path, tmp_path: Path) -> None:
    """Resuming restores counters + LR position, reaches the new target, stays finite."""
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
    resumed = Trainer(cfg2, callbacks=[cb2])

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
    training_parquet: Path, tmp_path: Path
) -> None:
    """``auto_resume=true`` with no explicit ``resume_from`` finds and resumes the checkpoint.

    Mirrors ``test_resume_restores_state_and_continues`` but exercises the scanning path
    (``Trainer`` discovers ``checkpoint-4`` under ``output_dir`` itself) instead of an
    operator-pinned ``resume_from`` -- the requeue scenario auto_resume exists for.
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
    resumed = Trainer(cfg2, callbacks=[])

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
