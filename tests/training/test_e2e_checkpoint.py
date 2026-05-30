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
from typing import TYPE_CHECKING

import pytest

from oplm.training.optim import get_schedule_fn
from tests.training.conftest import FullRecordingCallback, tiny_train_cfg

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.slow

_BATCH_SIZE = 4


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

    # Saves fire on the cadence (2, 4, 6) plus the unconditional final save at 6.
    assert callback.checkpoint_steps == [2, 4, 6, 6]

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
