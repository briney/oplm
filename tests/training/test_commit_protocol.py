"""Atomic checkpoint commit protocol + shared committed-only discovery.

``save_checkpoint`` writes into a ``checkpoint-<step>.tmp/`` staging directory and only
becomes visible to discovery after a barrier-guarded rename to ``checkpoint-<step>/`` on the
main process. This means a kill mid-save can never leave a resumable-looking checkpoint
behind: discovery (``latest_checkpoint``) and rotation (``_rotate_checkpoints``) both ignore
``*.tmp`` directories entirely.

Tests that exercise the real ``save_checkpoint`` path spin up a CPU ``Accelerator`` and are
marked slow; the pure-filesystem tests (torn ``.tmp`` visibility, rotation, and the shared
sweep/training identity) do not need torch and run fast.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from oplm.config import DataConfig, OplmConfig, TrainConfig
from oplm.model import OplmConfig as OplmModelConfig
from oplm.model import OplmForMaskedLM
from oplm.training.checkpoint import (
    _rotate_checkpoints,
    clean_stale_checkpoint_dirs,
    latest_checkpoint,
    mark_permanent,
    save_checkpoint,
)

if TYPE_CHECKING:
    from pathlib import Path

    from accelerate import Accelerator


def _cfg() -> OplmConfig:
    """Tiny root config whose ``model`` is the HF ``OplmConfig``."""
    return OplmConfig(
        model=OplmModelConfig(
            hidden_size=32,
            num_attention_heads=4,
            num_hidden_layers=2,
            max_position_embeddings=64,
        ),
        train=TrainConfig(wandb_enabled=False, mixed_precision="no"),
        data=DataConfig(num_workers=0, pin_memory=False),
    )


def _prepared_model(cfg: OplmConfig) -> tuple[Accelerator, OplmForMaskedLM]:
    """Build and prepare a tiny model on CPU so ``save_state`` has something to save."""
    from accelerate import Accelerator

    accelerator = Accelerator(cpu=True, mixed_precision="no")
    model = OplmForMaskedLM(cfg.model)
    model = accelerator.prepare(model)
    return accelerator, model


@pytest.mark.slow
def test_save_checkpoint_commits_via_rename(tmp_path: Path) -> None:
    """After a save: the committed dir exists, no ``.tmp`` dir remains, ``latest`` points at it."""
    cfg = _cfg()
    accelerator, model = _prepared_model(cfg)

    save_checkpoint(
        accelerator=accelerator,
        model=model,
        cfg=cfg,
        output_dir=str(tmp_path),
        global_step=10,
        epoch=1,
        samples_seen=40,
        tokens_seen=400,
    )

    committed = tmp_path / "checkpoint-10"
    assert committed.is_dir()
    assert (committed / "trainer_state.json").exists()

    tmp_dirs = list(tmp_path.glob("checkpoint-*.tmp"))
    assert tmp_dirs == []

    latest_pointer = tmp_path / "latest"
    assert latest_pointer.exists()
    assert latest_pointer.read_text().strip() == "checkpoint-10"


@pytest.mark.slow
def test_save_checkpoint_replaces_tmp_at_same_step_on_resave(tmp_path: Path) -> None:
    """Re-saving at the same step (e.g. after a requeue) replaces the committed dir cleanly."""
    cfg = _cfg()
    accelerator, model = _prepared_model(cfg)

    for tokens in (400, 999):
        save_checkpoint(
            accelerator=accelerator,
            model=model,
            cfg=cfg,
            output_dir=str(tmp_path),
            global_step=10,
            epoch=1,
            samples_seen=40,
            tokens_seen=tokens,
        )

    committed = tmp_path / "checkpoint-10"
    assert committed.is_dir()
    assert list(tmp_path.glob("checkpoint-*.tmp")) == []
    state = json.loads((committed / "trainer_state.json").read_text())
    assert state["tokens_seen"] == 999


def test_torn_tmp_dir_is_invisible(tmp_path: Path) -> None:
    """A torn ``.tmp`` dir from a killed-mid-save process never wins discovery."""
    (tmp_path / "checkpoint-500.tmp").mkdir()
    (tmp_path / "checkpoint-300").mkdir()
    assert latest_checkpoint(tmp_path) == tmp_path / "checkpoint-300"


def test_torn_tmp_dir_is_invisible_even_when_alone(tmp_path: Path) -> None:
    """A ``.tmp`` dir with no committed sibling yields no resume candidate at all."""
    (tmp_path / "checkpoint-500.tmp").mkdir()
    assert latest_checkpoint(tmp_path) is None


def test_rotation_ignores_tmp_dirs(tmp_path: Path) -> None:
    """A ``.tmp`` dir neither counts toward the rotation limit nor gets deleted."""
    (tmp_path / "checkpoint-100").mkdir()
    (tmp_path / "checkpoint-200").mkdir()
    (tmp_path / "checkpoint-300.tmp").mkdir()

    _rotate_checkpoints(tmp_path, save_total_limit=1)

    # Only the two committed dirs count toward the limit; the newest (200) survives,
    # 100 is rotated away, and the in-flight .tmp dir is left completely untouched.
    assert (tmp_path / "checkpoint-100").exists() is False
    assert (tmp_path / "checkpoint-200").exists()
    assert (tmp_path / "checkpoint-300.tmp").exists()


def test_old_dir_with_committed_final_is_removed_as_stale(tmp_path: Path) -> None:
    """Crash state (a): a kill *after* the new commit landed leaves both dirs present.

    The replace that produced ``checkpoint-10.old`` already finished (``checkpoint-10``
    exists), so the ``.old`` dir is stale and simply removed; the final commit is kept as-is.
    """
    final_dir = tmp_path / "checkpoint-10"
    final_dir.mkdir()
    (final_dir / "marker").write_text("new")
    aside_dir = tmp_path / "checkpoint-10.old"
    aside_dir.mkdir()
    (aside_dir / "marker").write_text("old")

    clean_stale_checkpoint_dirs(tmp_path)

    assert final_dir.is_dir()
    assert (final_dir / "marker").read_text() == "new"
    assert not aside_dir.exists()


def test_old_dir_without_committed_final_is_recovered(tmp_path: Path) -> None:
    """Crash state (b): a kill *between* the two renames leaves only the aside dir.

    ``checkpoint-10`` does not exist (the ``.tmp`` -> final rename never happened), so the
    aside dir — the only surviving checkpoint at this step — is recovered onto the final name.
    """
    aside_dir = tmp_path / "checkpoint-10.old"
    aside_dir.mkdir()
    (aside_dir / "marker").write_text("old")

    clean_stale_checkpoint_dirs(tmp_path)

    final_dir = tmp_path / "checkpoint-10"
    assert final_dir.is_dir()
    assert (final_dir / "marker").read_text() == "old"
    assert not aside_dir.exists()


def test_old_dir_invisible_to_latest_checkpoint(tmp_path: Path) -> None:
    """Crash state (c): an orphaned ``.old`` dir never wins discovery, even if newer-numbered."""
    (tmp_path / "checkpoint-9000").mkdir()
    (tmp_path / "checkpoint-10000.old").mkdir()
    assert latest_checkpoint(tmp_path) == tmp_path / "checkpoint-9000"


def test_old_dir_invisible_to_rotation(tmp_path: Path) -> None:
    """Crash state (c): an orphaned ``.old`` dir neither counts toward the limit nor is deleted."""
    (tmp_path / "checkpoint-100").mkdir()
    (tmp_path / "checkpoint-200").mkdir()
    (tmp_path / "checkpoint-100.old").mkdir()

    _rotate_checkpoints(tmp_path, save_total_limit=1)

    assert not (tmp_path / "checkpoint-100").exists()
    assert (tmp_path / "checkpoint-200").exists()
    assert (tmp_path / "checkpoint-100.old").exists()


# --- Permanent-checkpoint retention exemptions (Task 1.2) ----------------------


def test_rotation_exempts_keep_every_n_steps_checkpoints(tmp_path: Path) -> None:
    """With keep_every_n_steps=100, checkpoints on that boundary are never rotated.

    limit=1 over {100, 150, 200, 250}: 100 and 200 are permanent (step % 100 == 0);
    250 is the newest rolling checkpoint, so only 150 is deleted.
    """
    for step in (100, 150, 200, 250):
        (tmp_path / f"checkpoint-{step}").mkdir()

    _rotate_checkpoints(tmp_path, save_total_limit=1, keep_every_n_steps=100)

    assert (tmp_path / "checkpoint-100").exists()
    assert not (tmp_path / "checkpoint-150").exists()
    assert (tmp_path / "checkpoint-200").exists()
    assert (tmp_path / "checkpoint-250").exists()


def test_rotation_exempts_keep_marker_dirs_regardless_of_step(tmp_path: Path) -> None:
    """A dir with a ``KEEP`` marker survives rotation even off the step boundary."""
    for step in (100, 150, 200, 250):
        (tmp_path / f"checkpoint-{step}").mkdir()
    mark_permanent(tmp_path / "checkpoint-150")

    _rotate_checkpoints(tmp_path, save_total_limit=1)

    # 150 is KEEP-marked and survives despite not being the newest; 100 and 200 are
    # ordinary rolling checkpoints culled down to the newest (250).
    assert not (tmp_path / "checkpoint-100").exists()
    assert (tmp_path / "checkpoint-150").exists()
    assert not (tmp_path / "checkpoint-200").exists()
    assert (tmp_path / "checkpoint-250").exists()


def test_mark_permanent_writes_keep_marker(tmp_path: Path) -> None:
    """mark_permanent writes a ``KEEP`` marker file inside the checkpoint dir."""
    checkpoint_dir = tmp_path / "checkpoint-100"
    checkpoint_dir.mkdir()

    mark_permanent(checkpoint_dir)

    assert (checkpoint_dir / "KEEP").exists()


def test_rotation_without_keep_every_n_steps_behaves_as_before(tmp_path: Path) -> None:
    """Omitting keep_every_n_steps preserves the original unconditional rotation behavior."""
    for step in (100, 150, 200, 250):
        (tmp_path / f"checkpoint-{step}").mkdir()

    _rotate_checkpoints(tmp_path, save_total_limit=1)

    assert not (tmp_path / "checkpoint-100").exists()
    assert not (tmp_path / "checkpoint-150").exists()
    assert not (tmp_path / "checkpoint-200").exists()
    assert (tmp_path / "checkpoint-250").exists()


def test_sweep_run_uses_committed_only() -> None:
    """``sweep.run.latest_checkpoint`` is the shared function, not a local copy."""
    from oplm.sweep import run
    from oplm.training import checkpoint

    assert run.latest_checkpoint is checkpoint.latest_checkpoint


def test_latest_checkpoint_picks_numeric_max_ignoring_tmp(tmp_path: Path) -> None:
    """Numeric ordering still wins over a higher-numbered but uncommitted ``.tmp`` dir."""
    (tmp_path / "checkpoint-9000").mkdir()
    (tmp_path / "checkpoint-10000.tmp").mkdir()
    assert latest_checkpoint(tmp_path) == tmp_path / "checkpoint-9000"


@pytest.mark.slow
def test_stale_tmp_cleanup_at_trainer_start(tmp_path: Path, training_parquet: Path) -> None:
    """A torn ``.tmp`` dir left over from a killed run is deleted before training resumes."""
    from tests.training.conftest import tiny_train_cfg

    (tmp_path / "checkpoint-3.tmp").mkdir(parents=True)
    (tmp_path / "checkpoint-3.tmp" / "marker").write_text("torn")

    cfg = tiny_train_cfg(tmp_path, training_parquet, max_steps=1, batch_size=4)

    from oplm.training.trainer import Trainer

    Trainer(cfg).train()

    assert not (tmp_path / "checkpoint-3.tmp").exists()
