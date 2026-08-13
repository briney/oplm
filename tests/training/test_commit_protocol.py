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
import logging
from typing import TYPE_CHECKING

import pytest
import torch

from oplm.config import DataConfig, OplmConfig, TrainConfig
from oplm.model import OplmConfig as OplmModelConfig
from oplm.model import OplmForMaskedLM
from oplm.training.checkpoint import (
    _rotate_checkpoints,
    clean_stale_checkpoint_dirs,
    latest_checkpoint,
    list_committed_checkpoints,
    load_checkpoint,
    mark_permanent,
    nth_latest_checkpoint,
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


def _prepared_model(
    cfg: OplmConfig,
) -> tuple[Accelerator, OplmForMaskedLM, torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """Build and prepare a tiny model + optimizer + scheduler on CPU for DCP save/load."""
    from accelerate import Accelerator

    accelerator = Accelerator(cpu=True, mixed_precision="no")
    model = OplmForMaskedLM(cfg.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    model, optimizer = accelerator.prepare(model, optimizer)
    return accelerator, model, optimizer, scheduler


@pytest.mark.slow
def test_save_checkpoint_commits_via_rename(tmp_path: Path) -> None:
    """After a save: the committed dir exists, no ``.tmp`` dir remains, ``latest`` points at it."""
    cfg = _cfg()
    accelerator, model, optimizer, scheduler = _prepared_model(cfg)

    save_checkpoint(
        accelerator=accelerator,
        model=model,
        optimizers=[optimizer],
        schedulers=[scheduler],
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
    accelerator, model, optimizer, scheduler = _prepared_model(cfg)

    for tokens in (400, 999):
        save_checkpoint(
            accelerator=accelerator,
            model=model,
            optimizers=[optimizer],
            schedulers=[scheduler],
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


# --- Task 2.2: list_committed_checkpoints / nth_latest_checkpoint --------------------


def test_list_committed_checkpoints_orders_newest_first(tmp_path: Path) -> None:
    """Committed dirs come back newest-step-first; ``.tmp``/``.old`` are excluded."""
    for step in (100, 300, 200):
        (tmp_path / f"checkpoint-{step}").mkdir()
    (tmp_path / "checkpoint-9000.tmp").mkdir()
    (tmp_path / "checkpoint-8000.old").mkdir()

    result = list_committed_checkpoints(tmp_path)

    assert [p.name for p in result] == ["checkpoint-300", "checkpoint-200", "checkpoint-100"]


def test_list_committed_checkpoints_empty_or_missing_dir(tmp_path: Path) -> None:
    """No committed checkpoints (fresh or nonexistent output_dir) yields an empty list."""
    assert list_committed_checkpoints(tmp_path) == []
    assert list_committed_checkpoints(tmp_path / "does-not-exist") == []


def test_nth_latest_checkpoint_walks_backward_from_newest(tmp_path: Path) -> None:
    """``n=0`` is the newest, ``n=1`` the next-newest, etc.; out of range is ``None``."""
    for step in (100, 200, 300):
        (tmp_path / f"checkpoint-{step}").mkdir()

    assert nth_latest_checkpoint(tmp_path, 0) == tmp_path / "checkpoint-300"
    assert nth_latest_checkpoint(tmp_path, 1) == tmp_path / "checkpoint-200"
    assert nth_latest_checkpoint(tmp_path, 2) == tmp_path / "checkpoint-100"
    assert nth_latest_checkpoint(tmp_path, 3) is None


# --- Task 2.2: auto_resume fallback to the previous committed checkpoint -------------


@pytest.mark.slow
def test_auto_resume_falls_back_to_previous_checkpoint_on_torn_metadata(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A torn ``.metadata`` on the newest checkpoint falls back to the previous one.

    Mirrors the requeue scenario a corrupted-shard kill would produce: the newest
    checkpoint's ``.metadata`` is truncated post-commit (simulating a kill mid-``dcp.
    save`` that ``save_checkpoint``'s own commit-rename protocol does not protect
    against -- see ``validate_checkpoint_for_resume``'s docstring). ``auto_resume``
    must resolve to the next-newest *committed* checkpoint instead, with a loud warning.
    """
    from oplm.training.trainer import _resolve_resume_target

    cfg = _cfg()
    accelerator, model, optimizer, scheduler = _prepared_model(cfg)

    for step in (4, 8):
        save_checkpoint(
            accelerator=accelerator,
            model=model,
            optimizers=[optimizer],
            schedulers=[scheduler],
            cfg=cfg,
            output_dir=str(tmp_path),
            global_step=step,
            epoch=1,
            samples_seen=step * 4,
            tokens_seen=step * 40,
        )

    (tmp_path / "checkpoint-8" / ".metadata").write_bytes(b"not a valid dcp metadata blob")

    with caplog.at_level(logging.WARNING):
        resolved = _resolve_resume_target(
            accelerator,
            resume_from=None,
            auto_resume=True,
            output_dir=str(tmp_path),
            cfg=cfg,
        )

    assert resolved == str(tmp_path / "checkpoint-4")
    assert "checkpoint-8" in caplog.text
    assert "falling back" in caplog.text.lower()


@pytest.mark.slow
def test_auto_resume_falls_back_on_schedule_incompatible_newest_checkpoint(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A schedule-incompatible newest checkpoint also triggers the fallback path.

    ``validate_checkpoint_for_resume`` runs the same schedule-compat check
    ``load_checkpoint`` does. Checkpoint-4 is saved with the schedule the live config
    still matches; checkpoint-8 is saved with a different ``warmup_steps`` (as if a
    separate, differently-configured run had written into the same ``output_dir`` --
    contrived, but it exercises the same "newest candidate fails validation" branch a
    real config-vs-checkpoint drift would). The newest is rejected before the
    broadcast, exactly like torn DCP metadata is, and checkpoint-4 is selected instead.
    """
    from oplm.training.trainer import _resolve_resume_target

    live_cfg = _cfg()
    accelerator, model, optimizer, scheduler = _prepared_model(live_cfg)

    save_checkpoint(
        accelerator=accelerator,
        model=model,
        optimizers=[optimizer],
        schedulers=[scheduler],
        cfg=live_cfg,
        output_dir=str(tmp_path),
        global_step=4,
        epoch=1,
        samples_seen=16,
        tokens_seen=160,
    )

    incompatible_cfg = _cfg()
    incompatible_cfg.train.warmup_steps = live_cfg.train.warmup_steps + 500
    save_checkpoint(
        accelerator=accelerator,
        model=model,
        optimizers=[optimizer],
        schedulers=[scheduler],
        cfg=incompatible_cfg,
        output_dir=str(tmp_path),
        global_step=8,
        epoch=1,
        samples_seen=32,
        tokens_seen=320,
    )

    with caplog.at_level(logging.WARNING):
        resolved = _resolve_resume_target(
            accelerator,
            resume_from=None,
            auto_resume=True,
            output_dir=str(tmp_path),
            cfg=live_cfg,
        )

    assert resolved == str(tmp_path / "checkpoint-4")
    assert "warmup_steps" in caplog.text


@pytest.mark.slow
def test_auto_resume_raises_when_every_candidate_fails_validation(tmp_path: Path) -> None:
    """All candidates corrupted (newest + every fallback) raises, rank-identically."""
    from oplm.training.trainer import _resolve_resume_target

    cfg = _cfg()
    accelerator, model, optimizer, scheduler = _prepared_model(cfg)

    for step in (4, 8):
        save_checkpoint(
            accelerator=accelerator,
            model=model,
            optimizers=[optimizer],
            schedulers=[scheduler],
            cfg=cfg,
            output_dir=str(tmp_path),
            global_step=step,
            epoch=1,
            samples_seen=step * 4,
            tokens_seen=step * 40,
        )
        (tmp_path / f"checkpoint-{step}" / ".metadata").write_bytes(b"garbage")

    with pytest.raises(RuntimeError, match="auto_resume"):
        _resolve_resume_target(
            accelerator,
            resume_from=None,
            auto_resume=True,
            output_dir=str(tmp_path),
            cfg=cfg,
        )


@pytest.mark.slow
def test_explicit_resume_from_never_falls_back_even_when_corrupted(tmp_path: Path) -> None:
    """An explicit ``resume_from`` is returned unchanged, even if it is corrupted.

    Falling back is an ``auto_resume``-only behavior; an operator-pinned ``resume_from``
    is never second-guessed at resolve time -- if it's bad, the failure surfaces later,
    loudly, at ``dcp.load`` (see the next test), not silently as a swap to a different
    checkpoint the operator didn't ask for.
    """
    from oplm.training.trainer import _resolve_resume_target

    cfg = _cfg()
    accelerator, model, optimizer, scheduler = _prepared_model(cfg)

    for step in (4, 8):
        save_checkpoint(
            accelerator=accelerator,
            model=model,
            optimizers=[optimizer],
            schedulers=[scheduler],
            cfg=cfg,
            output_dir=str(tmp_path),
            global_step=step,
            epoch=1,
            samples_seen=step * 4,
            tokens_seen=step * 40,
        )

    # Corrupt the *older* checkpoint and pin resume_from to it explicitly. Note
    # auto_resume=True here too -- resume_from must still win with zero validation.
    (tmp_path / "checkpoint-4" / ".metadata").write_bytes(b"garbage")

    resolved = _resolve_resume_target(
        accelerator,
        resume_from=str(tmp_path / "checkpoint-4"),
        auto_resume=True,
        output_dir=str(tmp_path),
        cfg=cfg,
    )

    assert resolved == str(tmp_path / "checkpoint-4")


@pytest.mark.slow
def test_explicit_resume_from_corrupted_checkpoint_raises_at_load(tmp_path: Path) -> None:
    """A corrupted checkpoint loaded via explicit ``resume_from`` raises at load time."""
    cfg = _cfg()
    accelerator, model, optimizer, scheduler = _prepared_model(cfg)

    save_checkpoint(
        accelerator=accelerator,
        model=model,
        optimizers=[optimizer],
        schedulers=[scheduler],
        cfg=cfg,
        output_dir=str(tmp_path),
        global_step=4,
        epoch=1,
        samples_seen=16,
        tokens_seen=160,
    )
    committed = tmp_path / "checkpoint-4"
    (committed / ".metadata").write_bytes(b"garbage")

    fresh_accelerator, fresh_model, fresh_optimizer, fresh_scheduler = _prepared_model(cfg)
    # dcp.load's collective plan exchange wraps a corrupted-metadata failure in
    # torch.distributed.checkpoint.api.CheckpointException, which subclasses
    # BaseException directly (not Exception) -- broader than a bare `Exception` catch.
    with pytest.raises(BaseException):  # noqa: B017 -- DCP's error type is an internal detail
        load_checkpoint(
            fresh_accelerator,
            fresh_model,
            [fresh_optimizer],
            [fresh_scheduler],
            str(committed),
            cfg,
        )


# --- review fix: symmetric world-size-increase rejection (RNG sidecar count) ---------


@pytest.mark.slow
def test_validate_rejects_resume_at_a_larger_world_size_than_the_rng_sidecars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ws-increase resume fails validation on the MAIN process, symmetrically.

    Important review finding: each rank's RNG state lives in its own
    ``rng_state_<rank>.pt`` sidecar, so resuming a checkpoint saved at world size N with
    a live world size > N leaves the extra ranks with no sidecar -- and
    ``_restore_rng_sidecar`` raises there, on *those ranks only*, deep inside
    ``load_checkpoint``. That asymmetric raise is exactly the desynchronized-exception
    shape the pre-broadcast validation discipline exists to avoid. The cheap sidecar
    *count* check now runs in ``validate_checkpoint_for_resume`` (main process, before
    the broadcast), so every rank fails identically with one message naming both counts
    and the ``OPLM_ALLOW_MISSING_RNG`` escape hatch.

    A world-size-2 checkpoint is simulated by copying rank 0's real sidecar to rank 1's
    name (the save here is genuinely single-process), then validated against a fake
    accelerator reporting ``num_processes=4``.
    """
    import shutil

    from oplm.training.checkpoint import validate_checkpoint_for_resume

    cfg = _cfg()
    accelerator, model, optimizer, scheduler = _prepared_model(cfg)
    save_checkpoint(
        accelerator=accelerator,
        model=model,
        optimizers=[optimizer],
        schedulers=[scheduler],
        cfg=cfg,
        output_dir=str(tmp_path),
        global_step=4,
        epoch=1,
        samples_seen=16,
        tokens_seen=160,
    )
    committed = tmp_path / "checkpoint-4"
    shutil.copy(committed / "rng_state_0.pt", committed / "rng_state_1.pt")

    class _FakeAccelerator:
        num_processes = 4

    live = _FakeAccelerator()

    monkeypatch.delenv("OPLM_ALLOW_MISSING_RNG", raising=False)
    with pytest.raises(ValueError) as exc_info:
        validate_checkpoint_for_resume(committed, cfg, world_size=live.num_processes)

    message = str(exc_info.value)
    assert "2" in message and "4" in message  # both counts are named
    assert "OPLM_ALLOW_MISSING_RNG" in message  # ... and so is the escape hatch

    # The escape hatch turns it back into a supported (RNG-resetting) resume.
    monkeypatch.setenv("OPLM_ALLOW_MISSING_RNG", "1")
    validate_checkpoint_for_resume(committed, cfg, world_size=live.num_processes)

    # A same-or-smaller live world size is always fine, env var or not.
    monkeypatch.delenv("OPLM_ALLOW_MISSING_RNG", raising=False)
    validate_checkpoint_for_resume(committed, cfg, world_size=2)
    validate_checkpoint_for_resume(committed, cfg, world_size=1)
    validate_checkpoint_for_resume(committed, cfg)  # unknown world size -> not checked


# --- Task 2.2 fix round: auto_map regression (serialize_config / load_config) --------


@pytest.mark.slow
def test_second_checkpoints_config_yaml_round_trips_through_load_config(tmp_path: Path) -> None:
    """Regression: a checkpoint saved after the first one must still ``load_config``.

    ``OplmConfig.register_for_auto_class`` (called once at ``oplm`` package import, see
    ``oplm/__init__.py``) plus ``PreTrainedModel.save_pretrained`` stamp ``auto_map`` onto
    the shared, mutable ``cfg.model`` instance *in place* the first time
    ``save_pretrained`` runs against it -- this checkpoint's own ``hf/`` export, written a
    few lines after ``config.yaml`` inside ``save_checkpoint``. Every checkpoint saved by
    the same process *after* the first one therefore has its ``config.yaml`` written
    *after* that mutation already landed. If ``serialize_config`` ever again stopped
    stripping ``auto_map`` (see its ``model_dict.pop("auto_map", None)`` line), this test
    fails with ``ValueError: Unknown model config key(s): ['auto_map']`` when
    ``load_config`` tries to reload the second checkpoint's ``config.yaml`` (raised by
    ``oplm.config._reject_unknown_model_keys``, since a freshly constructed default
    ``OplmModelConfig()`` instance's own ``to_dict()`` never has ``auto_map`` either --
    it is only ever added by ``save_pretrained``, never by construction).
    """
    from oplm.config import load_config

    cfg = _cfg()
    accelerator, model, optimizer, scheduler = _prepared_model(cfg)

    for step in (4, 8):
        save_checkpoint(
            accelerator=accelerator,
            model=model,
            optimizers=[optimizer],
            schedulers=[scheduler],
            cfg=cfg,
            output_dir=str(tmp_path),
            global_step=step,
            epoch=1,
            samples_seen=step * 4,
            tokens_seen=step * 40,
        )

    # The FIRST checkpoint's config.yaml is written before any save_pretrained call has
    # ever run against cfg.model, so it never had auto_map to strip in the first place --
    # it would pass even with the fix reverted. The bug (and the fix) only shows up on
    # the SECOND checkpoint onward, which is what this test targets.
    second_config_path = tmp_path / "checkpoint-8" / "config.yaml"
    assert "auto_map" not in second_config_path.read_text()

    reloaded = load_config(["--config", str(second_config_path)])
    assert reloaded.model.hidden_size == cfg.model.hidden_size
    assert reloaded.model.num_hidden_layers == cfg.model.num_hidden_layers
