"""G13 — wandb tracker path (docs/TESTING_E2E.md §5, optional).

Runs a tiny trainer with ``wandb_enabled=True`` under ``WANDB_MODE=offline`` so the
real ``init_trackers`` + ``_config_to_flat_dict`` + ``accelerator.log`` path runs
without a network or login. Config flattening is a genuine serialization risk
(HF config + dataclasses -> one flat dict), so it is also asserted directly.

Also covers Task 1.7 (W&B run continuity): the persisted ``wandb_run_id`` file and
``trainer_state.json`` field, and that a resumed run's ``wandb.init`` receives
``id=<first run's id>, resume="allow"`` so a requeue continues the same W&B run
instead of starting a new one.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from tests.training.conftest import FullRecordingCallback, tiny_train_cfg

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.slow


def test_config_to_flat_dict_is_flat_and_namespaced(training_parquet: Path, tmp_path: Path) -> None:
    """``_config_to_flat_dict`` flattens model/train/data into one namespaced dict."""
    from oplm.training.trainer import _config_to_flat_dict

    cfg = tiny_train_cfg(tmp_path, training_parquet)
    flat = _config_to_flat_dict(cfg)

    assert isinstance(flat, dict)
    assert all("/" in key for key in flat)
    for prefix in ("model/", "train/", "data/"):
        assert any(key.startswith(prefix) for key in flat), f"no {prefix} keys"
    # Representative fields survive the flattening.
    assert flat["train/max_steps"] == cfg.train.max_steps
    assert flat["model/hidden_size"] == cfg.model.hidden_size


def test_wandb_offline_run_completes(
    training_parquet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wandb-enabled run completes end to end in offline mode (init_trackers + log)."""
    pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))

    from oplm.training.trainer import Trainer

    cfg = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=3,
        wandb_enabled=True,
        wandb_run_name="e2e-offline",
    )
    callback = FullRecordingCallback()
    Trainer(cfg, callbacks=[callback]).train()

    assert callback.train_start_count == 1
    assert callback.train_end_count == 1
    assert callback.train_log_steps  # metrics were logged through the wandb tracker

    # wandb logs are consolidated under output_dir (dir=output_dir), not ./wandb.
    assert (tmp_path / "wandb").is_dir()

    # Task 1.7: the run id is persisted right after init_trackers, both as a plain
    # marker file and inside the checkpoint's trainer_state.json.
    run_id_path = tmp_path / "wandb_run_id"
    assert run_id_path.is_file()
    run_id = run_id_path.read_text().strip()
    assert run_id


def test_resume_continues_the_same_wandb_run(
    training_parquet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A requeue (auto_resume) reuses the first run's wandb id instead of starting a new run.

    Phase 1 trains to step 4 and checkpoints under ``WANDB_MODE=offline``. Phase 2 is a
    fresh ``Trainer`` with ``auto_resume=True`` (no explicit ``resume_from``) targeting the
    same ``output_dir`` -- the exact requeue scenario Task 1.5 exists for. It must read the
    id persisted by phase 1 and pass ``id=<that id>, resume="allow"`` into ``wandb.init`` (via
    accelerate's ``WandBTracker``) so W&B stitches both runs into one continuous timeline, and
    it must re-persist the *same* id (not a new one) to both the marker file and the new
    checkpoint's ``trainer_state.json``.
    """
    pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))

    from accelerate.tracking import WandBTracker

    from oplm.training.trainer import Trainer

    captured_init_kwargs: list[dict[str, object]] = []
    original_init = WandBTracker.__init__

    def _capturing_init(self: WandBTracker, run_name: str, **kwargs: object) -> None:
        captured_init_kwargs.append(dict(kwargs))
        original_init(self, run_name, **kwargs)

    monkeypatch.setattr(WandBTracker, "__init__", _capturing_init)

    # Phase 1: train to step 4 and checkpoint; a fresh run, no id to resume.
    cfg1 = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=4,
        save_every=4,
        wandb_enabled=True,
        wandb_run_name="e2e-continuity",
    )
    Trainer(cfg1, callbacks=[]).train()

    run_id_path = tmp_path / "wandb_run_id"
    first_run_id = run_id_path.read_text().strip()
    assert first_run_id

    first_state = json.loads((tmp_path / "checkpoint-4" / "trainer_state.json").read_text())
    assert first_state["wandb_run_id"] == first_run_id

    assert len(captured_init_kwargs) == 1
    assert "id" not in captured_init_kwargs[0]
    assert captured_init_kwargs[0].get("resume") != "allow"

    # Phase 2: fresh trainer, resume_from unset, auto_resume=true -- discovers checkpoint-4
    # and must resume the same wandb run.
    cfg2 = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=8,
        save_every=8,
        auto_resume=True,
        wandb_enabled=True,
        wandb_run_name="e2e-continuity",
    )
    assert cfg2.train.resume_from is None
    Trainer(cfg2, callbacks=[]).train()

    second_run_id = run_id_path.read_text().strip()
    assert second_run_id == first_run_id

    second_state = json.loads((tmp_path / "checkpoint-8" / "trainer_state.json").read_text())
    assert second_state["wandb_run_id"] == first_run_id

    assert len(captured_init_kwargs) == 2
    assert captured_init_kwargs[1]["id"] == first_run_id
    assert captured_init_kwargs[1]["resume"] == "allow"


def test_wandb_run_none_after_init_trackers_is_skipped_gracefully(
    training_parquet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If ``wandb.run`` stays ``None`` after ``init_trackers`` (an offline-mode edge case),
    persisting the run id is skipped gracefully instead of raising.

    Forces the edge case by letting the real (offline) ``WandBTracker.start`` -> ``wandb.init``
    run normally (so ``accelerate``'s subsequent ``store_init_configuration`` -> ``wandb.config.
    update`` still has a live run to write into), then immediately clearing the module-global
    ``wandb.run`` pointer -- simulating the trainer observing ``wandb.run is None`` right after
    ``init_trackers`` returns, without tearing down the actual (offline) run underneath it.
    """
    pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))

    import wandb
    from accelerate.tracking import WandBTracker

    from oplm.training.trainer import Trainer

    original_start = WandBTracker.start

    def _start_then_clear_run(self: WandBTracker) -> None:
        original_start(self)
        wandb.run = None

    monkeypatch.setattr(WandBTracker, "start", _start_then_clear_run)

    cfg = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=2,
        wandb_enabled=True,
        wandb_run_name="e2e-no-active-run",
    )
    trainer = Trainer(cfg, callbacks=[])

    assert trainer._wandb_run_id is None
    assert not (tmp_path / "wandb_run_id").exists()
