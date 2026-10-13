"""G13 — wandb tracker path (docs/TESTING_E2E.md §5, optional).

Runs a tiny trainer with ``wandb_enabled=True`` under ``WANDB_MODE=offline`` so the
real ``init_trackers`` + ``_config_to_flat_dict`` + ``accelerator.log`` path runs
without a network or login. Config flattening is a genuine serialization risk
(HF config + dataclasses -> one flat dict), so it is also asserted directly.

Also covers Task 1.7 (W&B run continuity): the persisted ``wandb_run_id`` file and
``trainer_state.json`` field, and that a resumed run's ``wandb.init`` receives
``id=<first run's id>, resume="allow"`` so a requeue continues the same W&B run
instead of starting a new one; and the step-aligned history: every row is logged at
``step=global_step``, a resume behind the server's history rewinds it to the checkpoint
(``resume_from``), and a refused rewind falls back to the plain resume.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

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


def _offline_wandb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))


def _capture_wandb_init_kwargs(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    from accelerate.tracking import WandBTracker

    captured: list[dict[str, object]] = []
    original_init = WandBTracker.__init__

    def _capturing_init(self: WandBTracker, run_name: str, **kwargs: object) -> None:
        captured.append(dict(kwargs))
        original_init(self, run_name, **kwargs)

    monkeypatch.setattr(WandBTracker, "__init__", _capturing_init)
    return captured


def _train_and_checkpoint_at_4(tmp_path: Path, training_parquet: Path) -> str:
    """Phase 1 of the resume tests: train to step 4, checkpoint, return the W&B run id."""
    from oplm.training.trainer import Trainer

    cfg = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=4,
        save_every=4,
        wandb_enabled=True,
        wandb_run_name="e2e-rewind",
    )
    Trainer(cfg, callbacks=[]).train()
    return (tmp_path / "wandb_run_id").read_text().strip()


def _resume_cfg(tmp_path: Path, training_parquet: Path) -> Any:
    return tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=6,
        save_every=6,
        auto_resume=True,
        wandb_enabled=True,
        wandb_run_name="e2e-rewind",
    )


def test_every_wandb_row_is_logged_at_the_training_step(
    training_parquet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Train, eval-free, and both diagnostics callbacks log with ``step=global_step``.

    W&B's ``Step`` is its history counter; passing the training step keeps it equal to
    ``train/global_step`` instead of counting log calls.
    """
    _offline_wandb(tmp_path, monkeypatch)
    from accelerate.tracking import WandBTracker

    from oplm.training.mup import StabilityDiagnosticsCallback
    from oplm.training.trainer import Trainer

    logged: list[tuple[int | None, dict[str, Any]]] = []
    original_log = WandBTracker.log

    def _recording_log(
        self: WandBTracker, values: dict[str, Any], step: int | None = None, **kwargs: Any
    ) -> None:
        logged.append((step, dict(values)))
        original_log(self, values, step=step, **kwargs)

    monkeypatch.setattr(WandBTracker, "log", _recording_log)
    cfg = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=4,
        log_every=1,
        max_grad_norm=1.0,
        weight_diag_every=2,
        wandb_enabled=True,
    )
    Trainer(cfg, callbacks=[StabilityDiagnosticsCallback(probe_every=0)]).train()

    assert logged
    assert all(step is not None and values["train/global_step"] == step for step, values in logged)
    assert [step for step, values in logged if "train/loss" in values] == [1, 2, 3, 4]
    assert [step for step, values in logged if "diag/grad_norm" in values] == [1, 2, 3, 4]
    assert [step for step, values in logged if "diag/weight_step" in values] == [2, 4]


@pytest.mark.parametrize(("server_last_step", "rewinds"), [(6, True), (4, False), (None, False)])
def test_resume_rewinds_wandb_only_when_its_history_is_ahead(
    training_parquet: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    server_last_step: int | None,
    rewinds: bool,
) -> None:
    """Checkpoint 4 with the server at 6 rewinds to step 5; at 4 or unknown, a plain resume."""
    _offline_wandb(tmp_path, monkeypatch)
    import oplm.training.trainer as trainer_module

    run_id = _train_and_checkpoint_at_4(tmp_path, training_parquet)
    captured = _capture_wandb_init_kwargs(monkeypatch)
    monkeypatch.setattr(
        trainer_module, "_wandb_last_history_step", lambda project, rid: server_last_step
    )
    trainer = trainer_module.Trainer(_resume_cfg(tmp_path, training_parquet), callbacks=[])
    trainer.train()

    (kwargs,) = captured
    if rewinds:
        assert kwargs["resume_from"] == f"{run_id}?_step=5"
        assert "id" not in kwargs and "resume" not in kwargs
        assert trainer._wandb_rewind == "applied"
    else:
        assert "resume_from" not in kwargs
        assert kwargs["id"] == run_id and kwargs["resume"] == "allow"
        assert trainer._wandb_rewind is None
    assert (tmp_path / "wandb_run_id").read_text().strip() == run_id


def test_refused_rewind_falls_back_to_a_plain_resume(
    training_parquet: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A server that refuses the rewind (private preview) costs a warning, never the run."""
    _offline_wandb(tmp_path, monkeypatch)
    from accelerate.tracking import WandBTracker

    import oplm.training.trainer as trainer_module

    run_id = _train_and_checkpoint_at_4(tmp_path, training_parquet)
    captured = _capture_wandb_init_kwargs(monkeypatch)
    monkeypatch.setattr(trainer_module, "_wandb_last_history_step", lambda project, rid: 6)
    original_start = WandBTracker.start

    def _refuse_rewind(self: WandBTracker) -> None:
        if "resume_from" in self.init_kwargs:
            raise RuntimeError("Rewind is in private preview -- contact support@wandb.com")
        original_start(self)

    monkeypatch.setattr(WandBTracker, "start", _refuse_rewind)
    import wandb

    teardowns: list[None] = []
    original_teardown = wandb.teardown

    def _counting_teardown(*args: Any, **kwargs: Any) -> None:
        teardowns.append(None)
        original_teardown(*args, **kwargs)

    # A refused online init leaves the run id registered in W&B's service; the fallback must
    # tear the service down before retrying (offline cannot reproduce the "in use" error).
    monkeypatch.setattr(wandb, "teardown", _counting_teardown)
    with caplog.at_level(logging.WARNING, logger="oplm.training.trainer"):
        trainer = trainer_module.Trainer(_resume_cfg(tmp_path, training_parquet), callbacks=[])
    assert len(teardowns) == 1
    assert trainer._wandb_rewind == "unavailable"
    assert len(trainer.accelerator.trackers) == 1
    assert [("resume_from" in k, k.get("id"), k.get("resume")) for k in captured] == [
        (True, None, None),
        (False, run_id, "allow"),
    ]
    assert "W&B history rewind unavailable" in caplog.text
    assert "steps 5-6 keep the earlier attempt's values" in caplog.text
    trainer.train()  # logging goes through the surviving tracker
    assert trainer.global_step == 6


def test_last_history_step_is_not_queried_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Offline (and disabled) W&B never touches the network for the rewind decision."""
    pytest.importorskip("wandb")
    from oplm.training.trainer import _wandb_last_history_step

    for mode in ("offline", "disabled"):
        monkeypatch.setenv("WANDB_MODE", mode)
        assert _wandb_last_history_step("any-project", "any-run") is None


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
