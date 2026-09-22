"""Separate looping stages initialize weights and resume their own training state."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import torch

from oplm.model import OplmForMaskedLM
from oplm.training.trainer import Trainer
from tests.training.conftest import (
    FullRecordingCallback,
    configure_accelerator_device,
    tiny_train_cfg,
)
from tests.training.test_e2e_data_exact import _force_keep_sequence_ids, _wrap_recorder

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("parent_optimizer,optimizer", [("adamw", "muon"), ("muon", "adamw")])
@pytest.mark.parametrize("mode", ["full", "selective"])
def test_new_stage_starts_from_weights_only(
    tmp_path: Path,
    training_parquet: Path,
    monkeypatch: pytest.MonkeyPatch,
    parent_optimizer: str,
    optimizer: str,
    mode: str,
) -> None:
    configure_accelerator_device("cpu", monkeypatch)
    parent_cfg = tiny_train_cfg(
        tmp_path / "parent", training_parquet, max_steps=2, optimizer=parent_optimizer
    )
    Trainer(parent_cfg).train()
    source = tmp_path / "parent" / "checkpoint-2" / "hf"
    expected = OplmForMaskedLM.from_pretrained(source).state_dict()
    cfg = tiny_train_cfg(
        tmp_path / "child",
        training_parquet,
        max_steps=3,
        init_from=str(source),
        num_loops=2,
        loop_strategy="stack",
        lr=2e-4,
        warmup_steps=1,
        optimizer=optimizer,
        gradient_checkpointing=True,
        gradient_checkpointing_mode=mode,
    )
    callback = FullRecordingCallback()
    trainer = Trainer(cfg, callbacks=[callback])
    raw = trainer.accelerator.unwrap_model(trainer.model)
    assert raw.training
    assert raw.oplm.backbone.gradient_checkpointing
    assert raw.oplm.backbone.gradient_checkpointing_mode == mode
    assert trainer.global_step == trainer.tokens_seen == trainer._batches_in_epoch == 0
    assert trainer.epoch == 0
    for key, value in expected.items():
        torch.testing.assert_close(raw.state_dict()[key].cpu(), value.cpu(), rtol=0, atol=0)
    for opt in trainer.optimizers:
        assert not opt.state_dict()["state"]
        assert opt.param_groups[0]["lr"] == 0
    trainer.train()
    assert trainer.global_step == 3
    assert callback.train_logs[0][1]["train/lr"] == pytest.approx(2e-4)
    assert (tmp_path / "child" / "checkpoint-3" / "hf" / "config.json").is_file()


@pytest.mark.parametrize("auto", [False, True])
def test_child_resume_never_accesses_parent(
    tmp_path: Path,
    training_parquet: Path,
    monkeypatch: pytest.MonkeyPatch,
    auto: bool,
) -> None:
    configure_accelerator_device("cpu", monkeypatch)
    _force_keep_sequence_ids(monkeypatch)
    parent = tiny_train_cfg(tmp_path / "parent", training_parquet, max_steps=2)
    Trainer(parent).train()
    source = tmp_path / "parent" / "checkpoint-2"
    options = dict(
        max_steps=3,
        save_every=2,
        init_from=str(source),
        num_loops=2,
        loop_strategy="interleave",
        warmup_steps=1,
    )
    control = Trainer(tiny_train_cfg(tmp_path / "child", training_parquet, **options))
    rows = _wrap_recorder(control)
    control.train()
    source.rename(source.with_name("unavailable"))
    (tmp_path / "child" / "checkpoint-3").rename(tmp_path / "control-final")
    import oplm.training.initialization as initialization

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("resume must not inspect init_from")

    monkeypatch.setattr(initialization, "resolve_initialization_source", forbidden)
    monkeypatch.setattr(initialization, "load_initial_model", forbidden)
    resumed = Trainer(
        tiny_train_cfg(
            tmp_path / "child",
            training_parquet,
            **options,
            auto_resume=auto,
            resume_from=None if auto else str(tmp_path / "child" / "checkpoint-2"),
        )
    )
    assert resumed.global_step == 2
    assert resumed.tokens_seen > 0
    assert resumed._batches_in_epoch == 2
    assert all(opt.state_dict()["state"] for opt in resumed.optimizers)
    assert resumed.schedulers[0].state_dict()["last_epoch"] == 2
    resumed_rows = _wrap_recorder(resumed)
    resumed.train()
    assert resumed_rows.batches == rows.batches[2:]
    assert resumed.tokens_seen == control.tokens_seen


def test_empty_auto_resume_initializes_and_bad_explicit_resume_fails(
    tmp_path: Path,
    training_parquet: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_accelerator_device("cpu", monkeypatch)
    cfg = tiny_train_cfg(tmp_path / "scratch", training_parquet)
    parent = Trainer(cfg)
    source = tmp_path / "export"
    parent.accelerator.unwrap_model(parent.model).save_pretrained(source)
    child = Trainer(
        tiny_train_cfg(
            tmp_path / "child",
            training_parquet,
            init_from=str(source),
            auto_resume=True,
            num_loops=2,
        )
    )
    assert child.global_step == 0
    assert child.accelerator.unwrap_model(child.model).oplm.backbone.layer_execution_order == (
        0,
        1,
        0,
        1,
    )
    with pytest.raises((ValueError, FileNotFoundError)):
        Trainer(
            tiny_train_cfg(
                tmp_path / "bad",
                training_parquet,
                init_from=str(source),
                resume_from=str(tmp_path / "missing"),
            )
        )


@pytest.mark.parametrize("use_parent_run", [False, True])
def test_output_alias_rejected_before_config_or_tracker_write(
    tmp_path: Path,
    training_parquet: Path,
    monkeypatch: pytest.MonkeyPatch,
    use_parent_run: bool,
) -> None:
    configure_accelerator_device("cpu", monkeypatch)
    run = tmp_path / "parent"
    export = run / "checkpoint-2" / "hf"
    cfg = tiny_train_cfg(run, training_parquet)
    OplmForMaskedLM(cfg.model).save_pretrained(export)
    original = run / "config.yaml"
    original.write_text("parent config must survive")
    alias = tmp_path / "alias"
    alias.symlink_to(run if use_parent_run else export, target_is_directory=True)
    from accelerate import Accelerator

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("must validate before initializing tracker")

    monkeypatch.setattr(Accelerator, "init_trackers", forbidden)
    with pytest.raises(ValueError, match="output_dir"):
        Trainer(tiny_train_cfg(alias, training_parquet, init_from=str(export), wandb_enabled=True))
    assert original.read_text() == "parent config must survive"


def test_new_stage_does_not_reuse_parent_tracker_id(
    tmp_path: Path,
    training_parquet: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_accelerator_device("cpu", monkeypatch)
    from types import SimpleNamespace

    import wandb
    from accelerate import Accelerator

    parent = tmp_path / "parent"
    source = parent / "checkpoint-2" / "hf"
    cfg = tiny_train_cfg(parent, training_parquet)
    OplmForMaskedLM(cfg.model).save_pretrained(source)
    (parent / "wandb_run_id").write_text("parent-id")
    (source.parent / "trainer_state.json").write_text('{"wandb_run_id": "parent-id"}')
    captured = []

    def track(self: Accelerator, **kwargs: Any) -> None:
        captured.append(kwargs["init_kwargs"]["wandb"])
        monkeypatch.setattr(wandb, "run", SimpleNamespace(id="new-stage"))

    monkeypatch.setattr(Accelerator, "init_trackers", track)
    trainer = Trainer(
        tiny_train_cfg(
            tmp_path / "child", training_parquet, init_from=str(source), wandb_enabled=True
        )
    )
    assert "id" not in captured[0] and "resume" not in captured[0]
    assert trainer._wandb_run_id == "new-stage"
