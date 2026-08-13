"""DCP checkpoint format (Task 2.1): save/load through ``torch.distributed.checkpoint``.

``save_checkpoint`` no longer calls ``accelerator.save_state`` (which wrote
``model.safetensors``/``optimizer.bin`` via Accelerate's own serialization). Model and
optimizer state now go through ``torch.distributed.checkpoint`` (DCP) via
``get_state_dict``/``set_state_dict``, producing ``.metadata`` + ``__*.distcp`` shard
files at the checkpoint root instead. Per-rank RNG state is a small ``torch.save``
sidecar (``rng_state_<rank>.pt``) written alongside. ``trainer_state.json``,
``config.yaml``, and the ``hf/`` export (which keeps its own ``model.safetensors``) are
unchanged from Phase 1.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch

from oplm.config import DataConfig, OplmConfig, TrainConfig
from oplm.model import OplmConfig as OplmModelConfig
from oplm.model import OplmForMaskedLM
from oplm.training.checkpoint import (
    PendingSave,
    finalize_pending_save,
    load_checkpoint,
    save_checkpoint,
)
from oplm.training.optim import build_optimizers, build_schedulers
from tests.training.conftest import configure_accelerator_device, tiny_train_cfg

if TYPE_CHECKING:
    from accelerate import Accelerator

pytestmark = pytest.mark.slow


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
    """Build and prepare a tiny model + AdamW optimizer + LambdaLR scheduler on CPU."""
    from accelerate import Accelerator

    accelerator = Accelerator(cpu=True, mixed_precision="no")
    model = OplmForMaskedLM(cfg.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    model, optimizer = accelerator.prepare(model, optimizer)
    return accelerator, model, optimizer, scheduler


def _take_optimizer_step(
    cfg: OplmConfig,
    accelerator: Accelerator,
    model: OplmForMaskedLM,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
) -> None:
    """Run one real forward/backward/step so optimizer + scheduler state is non-trivial."""
    inputs = torch.randint(0, cfg.model.vocab_size, (2, 8))
    loss = model(input_ids=inputs, labels=inputs).loss
    accelerator.backward(loss)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()


def test_dcp_checkpoint_format_written_at_checkpoint_root(tmp_path: Path) -> None:
    """A single-process save writes DCP shard files + RNG sidecar, not Accelerate's format."""
    cfg = _cfg()
    accelerator, model, optimizer, scheduler = _prepared_model(cfg)
    _take_optimizer_step(cfg, accelerator, model, optimizer, scheduler)

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

    # DCP format present: metadata + at least one shard file.
    assert (committed / ".metadata").exists()
    assert list(committed.glob("__*.distcp")), "expected at least one __*.distcp shard file"

    # Per-rank RNG sidecar present for rank 0 (single process).
    assert (committed / "rng_state_0.pt").exists()

    # Phase 1 assets unchanged.
    assert (committed / "trainer_state.json").exists()
    assert (committed / "config.yaml").exists()
    assert (committed / "hf" / "config.json").exists()
    assert (committed / "hf" / "model.safetensors").exists()

    # Accelerate's save_state format is gone from the checkpoint ROOT (hf/ keeps its
    # own model.safetensors, which is a distinct, intentional artifact).
    assert not (committed / "model.safetensors").exists()
    assert not (committed / "optimizer.bin").exists()


def test_dcp_save_load_round_trip_restores_model_optimizer_and_scheduler(tmp_path: Path) -> None:
    """Saving then loading via DCP restores weights, optimizer momentum, and scheduler state."""
    cfg = _cfg()
    accelerator, model, optimizer, scheduler = _prepared_model(cfg)

    # A few real steps so AdamW momentum buffers and the scheduler's step count are
    # non-trivial -- a checkpoint that silently dropped this state would look correct
    # at step 0 but wrong here.
    for _ in range(3):
        _take_optimizer_step(cfg, accelerator, model, optimizer, scheduler)

    unwrapped = accelerator.unwrap_model(model)
    original_weight = unwrapped.lm_head.decoder.bias.detach().clone()
    original_optimizer_state = optimizer.state_dict()
    original_scheduler_state = scheduler.state_dict()

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

    fresh_accelerator, fresh_model, fresh_optimizer, fresh_scheduler = _prepared_model(cfg)
    state = load_checkpoint(
        fresh_accelerator,
        fresh_model,
        [fresh_optimizer],
        [fresh_scheduler],
        str(tmp_path / "checkpoint-10"),
        cfg,
    )

    # Task 2.2: load_checkpoint's returned trainer-state dict restores step/epoch/tokens.
    assert state["global_step"] == 10
    assert state["epoch"] == 1
    assert state["tokens_seen"] == 400
    assert state["samples_seen"] == 40

    fresh_unwrapped = fresh_accelerator.unwrap_model(fresh_model)
    assert torch.allclose(fresh_unwrapped.lm_head.decoder.bias.detach(), original_weight)

    # Optimizer momentum buffers (exp_avg/exp_avg_sq) round-trip, keyed by param index.
    restored_optimizer_state = fresh_optimizer.state_dict()
    for param_id, original_param_state in original_optimizer_state["state"].items():
        restored_param_state = restored_optimizer_state["state"][param_id]
        assert torch.allclose(restored_param_state["exp_avg"], original_param_state["exp_avg"])
        assert torch.allclose(
            restored_param_state["exp_avg_sq"], original_param_state["exp_avg_sq"]
        )

    assert fresh_scheduler.state_dict() == original_scheduler_state


def test_dcp_save_load_round_trip_with_compile(tmp_path: Path, reset_dynamo: None) -> None:
    """The DCP save/load path works when the model is wrapped in torch.compile.

    ``get_state_dict``/``set_state_dict`` resolve ``OptimizedModule``'s ``_orig_mod``
    wrapping automatically, so no special-casing is needed beyond what the ``hf/``
    export already does for ``save_pretrained``.
    """
    from accelerate import Accelerator

    cfg = _cfg()
    accelerator = Accelerator(cpu=True, mixed_precision="no")
    model = OplmForMaskedLM(cfg.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    model = torch.compile(model, dynamic=True)
    model, optimizer = accelerator.prepare(model, optimizer)

    _take_optimizer_step(cfg, accelerator, model, optimizer, scheduler)
    original_weight = (
        accelerator.unwrap_model(model)._orig_mod.lm_head.decoder.bias.detach().clone()
    )

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
    assert (committed / ".metadata").exists()
    assert not (committed / "model.safetensors").exists()

    fresh_model = OplmForMaskedLM(cfg.model)
    fresh_optimizer = torch.optim.AdamW(fresh_model.parameters(), lr=1e-3)
    fresh_scheduler = torch.optim.lr_scheduler.LambdaLR(fresh_optimizer, lambda _step: 1.0)
    load_checkpoint(
        accelerator, fresh_model, [fresh_optimizer], [fresh_scheduler], str(committed), cfg
    )
    assert torch.allclose(fresh_model.lm_head.decoder.bias.detach(), original_weight)


def test_scaler_sidecar_round_trips_fp16_gradscaler_state(tmp_path: Path) -> None:
    """The fp16 GradScaler's state round-trips through the ``scaler.pt`` sidecar.

    ``accelerator.save_state`` used to serialize GradScaler state as part of Accelerate's
    own checkpoint format; the DCP rewrite needs an explicit replacement (see
    ``_write_scaler_sidecar``/``_restore_scaler_sidecar``). A full fp16 end-to-end pilot
    run is not exercisable in this environment: ``Accelerate(cpu=True,
    mixed_precision="fp16")`` leaves ``accelerator.scaler`` as ``None`` (Accelerate's
    ``get_grad_scaler`` does not construct a CPU-backed fp16 scaler), so there is no CPU
    path that ever produces a non-``None`` ``accelerator.scaler`` to exercise. Instead we
    attach a real ``torch.amp.GradScaler`` directly onto a CPU accelerator's ``.scaler``
    attribute (exactly the attribute ``save_checkpoint``/``load_checkpoint`` read), give
    it non-default state via ``load_state_dict`` (real fp16 training would reach this
    state through ``scaler.scale(loss).backward()`` + ``scaler.update()`` on a GPU), and
    drive the save/load helpers exactly as the trainer does.
    """
    cfg = _cfg()
    accelerator, model, optimizer, scheduler = _prepared_model(cfg)

    non_default_scaler_state = {
        "scale": 4096.0,
        "growth_factor": 2.0,
        "backoff_factor": 0.5,
        "growth_interval": 2000,
        "_growth_tracker": 7,
    }
    scaler = torch.amp.GradScaler()
    scaler.load_state_dict(non_default_scaler_state)
    accelerator.scaler = scaler

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
    assert (committed / "scaler.pt").exists()

    fresh_accelerator, fresh_model, fresh_optimizer, fresh_scheduler = _prepared_model(cfg)
    fresh_accelerator.scaler = torch.amp.GradScaler()  # starts from defaults

    load_checkpoint(
        fresh_accelerator,
        fresh_model,
        [fresh_optimizer],
        [fresh_scheduler],
        str(committed),
        cfg,
    )

    assert fresh_accelerator.scaler.state_dict() == non_default_scaler_state


def test_scaler_sidecar_absent_when_no_scaler(tmp_path: Path) -> None:
    """No ``scaler.pt`` is written when ``accelerator.scaler`` is ``None`` (the default)."""
    cfg = _cfg()
    accelerator, model, optimizer, scheduler = _prepared_model(cfg)
    assert accelerator.scaler is None

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

    assert not (tmp_path / "checkpoint-10" / "scaler.pt").exists()


def _muon_cfg() -> OplmConfig:
    """Tiny root config using the Muon+auxiliary-AdamW dual-optimizer path."""
    return OplmConfig(
        model=OplmModelConfig(
            hidden_size=32,
            num_attention_heads=4,
            num_hidden_layers=2,
            max_position_embeddings=64,
        ),
        train=TrainConfig(wandb_enabled=False, mixed_precision="no", optimizer="muon"),
        data=DataConfig(num_workers=0, pin_memory=False),
    )


def test_dcp_round_trip_preserves_muon_and_aux_adamw_ordering(tmp_path: Path) -> None:
    """The [Muon, aux AdamW] optimizer pair round-trips through DCP in the right order.

    ``build_optimizers(cfg.train.optimizer="muon")`` returns a *list* of two distinct
    optimizer instances (Muon for eligible 2D hidden weights, AdamW for everything else),
    each with its own scheduler. ``_ModelOptState`` passes this list straight through to
    ``get_state_dict``/``set_state_dict``, which key state by parameter identity, not by
    optimizer index -- but a mismatched save/load order (or a bug that only ever
    exercised the single-optimizer AdamW path) could still silently mix up which
    optimizer's state lands where. Saving into freshly-built, never-stepped optimizers
    and asserting both Muon's ``momentum_buffer`` and the aux AdamW's ``exp_avg`` pins
    this exactly.
    """
    from accelerate import Accelerator

    cfg = _muon_cfg()
    accelerator = Accelerator(cpu=True, mixed_precision="no")
    model = OplmForMaskedLM(cfg.model)
    optimizers = build_optimizers(model, cfg.train)
    schedulers = build_schedulers(optimizers, cfg.train, total_steps=100)
    assert len(optimizers) == 2, "expected [Muon, aux AdamW]"
    assert type(optimizers[0]).__name__ == "Muon"

    prepared = accelerator.prepare(model, *optimizers)
    model = prepared[0]
    optimizers = list(prepared[1:])

    for _ in range(3):
        inputs = torch.randint(0, cfg.model.vocab_size, (2, 8))
        loss = model(input_ids=inputs, labels=inputs).loss
        accelerator.backward(loss)
        for optimizer in optimizers:
            optimizer.step()
        for sched in schedulers:
            sched.step()
        for optimizer in optimizers:
            optimizer.zero_grad()

    muon_optimizer = getattr(optimizers[0], "optimizer", optimizers[0])
    aux_adamw_optimizer = getattr(optimizers[1], "optimizer", optimizers[1])
    muon_param = muon_optimizer.param_groups[0]["params"][0]
    adamw_param = aux_adamw_optimizer.param_groups[0]["params"][0]
    original_muon_momentum = muon_optimizer.state[muon_param]["momentum_buffer"].clone()
    original_adamw_exp_avg = aux_adamw_optimizer.state[adamw_param]["exp_avg"].clone()

    save_checkpoint(
        accelerator=accelerator,
        model=model,
        optimizers=optimizers,
        schedulers=schedulers,
        cfg=cfg,
        output_dir=str(tmp_path),
        global_step=10,
        epoch=1,
        samples_seen=40,
        tokens_seen=400,
    )

    # Fresh, never-stepped optimizers -- their .state dicts start empty, so any restored
    # state can only have come from the checkpoint.
    fresh_model = OplmForMaskedLM(cfg.model)
    fresh_optimizers = build_optimizers(fresh_model, cfg.train)
    fresh_schedulers = build_schedulers(fresh_optimizers, cfg.train, total_steps=100)
    state = load_checkpoint(
        accelerator,
        fresh_model,
        fresh_optimizers,
        fresh_schedulers,
        str(tmp_path / "checkpoint-10"),
        cfg,
    )

    # Task 2.2: step/epoch/tokens restore alongside Muon's own momentum buffer below.
    assert state["global_step"] == 10
    assert state["epoch"] == 1
    assert state["tokens_seen"] == 400

    fresh_muon_optimizer = getattr(fresh_optimizers[0], "optimizer", fresh_optimizers[0])
    fresh_adamw_optimizer = getattr(fresh_optimizers[1], "optimizer", fresh_optimizers[1])
    fresh_muon_param = fresh_muon_optimizer.param_groups[0]["params"][0]
    fresh_adamw_param = fresh_adamw_optimizer.param_groups[0]["params"][0]

    restored_muon_momentum = fresh_muon_optimizer.state[fresh_muon_param]["momentum_buffer"]
    restored_adamw_exp_avg = fresh_adamw_optimizer.state[fresh_adamw_param]["exp_avg"]

    assert torch.allclose(restored_muon_momentum, original_muon_momentum)
    assert torch.allclose(restored_adamw_exp_avg, original_adamw_exp_avg)


# --- Task 2.2: schedule-compat validation --------------------------------------------


def test_load_checkpoint_raises_on_schedule_mismatch(tmp_path: Path) -> None:
    """A ``warmup_steps`` change between save and resume raises, naming the field."""
    save_cfg = _cfg()
    accelerator, model, optimizer, scheduler = _prepared_model(save_cfg)
    _take_optimizer_step(save_cfg, accelerator, model, optimizer, scheduler)

    save_checkpoint(
        accelerator=accelerator,
        model=model,
        optimizers=[optimizer],
        schedulers=[scheduler],
        cfg=save_cfg,
        output_dir=str(tmp_path),
        global_step=10,
        epoch=1,
        samples_seen=40,
        tokens_seen=400,
    )

    load_cfg = _cfg()
    load_cfg.train.warmup_steps = save_cfg.train.warmup_steps + 500

    fresh_accelerator, fresh_model, fresh_optimizer, fresh_scheduler = _prepared_model(load_cfg)
    with pytest.raises(ValueError, match="warmup_steps"):
        load_checkpoint(
            fresh_accelerator,
            fresh_model,
            [fresh_optimizer],
            [fresh_scheduler],
            str(tmp_path / "checkpoint-10"),
            load_cfg,
        )


def test_load_checkpoint_schedule_mismatch_error_names_both_values(tmp_path: Path) -> None:
    """The mismatch error names both the checkpoint's and the live config's values."""
    save_cfg = _cfg()
    accelerator, model, optimizer, scheduler = _prepared_model(save_cfg)
    _take_optimizer_step(save_cfg, accelerator, model, optimizer, scheduler)

    save_checkpoint(
        accelerator=accelerator,
        model=model,
        optimizers=[optimizer],
        schedulers=[scheduler],
        cfg=save_cfg,
        output_dir=str(tmp_path),
        global_step=10,
        epoch=1,
        samples_seen=40,
        tokens_seen=400,
    )

    load_cfg = _cfg()
    load_cfg.train.min_lr = save_cfg.train.min_lr + 1e-5
    load_cfg.train.scheduler = "wsd_linear"
    load_cfg.train.stable_steps = save_cfg.train.stable_steps + 1

    fresh_accelerator, fresh_model, fresh_optimizer, fresh_scheduler = _prepared_model(load_cfg)
    with pytest.raises(ValueError) as excinfo:
        load_checkpoint(
            fresh_accelerator,
            fresh_model,
            [fresh_optimizer],
            [fresh_scheduler],
            str(tmp_path / "checkpoint-10"),
            load_cfg,
        )

    message = str(excinfo.value)
    for field in ("min_lr", "scheduler", "stable_steps"):
        assert field in message
    # warmup_steps/max_steps/lr were unchanged and must not be flagged.
    assert "warmup_steps" not in message
    assert "max_steps" not in message


# --- Task 2.2 fix round: asymmetric max_steps policy ----------------------------------


def test_load_checkpoint_raises_when_live_max_steps_decreases(tmp_path: Path) -> None:
    """A ``max_steps`` *decrease* between save and resume raises, naming the field.

    Unlike an increase (a supported run-extension workflow -- see the next test), a
    smaller live ``max_steps`` is essentially always accidental, and the checkpoint's own
    ``global_step`` could even already exceed the new, shrunk total.
    """
    save_cfg = _cfg()
    save_cfg.train.max_steps = 100
    accelerator, model, optimizer, scheduler = _prepared_model(save_cfg)
    _take_optimizer_step(save_cfg, accelerator, model, optimizer, scheduler)

    save_checkpoint(
        accelerator=accelerator,
        model=model,
        optimizers=[optimizer],
        schedulers=[scheduler],
        cfg=save_cfg,
        output_dir=str(tmp_path),
        global_step=10,
        epoch=1,
        samples_seen=40,
        tokens_seen=400,
    )

    load_cfg = _cfg()
    load_cfg.train.max_steps = 50  # decreased from 100

    fresh_accelerator, fresh_model, fresh_optimizer, fresh_scheduler = _prepared_model(load_cfg)
    with pytest.raises(ValueError, match="max_steps"):
        load_checkpoint(
            fresh_accelerator,
            fresh_model,
            [fresh_optimizer],
            [fresh_scheduler],
            str(tmp_path / "checkpoint-10"),
            load_cfg,
        )


def test_load_checkpoint_raises_when_live_max_steps_leaves_nothing_to_train(
    tmp_path: Path,
) -> None:
    """``live max_steps <= checkpoint global_step`` raises even though ``max_steps`` rose.

    global_step=10 with checkpoint max_steps=10 (training already reached its target at
    save time); resuming with live max_steps=10 (unchanged, not even a decrease) would
    have nothing left to train, so this is still rejected.
    """
    save_cfg = _cfg()
    save_cfg.train.max_steps = 10
    accelerator, model, optimizer, scheduler = _prepared_model(save_cfg)
    _take_optimizer_step(save_cfg, accelerator, model, optimizer, scheduler)

    save_checkpoint(
        accelerator=accelerator,
        model=model,
        optimizers=[optimizer],
        schedulers=[scheduler],
        cfg=save_cfg,
        output_dir=str(tmp_path),
        global_step=10,
        epoch=1,
        samples_seen=40,
        tokens_seen=400,
    )

    load_cfg = _cfg()
    load_cfg.train.max_steps = 10  # equal to checkpoint's max_steps, but == global_step too

    fresh_accelerator, fresh_model, fresh_optimizer, fresh_scheduler = _prepared_model(load_cfg)
    with pytest.raises(ValueError, match="max_steps"):
        load_checkpoint(
            fresh_accelerator,
            fresh_model,
            [fresh_optimizer],
            [fresh_scheduler],
            str(tmp_path / "checkpoint-10"),
            load_cfg,
        )


def test_load_checkpoint_warns_and_proceeds_when_live_max_steps_increases(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A ``max_steps`` *increase* (deliberate run extension) warns but does not raise."""
    save_cfg = _cfg()
    save_cfg.train.max_steps = 10
    accelerator, model, optimizer, scheduler = _prepared_model(save_cfg)
    _take_optimizer_step(save_cfg, accelerator, model, optimizer, scheduler)

    save_checkpoint(
        accelerator=accelerator,
        model=model,
        optimizers=[optimizer],
        schedulers=[scheduler],
        cfg=save_cfg,
        output_dir=str(tmp_path),
        global_step=10,
        epoch=1,
        samples_seen=40,
        tokens_seen=400,
    )

    load_cfg = _cfg()
    load_cfg.train.max_steps = 20  # increased from 10 -- deliberate extension

    fresh_accelerator, fresh_model, fresh_optimizer, fresh_scheduler = _prepared_model(load_cfg)
    with caplog.at_level(logging.WARNING):
        state = load_checkpoint(
            fresh_accelerator,
            fresh_model,
            [fresh_optimizer],
            [fresh_scheduler],
            str(tmp_path / "checkpoint-10"),
            load_cfg,
        )

    assert state["global_step"] == 10
    assert "max_steps" in caplog.text
    assert "10" in caplog.text and "20" in caplog.text


# --- Task 2.2: hard RNG-sidecar error + OPLM_ALLOW_MISSING_RNG escape hatch -----------


def test_load_checkpoint_raises_on_missing_rng_sidecar(tmp_path: Path) -> None:
    """A missing ``rng_state_<rank>.pt`` sidecar is a hard error by default."""
    cfg = _cfg()
    accelerator, model, optimizer, scheduler = _prepared_model(cfg)
    _take_optimizer_step(cfg, accelerator, model, optimizer, scheduler)

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
    (committed / "rng_state_0.pt").unlink()

    fresh_accelerator, fresh_model, fresh_optimizer, fresh_scheduler = _prepared_model(cfg)
    with pytest.raises(RuntimeError, match="RNG sidecar not found"):
        load_checkpoint(
            fresh_accelerator,
            fresh_model,
            [fresh_optimizer],
            [fresh_scheduler],
            str(committed),
            cfg,
        )


def test_load_checkpoint_allows_missing_rng_sidecar_via_env_escape_hatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """``OPLM_ALLOW_MISSING_RNG=1`` turns a missing sidecar into a warning, not an error."""
    cfg = _cfg()
    accelerator, model, optimizer, scheduler = _prepared_model(cfg)
    _take_optimizer_step(cfg, accelerator, model, optimizer, scheduler)

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
    (committed / "rng_state_0.pt").unlink()

    monkeypatch.setenv("OPLM_ALLOW_MISSING_RNG", "1")
    fresh_accelerator, fresh_model, fresh_optimizer, fresh_scheduler = _prepared_model(cfg)
    with caplog.at_level(logging.WARNING):
        state = load_checkpoint(
            fresh_accelerator,
            fresh_model,
            [fresh_optimizer],
            [fresh_scheduler],
            str(committed),
            cfg,
        )

    assert state["global_step"] == 10
    assert "RNG sidecar not found" in caplog.text
    assert "OPLM_ALLOW_MISSING_RNG" in caplog.text


# --- Task 2.3: async save --------------------------------------------------------------


def test_save_checkpoint_async_writes_sidecars_without_committing(tmp_path: Path) -> None:
    """``blocking=False`` writes sidecars synchronously into ``tmp_dir`` but defers the commit.

    ``dcp.async_save``'s synchronous staging step returns fast (the actual disk write
    runs on a background thread), but the sidecars -- RNG, ``trainer_state.json``,
    ``config.yaml``, ``hf/`` export -- are always written synchronously regardless of
    ``blocking``, since they must snapshot trigger-time state. The commit (rename to
    ``checkpoint-<step>/``, ``latest`` pointer, rotation) must NOT have happened yet:
    that is exactly what :func:`finalize_pending_save` defers.
    """
    cfg = _cfg()
    accelerator, model, optimizer, scheduler = _prepared_model(cfg)
    _take_optimizer_step(cfg, accelerator, model, optimizer, scheduler)

    result = save_checkpoint(
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
        blocking=False,
    )

    assert isinstance(result, PendingSave)
    assert result.final_dir == tmp_path / "checkpoint-10"

    tmp_dir = tmp_path / "checkpoint-10.tmp"
    assert tmp_dir.is_dir()
    assert (tmp_dir / "rng_state_0.pt").exists()
    assert (tmp_dir / "trainer_state.json").exists()
    assert (tmp_dir / "config.yaml").exists()
    assert (tmp_dir / "hf" / "config.json").exists()

    # Not committed yet: no final dir, no latest pointer.
    assert not (tmp_path / "checkpoint-10").exists()
    assert not (tmp_path / "latest").exists()

    # Clean up the still-pending write so this test doesn't leak a background thread
    # holding file handles open past the test's tmp_path teardown.
    result.future.result()


def test_finalize_pending_save_completes_the_commit(tmp_path: Path) -> None:
    """``finalize_pending_save`` blocks on the future and runs the exact same commit tail.

    The committed checkpoint produced via the async path must be indistinguishable
    from one produced via the blocking path: same DCP shard/metadata files, same
    sidecars, same ``latest`` pointer, no leftover ``.tmp`` dir.
    """
    cfg = _cfg()
    accelerator, model, optimizer, scheduler = _prepared_model(cfg)
    _take_optimizer_step(cfg, accelerator, model, optimizer, scheduler)

    pending = save_checkpoint(
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
        blocking=False,
    )
    assert isinstance(pending, PendingSave)

    final_dir = finalize_pending_save(accelerator, pending)

    assert final_dir == tmp_path / "checkpoint-10"
    committed = tmp_path / "checkpoint-10"
    assert committed.is_dir()
    assert (committed / ".metadata").exists()
    assert list(committed.glob("__*.distcp"))
    assert (committed / "rng_state_0.pt").exists()
    assert (committed / "trainer_state.json").exists()
    assert (committed / "hf" / "model.safetensors").exists()
    assert not (tmp_path / "checkpoint-10.tmp").exists()
    assert (tmp_path / "latest").read_text().strip() == "checkpoint-10"

    # Round-trips through load_checkpoint exactly like a blocking-saved checkpoint.
    fresh_accelerator, fresh_model, fresh_optimizer, fresh_scheduler = _prepared_model(cfg)
    state = load_checkpoint(
        fresh_accelerator, fresh_model, [fresh_optimizer], [fresh_scheduler], str(committed), cfg
    )
    assert state["global_step"] == 10
    assert state["tokens_seen"] == 400


def test_async_periodic_saves_defer_commit_past_the_triggering_step(
    training_parquet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real pilot run: ``save_every=2``/``max_steps=8`` commits 4 checkpoints via async saves.

    Periodic saves default to ``blocking=False`` (Task 2.3): ``dcp.async_save`` returns a
    ``Future`` immediately and the trainer keeps training while the background write
    proceeds; the actual commit (rename/pointer/rotation) is deferred to a later,
    rank-synchronized point. This spies on both the trigger point
    (``oplm.training.checkpoint.save_checkpoint`` called with ``blocking=False``,
    recording ``global_step``) and the finalize/commit point
    (``Trainer._finalize_pending_save``, recording ``global_step`` at commit time) to
    prove that at least one checkpoint's commit happens strictly after further training
    steps ran -- not merely that ``async_save`` "returns fast" (a trivially-fast
    background write could satisfy that even under a synchronous same-step commit).
    """
    from oplm.training import checkpoint as checkpoint_module
    from oplm.training.trainer import Trainer
    from tests.training.conftest import FullRecordingCallback, tiny_train_cfg

    cfg = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=8,
        save_every=2,
        save_total_limit=10,
        log_every=1,
    )
    callback = FullRecordingCallback()
    trainer = Trainer(cfg, callbacks=[callback])

    trigger_steps: list[int] = []
    real_save_checkpoint = checkpoint_module.save_checkpoint

    def _spy_save_checkpoint(*args: object, **kwargs: object) -> object:
        if kwargs.get("blocking", True) is False:
            trigger_steps.append(int(kwargs["global_step"]))  # type: ignore[arg-type]
        return real_save_checkpoint(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(checkpoint_module, "save_checkpoint", _spy_save_checkpoint)

    commit_steps: list[int] = []
    real_finalize = trainer._finalize_pending_save

    def _spy_finalize() -> None:
        commit_steps.append(trainer.global_step)
        real_finalize()

    monkeypatch.setattr(trainer, "_finalize_pending_save", _spy_finalize)

    trainer.train()

    assert trigger_steps == [2, 4, 6, 8]
    assert len(commit_steps) == 4
    # commit_steps records the trainer's live global_step at the moment each commit
    # actually runs -- necessarily >= the checkpoint's own (trigger-time) step, since
    # commits only happen at or after the step that triggered them, never before.
    assert callback.checkpoint_steps == [2, 4, 6, 8]
    for trigger, commit in zip(trigger_steps, commit_steps, strict=True):
        assert commit >= trigger

    # The key regression this test guards against: at least one checkpoint's commit
    # must land strictly after the step that triggered its async save -- i.e. the
    # commit was genuinely deferred, not folded back into a synchronous same-step
    # rename.
    assert any(
        commit > trigger for trigger, commit in zip(trigger_steps, commit_steps, strict=True)
    )

    committed = sorted(p.name for p in tmp_path.iterdir() if p.name.startswith("checkpoint-"))
    assert committed == ["checkpoint-2", "checkpoint-4", "checkpoint-6", "checkpoint-8"]
    assert list(tmp_path.glob("checkpoint-*.tmp")) == []
    assert trainer._pending_save is None


def test_drain_during_pending_async_save_finalizes_then_takes_blocking_save(
    training_parquet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A drain landing while an async periodic save is still pending finalizes it first.

    Forces the async save's ``Future`` to still be unresolved at the exact moment drain
    is observed (via a controlled ``concurrent.futures.Future`` substituted for the real
    one, with the real write still happening synchronously underneath -- see
    ``_fake_async_save`` below), so the force-finalize branch in ``Trainer.train``'s
    drain handling is what actually completes the commit, not the opportunistic
    per-step gate. Drain's own checkpoint must then be a separate, blocking save: both
    checkpoints must exist, in commit order, and no ``PendingSave``/``.tmp`` dir must be
    left behind.
    """
    from concurrent.futures import Future

    from oplm.training import checkpoint as checkpoint_module
    from oplm.training.trainer import Trainer
    from tests.training.conftest import FullRecordingCallback, tiny_train_cfg

    cfg = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=20,
        save_every=2,
        save_total_limit=10,
        log_every=1,
    )
    callback = FullRecordingCallback()
    trainer = Trainer(cfg, callbacks=[callback])

    real_dcp_save = checkpoint_module.dcp.save

    def _fake_async_save(state_dict: object, *, checkpoint_id: str, **_: object) -> Future[object]:
        # Do the real write synchronously -- only the Future's completion signal is
        # test-controlled, so the on-disk result is genuine while the "still pending"
        # window is fully deterministic instead of racing a real background thread.
        real_dcp_save(state_dict, checkpoint_id=checkpoint_id)
        return Future()

    monkeypatch.setattr(checkpoint_module.dcp, "async_save", _fake_async_save)

    # Drain becomes "requested" only once training has passed step 2 -- i.e. after the
    # save_every=2 periodic save has already triggered (and is still pending, since its
    # controlled Future is never resolved except by the spy below).
    class _DrainAfterStep:
        def __init__(self, trainer: Trainer, step_threshold: int) -> None:
            self._trainer = trainer
            self._step_threshold = step_threshold

        @property
        def requested(self) -> bool:
            return self._trainer.global_step >= self._step_threshold

    monkeypatch.setattr(trainer, "_drain_signal", _DrainAfterStep(trainer, 3))

    finalize_future_done_at_call: list[bool] = []
    real_finalize = trainer._finalize_pending_save

    def _spy_finalize() -> None:
        pending = trainer._pending_save
        assert pending is not None
        finalize_future_done_at_call.append(pending.handle.future.done())
        if not pending.handle.future.done():
            pending.handle.future.set_result(None)
        real_finalize()

    monkeypatch.setattr(trainer, "_finalize_pending_save", _spy_finalize)

    with pytest.raises(SystemExit) as excinfo:
        trainer.train()

    from oplm.training.signals import DRAIN_EXIT_CODE

    assert excinfo.value.code == DRAIN_EXIT_CODE

    # The force-finalize branch (drain), not the opportunistic per-step gate, is what
    # completed the commit: the future was genuinely still pending when observed.
    assert finalize_future_done_at_call == [False]

    assert callback.checkpoint_steps == [2, 3]
    assert (tmp_path / "checkpoint-2").is_dir()
    assert (tmp_path / "checkpoint-3").is_dir()
    assert list(tmp_path.glob("checkpoint-*.tmp")) == []
    assert trainer._pending_save is None


# --- Task 2.4: reshard e2e (world-size 2 -> 1) -----------------------------------------

_RESHARD_WORKER_MAX_STEPS = 3
_RESHARD_RESUME_EXTRA_STEPS = 2


def test_dcp_checkpoint_saved_at_world_size_2_resumes_at_world_size_1(
    training_parquet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ws=2 pilot's checkpoint loads at ws=1: restored step, weights, and continued training.

    This is the headline resilience property the DCP checkpoint format exists to
    provide: ``get_state_dict``/``set_state_dict`` make the on-disk format parallelism-
    and world-size-agnostic, which is what lets a many-node run be inspected, debugged,
    or resumed at a different world size without a checkpoint format change.

    Both ranks here train under plain DDP (CPU/gloo, no FSDP2), so every rank already
    holds the full, unsharded model/optimizer state -- this test proves the format's
    world-size-agnostic save/load *contract*, not an actual shard-boundary reshard
    across a change in shard count. That arrives with FSDP2/HSDP (Phase 5); this test
    should be revisited then to add an FSDP2-sharded variant.

    Phase 1 launches ``tests/training/_reshard_dcp_worker.py`` under
    ``torch.distributed.run --nproc_per_node=2`` (forced onto CPU via
    ``ACCELERATE_USE_CPU``/``CUDA_VISIBLE_DEVICES=""``, mirroring
    ``test_resume_target_broadcast.py``'s subprocess pattern): 2 ranks train 3 steps and
    commit ``checkpoint-3``. Phase 2 (this process, single rank, also forced onto CPU
    for a hermetic device match) constructs a fresh ``Trainer`` with ``auto_resume=True``
    against the same ``run_dir`` and asserts it restores ``global_step == 3`` and the
    exact ``lm_head.decoder.bias`` values rank 0 saved, then trains 2 more steps to
    ``global_step == 5`` without error.

    Two documented, deliberately out-of-scope wrinkles:

    - **RNG sidecars are per-rank and asymmetric across this direction.** A checkpoint
      saved at world size 2 has ``rng_state_0.pt`` and ``rng_state_1.pt``; a
      world-size-1 resume only ever reads rank 0's sidecar, so this direction (2 -> 1)
      never hits the missing-sidecar error path. The *reverse* direction (a checkpoint
      saved at world size 1, resumed at world size 2) would hard-error for rank 1 --
      ``_restore_rng_sidecar`` raises ``RuntimeError`` unless
      ``OPLM_ALLOW_MISSING_RNG=1`` is set (see
      ``test_load_checkpoint_raises_on_missing_rng_sidecar`` above) -- but exercising
      that direction is out of scope here.
    - **Data-exactness across the world-size change is not asserted.** The data cursor
      (Phase 3) does not exist yet, and ``ShardedProteinDataset`` stripes by world size,
      so the world-size-1 resumed run sees a different data striping than the
      world-size-2 run did -- it does NOT continue from "the same" data position. That
      is fine for what this test asserts (state restoration + training continuing
      without error), but once Phase 3 lands, its data-cursor layout guard will make a
      world-size change across a resume require ``train.resume_data_position=false``
      explicitly. That setting is passed on the resume config below now, ahead of
      Phase 3, so this test does not need updating when the guard lands.
    """
    from oplm.training.trainer import Trainer

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    worker = Path(__file__).with_name("_reshard_dcp_worker.py")

    # The worker imports tests.training.conftest (tiny_train_cfg), so the repo root --
    # not just src/ -- must be on the child's PYTHONPATH; this parent process only
    # inherits src/ from its own launch env (see the ENV GOTCHA note in the task brief).
    repo_root = Path(__file__).resolve().parents[2]
    src_dir = repo_root / "src"
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    child_pythonpath = os.pathsep.join(
        p for p in (str(repo_root), str(src_dir), existing_pythonpath) if p
    )
    env = {
        **os.environ,
        "ACCELERATE_USE_CPU": "true",
        # See test_resume_target_broadcast.py's identical env for why CUDA must be
        # hidden entirely from the child processes: accelerate's wait_for_everyone
        # calls torch.distributed.barrier(device_ids=[local_process_index]) even for
        # a gloo process group, which maps local_process_index onto a CUDA device
        # ordinal and fails once it exceeds the visible GPU count.
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONPATH": child_pythonpath,
    }
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "--rdzv_backend=c10d",
            "--rdzv_endpoint=localhost:0",
            str(worker),
            str(run_dir),
            str(training_parquet),
            str(out_dir),
            str(_RESHARD_WORKER_MAX_STEPS),
        ],
        check=True,
        timeout=300,
        env=env,
    )

    committed = run_dir / f"checkpoint-{_RESHARD_WORKER_MAX_STEPS}"
    assert committed.is_dir()
    assert (committed / "rng_state_0.pt").exists()
    assert (committed / "rng_state_1.pt").exists()

    rank0_state = json.loads((out_dir / "rank0.json").read_text())
    rank1_state = json.loads((out_dir / "rank1.json").read_text())
    assert rank0_state["global_step"] == rank1_state["global_step"] == _RESHARD_WORKER_MAX_STEPS

    reference_weight = torch.load(out_dir / "reference_weight.pt", weights_only=True)

    # Phase 2: single-process resume, forced onto CPU to match the pilot run's device
    # (this box may have a GPU; the point of this test is the world-size change, not
    # an incidental device change stacked on top of it).
    configure_accelerator_device("cpu", monkeypatch)

    resume_max_steps = _RESHARD_WORKER_MAX_STEPS + _RESHARD_RESUME_EXTRA_STEPS
    resume_cfg = tiny_train_cfg(
        run_dir,
        training_parquet,
        max_steps=resume_max_steps,
        save_every=resume_max_steps,
        auto_resume=True,
        log_every=1,
    )
    # Future-proofing for Phase 3 (data cursor): ShardedProteinDataset stripes by
    # world size, so this world-size-2-to-1 resume sees different data striping than
    # the original run -- fine for this test (state restoration is what's asserted,
    # not data exactness), but once the Phase 3 cursor layout guard lands, a
    # world-size change across a resume will require this explicitly. The field
    # exists now (default True, currently unread), so set it ahead of that landing.
    resume_cfg.train.resume_data_position = False

    resumed = Trainer(resume_cfg, callbacks=[])
    assert resumed.global_step == _RESHARD_WORKER_MAX_STEPS

    resumed_unwrapped = resumed.accelerator.unwrap_model(resumed.model)
    resumed_weight = resumed_unwrapped.lm_head.decoder.bias.detach()
    # Bit-exact, not merely close: this is a load-only comparison (no training happens
    # between the checkpoint commit and this assertion) of the same dtype (float32), so
    # DCP's round-trip must reproduce the saved bytes exactly -- torch.allclose's default
    # tolerance would silently paper over a real precision-losing bug in the reshard path.
    assert torch.equal(resumed_weight, reference_weight)

    resumed.train()
    assert resumed.global_step == resume_max_steps


# --- Task 5.1b: dedicated checkpoint process group --------------------------------------


class _FakeAcceleratorForGroup:
    """Stand-in for Accelerate's Accelerator: only ``num_processes`` is read by
    ``build_checkpoint_process_group``.
    """

    def __init__(self, num_processes: int) -> None:
        self.num_processes = num_processes


def test_build_checkpoint_process_group_returns_none_for_single_process() -> None:
    """No process group is built (or even inspected) for a single-process run."""
    from oplm.training.checkpoint import build_checkpoint_process_group

    assert build_checkpoint_process_group(_FakeAcceleratorForGroup(1)) is None


def test_build_checkpoint_process_group_returns_none_without_a_live_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``num_processes > 1`` alone is not enough -- a dead/absent process group is also None.

    Mirrors ``build_upload_group``'s identical defensive check: a caller that reports
    ``num_processes > 1`` without ``torch.distributed`` actually being initialized (not a
    real production shape, but defensive) must not attempt ``dist.new_group``.
    """
    import torch.distributed as dist

    from oplm.training.checkpoint import build_checkpoint_process_group

    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    assert build_checkpoint_process_group(_FakeAcceleratorForGroup(2)) is None


def test_build_checkpoint_process_group_calls_new_group_exactly_once_with_gloo_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multi-process: ``dist.new_group`` is called exactly once, with ``backend="gloo"``.

    ``torch.distributed.new_group`` is itself the collective every rank must reach
    identically (see the function's own docstring); this spies on the real distributed
    module (rather than faking it out entirely) to pin the exact call shape --
    ``backend="gloo"`` (never inherited from whatever the default process group's own
    backend is) and the caller's ``timeout`` passed straight through -- without needing a
    real multi-rank launch (that proof is the 2-rank e2e pilot in
    ``test_e2e_checkpoint_pg.py``).
    """
    from datetime import timedelta

    import torch.distributed as dist

    from oplm.training.checkpoint import build_checkpoint_process_group

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    calls: list[dict[str, object]] = []
    sentinel_group = object()

    def _fake_new_group(**kwargs: object) -> object:
        calls.append(kwargs)
        return sentinel_group

    monkeypatch.setattr(dist, "new_group", _fake_new_group)

    timeout = timedelta(minutes=15)
    result = build_checkpoint_process_group(_FakeAcceleratorForGroup(2), timeout=timeout)

    assert result is sentinel_group
    assert calls == [{"backend": "gloo", "timeout": timeout}]


def test_save_checkpoint_blocking_passes_process_group_to_dcp_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``blocking=True`` threads ``process_group`` straight through to ``dcp.save``.

    Captures the real ``dcp.save`` call's kwargs via monkeypatch (rather than faking out
    the whole save) so this pins the exact plumbing Task 5.1b added to
    ``save_checkpoint`` without re-deriving DCP's own save behavior.
    """
    from oplm.training import checkpoint as checkpoint_module

    cfg = _cfg()
    accelerator, model, optimizer, scheduler = _prepared_model(cfg)
    _take_optimizer_step(cfg, accelerator, model, optimizer, scheduler)

    captured: dict[str, object] = {}
    real_dcp_save = checkpoint_module.dcp.save

    def _spy_save(*args: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return real_dcp_save(*args, **kwargs)

    monkeypatch.setattr(checkpoint_module.dcp, "save", _spy_save)

    sentinel_group = object()
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
        process_group=sentinel_group,  # type: ignore[arg-type]
    )

    assert captured["process_group"] is sentinel_group


def test_save_checkpoint_async_passes_process_group_to_dcp_async_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``blocking=False`` threads ``process_group`` straight through to ``dcp.async_save``.

    Same shape as the blocking test above, but on the async path -- the one Task 5.1b
    actually exists to fix, since ``dcp.async_save``'s background write thread is what
    runs collectives that could otherwise interleave with the training loop's own.
    """
    from oplm.training import checkpoint as checkpoint_module

    cfg = _cfg()
    accelerator, model, optimizer, scheduler = _prepared_model(cfg)
    _take_optimizer_step(cfg, accelerator, model, optimizer, scheduler)

    captured: dict[str, object] = {}
    real_dcp_async_save = checkpoint_module.dcp.async_save

    def _spy_async_save(*args: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return real_dcp_async_save(*args, **kwargs)

    monkeypatch.setattr(checkpoint_module.dcp, "async_save", _spy_async_save)

    sentinel_group = object()
    result = save_checkpoint(
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
        blocking=False,
        process_group=sentinel_group,  # type: ignore[arg-type]
    )

    assert captured["process_group"] is sentinel_group

    # Clean up the still-pending write so this test doesn't leak a background thread
    # holding file handles open past the test's tmp_path teardown.
    assert isinstance(result, PendingSave)
    result.future.result()
