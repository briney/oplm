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

import logging
from typing import TYPE_CHECKING

import pytest
import torch

from oplm.config import DataConfig, OplmConfig, TrainConfig
from oplm.model import OplmConfig as OplmModelConfig
from oplm.model import OplmForMaskedLM
from oplm.training.checkpoint import load_checkpoint, save_checkpoint
from oplm.training.optim import build_optimizers, build_schedulers

if TYPE_CHECKING:
    from pathlib import Path

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
