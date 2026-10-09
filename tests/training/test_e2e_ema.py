"""EMA through the real checkpoint layer and Trainer: sidecar round-trip, hf_ema/ export,
optimizer-step counting under accumulation, resume, and the pre-EMA-checkpoint case."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import pytest
import torch

from oplm.config import DataConfig, OplmConfig, TrainConfig
from oplm.model import OplmConfig as OplmModelConfig
from oplm.model import OplmForMaskedLM
from oplm.training.checkpoint import load_checkpoint, save_checkpoint
from oplm.training.ema import build_ema
from oplm.training.initialization import resolve_initialization_source
from tests.training.conftest import tiny_train_cfg

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.slow


def _cfg(decay: float) -> OplmConfig:
    return OplmConfig(
        model=OplmModelConfig(
            hidden_size=32, num_attention_heads=4, num_hidden_layers=2, max_position_embeddings=64
        ),
        train=TrainConfig(wandb_enabled=False, mixed_precision="no", ema_decay=decay),
        data=DataConfig(num_workers=0, pin_memory=False),
    )


def _prepared(cfg: OplmConfig) -> tuple[Any, Any, Any, Any, Any]:
    """CPU accelerator, prepared tiny model + AdamW + LambdaLR, and a fresh EMA tracker."""
    from accelerate import Accelerator

    accelerator = Accelerator(cpu=True, mixed_precision="no")
    model = OplmForMaskedLM(cfg.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    model, optimizer = accelerator.prepare(model, optimizer)
    ema = build_ema(accelerator.unwrap_model(model), cfg.train.ema_decay)
    return accelerator, model, optimizer, scheduler, ema


def test_ema_sidecar_and_hf_ema_round_trip(tmp_path: Path) -> None:
    cfg = _cfg(0.5)
    accelerator, model, optimizer, scheduler, ema = _prepared(cfg)
    inputs = torch.randint(0, cfg.model.vocab_size, (2, 8))
    for _ in range(3):
        loss = model(input_ids=inputs, labels=inputs).loss
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        ema.update_parameters(accelerator.unwrap_model(model))
    assert int(ema.n_averaged) == 3

    save_checkpoint(
        accelerator=accelerator,
        model=model,
        optimizers=[optimizer],
        schedulers=[scheduler],
        cfg=cfg,
        output_dir=str(tmp_path),
        global_step=3,
        epoch=0,
        samples_seen=6,
        tokens_seen=48,
        ema=ema,
    )
    committed = tmp_path / "checkpoint-3"
    assert (committed / "ema.pt").is_file()
    assert (committed / "hf_ema" / "config.json").is_file()
    assert (committed / "hf" / "config.json").is_file()

    # hf_ema/ carries the EMA weights (not the live ones) and is a valid init_from target.
    reloaded = OplmForMaskedLM.from_pretrained(committed / "hf_ema", local_files_only=True)
    ema_state = ema.module.state_dict()
    live_state = accelerator.unwrap_model(model).state_dict()
    for name, tensor in reloaded.state_dict().items():
        assert torch.equal(tensor, ema_state[name]), name
    assert any(not torch.equal(ema_state[n], live_state[n]) for n in live_state)
    assert (
        resolve_initialization_source(str(committed / "hf_ema")) == (committed / "hf_ema").resolve()
    )

    # A fresh tracker restores the update count and every tensor.
    fresh_accelerator, fresh_model, fresh_optimizer, fresh_scheduler, fresh_ema = _prepared(cfg)
    load_checkpoint(
        fresh_accelerator,
        fresh_model,
        [fresh_optimizer],
        [fresh_scheduler],
        str(committed),
        cfg,
        ema=fresh_ema,
    )
    assert int(fresh_ema.n_averaged) == 3
    for name, tensor in fresh_ema.module.state_dict().items():
        assert torch.equal(tensor, ema_state[name]), name


def test_ema_counts_optimizer_steps_and_survives_resume(
    tmp_path: Path, training_parquet: Path
) -> None:
    from oplm.training.trainer import Trainer

    common = dict(
        batch_size=4, gradient_accumulation_steps=2, log_every=1, warmup_steps=0, ema_decay=0.5
    )
    cfg1 = tiny_train_cfg(tmp_path, training_parquet, max_steps=4, save_every=4, **common)
    first = Trainer(cfg1)
    first.train()
    assert first._ema is not None
    assert int(first._ema.n_averaged) == 4  # optimizer steps, not the 8 micro-batches

    ckpt = tmp_path / "checkpoint-4"
    state = json.loads((ckpt / "trainer_state.json").read_text())
    assert state["ema"] == {"decay": 0.5, "sidecar": "ema.pt", "hf_dir": "hf_ema", "n_averaged": 4}
    assert (ckpt / "ema.pt").is_file()
    assert (ckpt / "hf_ema" / "model.safetensors").is_file()
    saved = torch.load(ckpt / "ema.pt", map_location="cpu", weights_only=True)

    cfg2 = tiny_train_cfg(
        tmp_path, training_parquet, max_steps=8, save_every=8, resume_from=str(ckpt), **common
    )
    resumed = Trainer(cfg2)
    assert resumed._ema is not None
    assert int(resumed._ema.n_averaged) == 4  # restored, not restarted
    for name, tensor in resumed._ema.state_dict().items():
        # The tracker lives on the training device (cuda:0 on a GPU box); the sidecar
        # was loaded to CPU above, and torch.equal refuses mixed devices.
        assert torch.equal(tensor.detach().cpu(), saved[name]), name

    resumed.train()
    assert int(resumed._ema.n_averaged) == 8 == resumed.global_step
    live = resumed._unwrapped_model.state_dict()
    for name, tensor in resumed._ema.module.state_dict().items():
        assert torch.isfinite(tensor).all(), name
    assert any(not torch.equal(resumed._ema.module.state_dict()[n], live[n]) for n in live)


def test_resume_into_ema_from_checkpoint_without_ema_warns(
    tmp_path: Path, training_parquet: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from oplm.training.trainer import Trainer

    cfg1 = tiny_train_cfg(tmp_path, training_parquet, max_steps=2, save_every=2, batch_size=4)
    Trainer(cfg1).train()
    ckpt = tmp_path / "checkpoint-2"
    assert not (ckpt / "ema.pt").exists()
    assert not (ckpt / "hf_ema").exists()

    cfg2 = tiny_train_cfg(
        tmp_path,
        training_parquet,
        max_steps=4,
        save_every=4,
        batch_size=4,
        resume_from=str(ckpt),
        ema_decay=0.5,
    )
    with caplog.at_level(logging.WARNING):
        resumed = Trainer(cfg2)
    assert "ema.pt" in caplog.text
    assert resumed._ema is not None
    assert int(resumed._ema.n_averaged) == 0

    resumed.train()
    assert int(resumed._ema.n_averaged) == 2
    assert (tmp_path / "checkpoint-4" / "ema.pt").is_file()
