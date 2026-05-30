"""Checkpoint save/load tests (Phase 6: config serialization + HF export).

Exercises the real :class:`accelerate.Accelerator` path: ``save_checkpoint`` writes
the resumable Accelerate state, ``trainer_state.json``, a re-loadable
``config.yaml``, and a ``from_pretrained``-loadable HF export under ``hf/``;
``load_checkpoint`` restores state and returns the metadata; rotation keeps at most
``save_total_limit`` checkpoints. Marked slow — it spins up an Accelerator and does
real safetensors IO.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from oplm.config import DataConfig, OplmConfig, TrainConfig
from oplm.model import OplmConfig as OplmModelConfig
from oplm.model import OplmForMaskedLM
from oplm.training.checkpoint import load_checkpoint, save_checkpoint

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


def _prepared_model(cfg: OplmConfig) -> tuple[Accelerator, OplmForMaskedLM]:
    """Build and prepare a tiny model on CPU so ``save_state`` has something to save."""
    from accelerate import Accelerator

    accelerator = Accelerator(cpu=True, mixed_precision="no")
    model = OplmForMaskedLM(cfg.model)
    model = accelerator.prepare(model)
    return accelerator, model


def test_hf_export_round_trips(tmp_path: Path) -> None:
    """``checkpoint-N/hf`` reloads via ``from_pretrained`` with matching weights."""
    cfg = _cfg()
    accelerator, model = _prepared_model(cfg)
    original_bias = accelerator.unwrap_model(model).lm_head.decoder.bias.detach().clone()

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

    hf_dir = tmp_path / "checkpoint-10" / "hf"
    assert (hf_dir / "config.json").exists()
    assert (hf_dir / "model.safetensors").exists()

    reloaded = OplmForMaskedLM.from_pretrained(str(hf_dir))
    assert torch.allclose(reloaded.lm_head.decoder.bias.detach(), original_bias)


def test_config_yaml_is_reloadable(tmp_path: Path) -> None:
    """The written ``config.yaml`` carries model/train/data and re-loads via load_config."""
    from oplm.config import load_config

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

    config_path = tmp_path / "checkpoint-10" / "config.yaml"
    assert config_path.exists()

    reloaded = load_config(["--config", str(config_path)])
    assert reloaded.model.hidden_size == cfg.model.hidden_size
    assert reloaded.model.num_hidden_layers == cfg.model.num_hidden_layers


def test_load_checkpoint_restores_state(tmp_path: Path) -> None:
    """``load_checkpoint`` returns the trainer-state metadata and restores Accelerate state."""
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

    state = load_checkpoint(accelerator, str(tmp_path / "checkpoint-10"))
    assert state["global_step"] == 10
    assert state["epoch"] == 1
    assert state["samples_seen"] == 40
    assert state["tokens_seen"] == 400


def test_rotation_keeps_save_total_limit(tmp_path: Path) -> None:
    """Saving beyond ``save_total_limit`` deletes the oldest checkpoints."""
    cfg = _cfg()
    accelerator, model = _prepared_model(cfg)
    for step in (10, 20, 30):
        save_checkpoint(
            accelerator=accelerator,
            model=model,
            cfg=cfg,
            output_dir=str(tmp_path),
            global_step=step,
            epoch=1,
            samples_seen=4 * step,
            tokens_seen=40 * step,
            save_total_limit=2,
        )

    remaining = sorted(d.name for d in tmp_path.iterdir() if d.name.startswith("checkpoint-"))
    assert remaining == ["checkpoint-20", "checkpoint-30"]  # oldest (10) rotated out
