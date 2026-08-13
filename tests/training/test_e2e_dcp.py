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
    load_checkpoint(
        fresh_accelerator,
        fresh_model,
        [fresh_optimizer],
        [fresh_scheduler],
        str(tmp_path / "checkpoint-10"),
    )

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
    load_checkpoint(accelerator, fresh_model, [fresh_optimizer], [fresh_scheduler], str(committed))
    assert torch.allclose(fresh_model.lm_head.decoder.bias.detach(), original_weight)
