"""Checkpoint save/load tests — PyTorch Distributed Checkpoint (DCP) path.

The native stack replaces Accelerate's ``save_state`` with
``torch.distributed.checkpoint``: ``save_checkpoint`` writes the sharded model +
primary-optimizer state, ``trainer_state.json``, a re-loadable ``config.yaml``, and
a ``from_pretrained``-loadable HF export under ``hf/``; ``load_checkpoint`` restores
the model and optimizer in place and returns the metadata; rotation keeps at most
``save_total_limit`` checkpoints. Every entry point needs a live process group, so
these tests form a single-rank Gloo group on CPU. Marked slow — real distributed
checkpoint IO plus safetensors export.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
import torch

from oplm.config import DataConfig, OplmConfig, TrainConfig
from oplm.model import OplmConfig as OplmModelConfig
from oplm.model import OplmForMaskedLM
from oplm.training.checkpoint import load_checkpoint, save_checkpoint

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

pytestmark = pytest.mark.slow


@pytest.fixture
def single_rank_pg() -> Iterator[None]:
    """Form a single-rank Gloo process group for the DCP collectives.

    ``save_checkpoint`` / ``load_checkpoint`` call ``dist.get_rank``, ``dist.barrier``,
    and the DCP state-dict collectives, all of which require an initialized group.
    Teardown is also guaranteed by the autouse ``_cleanup_distributed`` fixture.
    """
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29577")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


def _cfg() -> OplmConfig:
    """Tiny root config whose ``model`` is the HF ``OplmConfig``."""
    return OplmConfig(
        model=OplmModelConfig(
            hidden_size=32,
            num_attention_heads=4,
            num_hidden_layers=2,
            max_position_embeddings=64,
        ),
        train=TrainConfig(wandb_enabled=False),
        data=DataConfig(num_workers=0, pin_memory=False),
    )


def _model_and_optimizer(cfg: OplmConfig) -> tuple[OplmForMaskedLM, torch.optim.Optimizer]:
    """Build a tiny CPU model and an AdamW optimizer over its parameters."""
    model = OplmForMaskedLM(cfg.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, optimizer


def _take_one_step(model: OplmForMaskedLM, optimizer: torch.optim.Optimizer) -> None:
    """Run one forward/backward/step so weights move off init and optimizer state exists."""
    input_ids = torch.randint(0, model.config.vocab_size, (2, 8))
    labels = torch.randint(0, model.config.vocab_size, (2, 8))
    loss = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), labels=labels)[
        "loss"
    ]
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


def _state_dicts_match(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    """True iff two state dicts have the same keys and all-close tensors."""
    return a.keys() == b.keys() and all(torch.allclose(a[k], b[k]) for k in a)


def test_hf_export_round_trips(tmp_path: Path, single_rank_pg: None) -> None:
    """``checkpoint-N/hf`` reloads via ``from_pretrained`` with matching weights."""
    cfg = _cfg()
    model, optimizer = _model_and_optimizer(cfg)
    original_bias = model.lm_head.decoder.bias.detach().clone()

    save_checkpoint(
        model=model,
        optimizer=optimizer,
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


def test_config_yaml_is_reloadable(tmp_path: Path, single_rank_pg: None) -> None:
    """The written ``config.yaml`` carries model/train/data and re-loads via load_config."""
    from oplm.config import load_config

    cfg = _cfg()
    model, optimizer = _model_and_optimizer(cfg)
    save_checkpoint(
        model=model,
        optimizer=optimizer,
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


def test_load_checkpoint_round_trips_model_and_optimizer(
    tmp_path: Path, single_rank_pg: None
) -> None:
    """``load_checkpoint`` restores model weights + optimizer state and returns metadata."""
    cfg = _cfg()
    model, optimizer = _model_and_optimizer(cfg)
    _take_one_step(model, optimizer)  # perturb weights off init; populate AdamW state
    saved_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        cfg=cfg,
        output_dir=str(tmp_path),
        global_step=10,
        epoch=1,
        samples_seen=40,
        tokens_seen=400,
    )

    # Fresh model + optimizer with independent init and no optimizer state.
    fresh_model, fresh_optimizer = _model_and_optimizer(cfg)
    assert not _state_dicts_match(fresh_model.state_dict(), saved_sd)  # independent inits differ
    assert len(fresh_optimizer.state) == 0

    state = load_checkpoint(fresh_model, fresh_optimizer, str(tmp_path / "checkpoint-10"))

    # Metadata round-trips.
    assert state["global_step"] == 10
    assert state["epoch"] == 1
    assert state["samples_seen"] == 40
    assert state["tokens_seen"] == 400
    # Model weights now match the saved model exactly.
    assert _state_dicts_match(fresh_model.state_dict(), saved_sd)
    # Optimizer state was restored (AdamW exp_avg / exp_avg_sq per parameter).
    assert len(fresh_optimizer.state) > 0


def test_rotation_keeps_save_total_limit(tmp_path: Path, single_rank_pg: None) -> None:
    """Saving beyond ``save_total_limit`` deletes the oldest checkpoints."""
    cfg = _cfg()
    model, optimizer = _model_and_optimizer(cfg)
    for step in (10, 20, 30):
        save_checkpoint(
            model=model,
            optimizer=optimizer,
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
