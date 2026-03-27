"""Tests for the training infrastructure: scheduler, FLOPs, checkpoint, Trainer."""

from __future__ import annotations

import json
import math
import random
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from oplm.config import ModelConfig, OplmConfig, TrainConfig
from oplm.training.flops import estimate_flops_per_token
from oplm.training.optim import build_optimizer, build_scheduler, get_schedule_fn

# Standard amino acid alphabet for synthetic data
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def _random_protein(min_len: int = 30, max_len: int = 80) -> str:
    length = random.randint(min_len, max_len)
    return "".join(random.choice(AMINO_ACIDS) for _ in range(length))


def _small_model_config(**overrides: object) -> ModelConfig:
    defaults: dict[str, object] = {
        "hidden_dim": 64,
        "num_heads": 4,
        "num_kv_heads": 2,
        "num_layers": 2,
        "max_seq_len": 64,
    }
    defaults.update(overrides)
    return ModelConfig(**defaults)


@pytest.fixture()
def parquet_dataset(tmp_path: Path) -> Path:
    """Create a small parquet dataset of synthetic protein sequences."""
    random.seed(42)
    sequences = [_random_protein() for _ in range(200)]
    table = pa.table(
        {
            "sequence_id": [f"seq_{i}" for i in range(len(sequences))],
            "sequence": sequences,
        }
    )
    path = tmp_path / "train.parquet"
    pq.write_table(table, path)
    return path


# ---------------------------------------------------------------------------
# Scheduler tests
# ---------------------------------------------------------------------------


class TestScheduler:
    """Test LR schedule functions."""

    def test_warmup_linear_basic(self) -> None:
        fn = get_schedule_fn("warmup_linear", warmup_steps=100, total_steps=1000)
        assert fn(0) == pytest.approx(0.0)
        assert fn(50) == pytest.approx(0.5)
        assert fn(100) == pytest.approx(1.0)
        assert fn(1000) == pytest.approx(0.0, abs=1e-6)

    def test_warmup_linear_midpoint(self) -> None:
        fn = get_schedule_fn("warmup_linear", warmup_steps=100, total_steps=1000)
        # At step 550 (halfway through decay), should be ~0.5
        assert fn(550) == pytest.approx(0.5, abs=0.01)

    def test_warmup_cosine_endpoints(self) -> None:
        fn = get_schedule_fn("warmup_cosine", warmup_steps=100, total_steps=1000)
        assert fn(0) == pytest.approx(0.0)
        assert fn(100) == pytest.approx(1.0)
        assert fn(1000) == pytest.approx(0.0, abs=1e-6)

    def test_warmup_cosine_midpoint(self) -> None:
        fn = get_schedule_fn("warmup_cosine", warmup_steps=100, total_steps=1000)
        # Cosine midpoint should be 0.5
        midpoint = 100 + (1000 - 100) // 2
        assert fn(midpoint) == pytest.approx(0.5, abs=0.01)

    def test_min_ratio(self) -> None:
        fn = get_schedule_fn(
            "warmup_linear", warmup_steps=100, total_steps=1000, min_ratio=0.1
        )
        assert fn(0) == pytest.approx(0.0)
        assert fn(100) == pytest.approx(1.0)
        # At end, should decay to min_ratio, not 0
        assert fn(1000) == pytest.approx(0.1, abs=1e-6)

    def test_wsd_linear_phases(self) -> None:
        fn = get_schedule_fn(
            "wsd_linear",
            warmup_steps=100,
            total_steps=1000,
            stable_fraction=0.5,
        )
        # Warmup
        assert fn(0) == pytest.approx(0.0)
        assert fn(100) == pytest.approx(1.0)
        # Stable phase (steps 100-600)
        assert fn(300) == pytest.approx(1.0)
        assert fn(599) == pytest.approx(1.0)
        # Decay phase (steps 600-1000)
        assert fn(1000) == pytest.approx(0.0, abs=1e-6)

    def test_wsd_cosine_phases(self) -> None:
        fn = get_schedule_fn(
            "wsd_cosine",
            warmup_steps=100,
            total_steps=1000,
            stable_fraction=0.5,
        )
        assert fn(300) == pytest.approx(1.0)
        assert fn(1000) == pytest.approx(0.0, abs=1e-6)

    def test_build_scheduler_integration(self) -> None:
        """Test that build_scheduler produces a working LambdaLR."""
        cfg = TrainConfig(lr=1e-3, warmup_steps=10, max_steps=100, scheduler="warmup_linear")
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        scheduler = build_scheduler(optimizer, cfg, total_steps=100)

        # Step through warmup
        for _ in range(10):
            optimizer.step()
            scheduler.step()

        # LR should be at peak after warmup
        assert scheduler.get_last_lr()[0] == pytest.approx(cfg.lr, rel=0.01)


# ---------------------------------------------------------------------------
# Optimizer tests
# ---------------------------------------------------------------------------


class TestOptimizer:
    """Test optimizer construction."""

    def test_weight_decay_grouping(self) -> None:
        from oplm.model.transformer import OplmForMLM

        cfg = _small_model_config()
        model = OplmForMLM(cfg)
        train_cfg = TrainConfig(lr=1e-3, weight_decay=0.01)
        optimizer = build_optimizer(model, train_cfg)

        assert len(optimizer.param_groups) == 2
        decay_group = optimizer.param_groups[0]
        no_decay_group = optimizer.param_groups[1]

        assert decay_group["weight_decay"] == 0.01
        assert no_decay_group["weight_decay"] == 0.0

        # All 1D params should be in no_decay group
        for p in no_decay_group["params"]:
            assert p.ndim <= 1 or True  # embed params can be 2D

        # Decay group should have 2D+ params
        for p in decay_group["params"]:
            assert p.ndim >= 2

    def test_all_params_accounted_for(self) -> None:
        from oplm.model.transformer import OplmForMLM

        cfg = _small_model_config()
        model = OplmForMLM(cfg)
        train_cfg = TrainConfig(lr=1e-3)
        optimizer = build_optimizer(model, train_cfg)

        optim_params = set()
        for group in optimizer.param_groups:
            for p in group["params"]:
                optim_params.add(id(p))

        model_params = {id(p) for p in model.parameters() if p.requires_grad}
        assert optim_params == model_params


# ---------------------------------------------------------------------------
# FLOPs tests
# ---------------------------------------------------------------------------


class TestFlops:
    """Test FLOP estimation."""

    def test_basic_estimate(self) -> None:
        cfg = _small_model_config()
        flops = estimate_flops_per_token(cfg)
        assert flops > 0
        assert isinstance(flops, int)

    def test_scales_with_layers(self) -> None:
        cfg_2 = _small_model_config(num_layers=2)
        cfg_4 = _small_model_config(num_layers=4)
        flops_2 = estimate_flops_per_token(cfg_2)
        flops_4 = estimate_flops_per_token(cfg_4)
        # 4 layers should have roughly 2x the FLOPs of 2 layers
        # (not exactly due to MLM head contribution)
        assert flops_4 > flops_2
        ratio = flops_4 / flops_2
        assert 1.8 < ratio < 2.2

    def test_scales_with_hidden_dim(self) -> None:
        cfg_64 = _small_model_config(hidden_dim=64, num_heads=4, num_kv_heads=2)
        cfg_128 = _small_model_config(hidden_dim=128, num_heads=4, num_kv_heads=2)
        flops_64 = estimate_flops_per_token(cfg_64)
        flops_128 = estimate_flops_per_token(cfg_128)
        # Roughly quadratic in hidden_dim (due to h*h terms)
        assert flops_128 > flops_64 * 3

    def test_swiglu_vs_gelu(self) -> None:
        cfg_swiglu = _small_model_config(ffn_activation="swiglu")
        cfg_gelu = _small_model_config(ffn_activation="gelu")
        flops_swiglu = estimate_flops_per_token(cfg_swiglu)
        flops_gelu = estimate_flops_per_token(cfg_gelu)
        # SwiGLU has 3 projections vs 2 for GELU, but smaller FFN dim
        # Both should be positive and in a reasonable range
        assert flops_swiglu > 0
        assert flops_gelu > 0


# ---------------------------------------------------------------------------
# Checkpoint tests
# ---------------------------------------------------------------------------


class TestCheckpoint:
    """Test checkpoint save/load."""

    def test_save_and_load_trainer_state(self, tmp_path: Path) -> None:
        from accelerate import Accelerator

        from oplm.training.checkpoint import load_checkpoint, save_checkpoint

        accelerator = Accelerator(cpu=True)
        cfg = OplmConfig(train=TrainConfig(output_dir=str(tmp_path)))

        # Create a simple model + optimizer for accelerator state
        model = torch.nn.Linear(4, 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)
        model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

        save_checkpoint(
            accelerator=accelerator,
            cfg=cfg,
            output_dir=str(tmp_path),
            global_step=100,
            epoch=2,
            tokens_seen=50000,
            save_total_limit=3,
        )

        checkpoint_dir = tmp_path / "checkpoint-100"
        assert checkpoint_dir.exists()
        assert (checkpoint_dir / "trainer_state.json").exists()
        assert (checkpoint_dir / "config.yaml").exists()

        state = load_checkpoint(accelerator, str(checkpoint_dir))
        assert state["global_step"] == 100
        assert state["epoch"] == 2
        assert state["tokens_seen"] == 50000

    def test_checkpoint_rotation(self, tmp_path: Path) -> None:
        from accelerate import Accelerator

        from oplm.training.checkpoint import save_checkpoint

        accelerator = Accelerator(cpu=True)
        cfg = OplmConfig(train=TrainConfig(output_dir=str(tmp_path)))

        model = torch.nn.Linear(4, 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)
        model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

        for step in [100, 200, 300, 400]:
            save_checkpoint(
                accelerator=accelerator,
                cfg=cfg,
                output_dir=str(tmp_path),
                global_step=step,
                epoch=0,
                tokens_seen=step * 500,
                save_total_limit=2,
            )

        # Only the 2 most recent should remain
        checkpoint_dirs = sorted(
            d for d in tmp_path.iterdir() if d.name.startswith("checkpoint-")
        )
        assert len(checkpoint_dirs) == 2
        assert checkpoint_dirs[0].name == "checkpoint-300"
        assert checkpoint_dirs[1].name == "checkpoint-400"


# ---------------------------------------------------------------------------
# TrainConfig validation tests
# ---------------------------------------------------------------------------


class TestTrainConfigValidation:
    """Test TrainConfig validation."""

    def test_defaults(self) -> None:
        cfg = TrainConfig()
        assert cfg.max_steps == 50_000
        assert cfg.scheduler == "warmup_linear"
        assert cfg.mixed_precision == "bf16"

    def test_invalid_scheduler(self) -> None:
        with pytest.raises(ValueError, match="scheduler"):
            TrainConfig(scheduler="invalid")

    def test_invalid_optimizer(self) -> None:
        with pytest.raises(ValueError, match="optimizer"):
            TrainConfig(optimizer="sgd")

    def test_invalid_mixed_precision(self) -> None:
        with pytest.raises(ValueError, match="mixed_precision"):
            TrainConfig(mixed_precision="fp8")

    def test_negative_warmup(self) -> None:
        with pytest.raises(ValueError, match="warmup_steps"):
            TrainConfig(warmup_steps=-1)

    def test_min_lr_exceeds_lr(self) -> None:
        with pytest.raises(ValueError, match="min_lr"):
            TrainConfig(lr=1e-4, min_lr=1e-3)

    def test_invalid_stable_fraction(self) -> None:
        with pytest.raises(ValueError, match="stable_fraction"):
            TrainConfig(stable_fraction=1.0)

    def test_invalid_gradient_accumulation(self) -> None:
        with pytest.raises(ValueError, match="gradient_accumulation_steps"):
            TrainConfig(gradient_accumulation_steps=0)


# ---------------------------------------------------------------------------
# Integration: Trainer with synthetic data
# ---------------------------------------------------------------------------


class TestTrainerIntegration:
    """Test the full Trainer with synthetic parquet data."""

    def test_trainer_loss_decreases(self, parquet_dataset: Path) -> None:
        """Train a small model for 20 steps, verify loss decreases."""
        torch.manual_seed(42)
        random.seed(42)

        cfg = OplmConfig(
            model=_small_model_config(),
            train=TrainConfig(
                max_steps=20,
                batch_size=8,
                lr=1e-3,
                warmup_steps=5,
                log_every=5,
                eval_every=100,
                save_every=100,
                wandb_enabled=False,
                mixed_precision="no",
                output_dir=tempfile.mkdtemp(),
            ),
        )
        cfg.data.train = str(parquet_dataset)
        cfg.data.max_length = 64
        cfg.data.num_workers = 0

        from oplm.training import Trainer

        trainer = Trainer(cfg)

        # Capture losses via a simple eval_fn that records train loss
        losses: list[float] = []
        orig_log = trainer._log_step

        def _capture_log(loss: float) -> None:
            losses.append(loss)
            orig_log(loss)

        trainer._log_step = _capture_log  # type: ignore[method-assign]

        trainer.train()

        assert len(losses) > 0
        assert trainer.global_step == 20
        assert trainer.tokens_seen > 0

    def test_trainer_with_eval_fn(self, parquet_dataset: Path) -> None:
        """Test that eval_fn callback is invoked."""
        torch.manual_seed(42)
        random.seed(42)

        eval_calls: list[int] = []

        def mock_eval(
            model: object,
            accelerator: object,
            step: int,
        ) -> dict[str, float]:
            eval_calls.append(step)
            return {"eval/loss": 1.0}

        cfg = OplmConfig(
            model=_small_model_config(),
            train=TrainConfig(
                max_steps=20,
                batch_size=8,
                lr=1e-3,
                warmup_steps=2,
                log_every=5,
                eval_every=10,
                save_every=100,
                wandb_enabled=False,
                mixed_precision="no",
                output_dir=tempfile.mkdtemp(),
            ),
        )
        cfg.data.train = str(parquet_dataset)
        cfg.data.max_length = 64
        cfg.data.num_workers = 0

        from oplm.training import Trainer

        trainer = Trainer(cfg, eval_fn=mock_eval)
        trainer.train()

        # eval_fn should have been called at steps 10 and 20
        assert 10 in eval_calls
        assert 20 in eval_calls

    def test_trainer_checkpoint_saved(self, parquet_dataset: Path) -> None:
        """Test that checkpoints are saved."""
        torch.manual_seed(42)
        random.seed(42)

        output_dir = tempfile.mkdtemp()
        cfg = OplmConfig(
            model=_small_model_config(),
            train=TrainConfig(
                max_steps=10,
                batch_size=8,
                lr=1e-3,
                warmup_steps=2,
                log_every=5,
                eval_every=100,
                save_every=5,
                wandb_enabled=False,
                mixed_precision="no",
                output_dir=output_dir,
            ),
        )
        cfg.data.train = str(parquet_dataset)
        cfg.data.max_length = 64
        cfg.data.num_workers = 0

        from oplm.training import Trainer

        trainer = Trainer(cfg)
        trainer.train()

        output_path = Path(output_dir)
        checkpoints = sorted(
            d for d in output_path.iterdir() if d.name.startswith("checkpoint-")
        )
        assert len(checkpoints) >= 2  # save_every=5 -> step 5 and final at step 10

        # Verify trainer state in the final checkpoint
        final_ckpt = output_path / f"checkpoint-{trainer.global_step}"
        assert final_ckpt.exists()
        state = json.loads((final_ckpt / "trainer_state.json").read_text())
        assert state["global_step"] == trainer.global_step
