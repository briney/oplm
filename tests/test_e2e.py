"""End-to-end integration tests: real protein data, full Trainer pipeline.

Tests the complete training workflow with real sequences from the test fixtures
parquet shard. Covers CPU training, GPU training with bf16 mixed precision,
and various model feature combinations.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch

from oplm.config import ModelConfig, OplmConfig, TrainConfig
from oplm.data.tokenizer import ProteinTokenizer
from oplm.model.transformer import OplmForMLM

if TYPE_CHECKING:
    from oplm.training import Trainer


def _small_config(**overrides: object) -> ModelConfig:
    """Create a minimal config for fast e2e testing."""
    defaults: dict[str, object] = {
        "hidden_dim": 64,
        "num_heads": 4,
        "num_kv_heads": 2,
        "num_layers": 2,
        "max_seq_len": 128,
    }
    defaults.update(overrides)
    return ModelConfig(**defaults)


def _run_training(
    training_parquet: Path,
    model_overrides: dict[str, object] | None = None,
    *,
    mixed_precision: str = "no",
    batch_size: int = 8,
    max_steps: int = 20,
) -> tuple[Trainer, list[float]]:
    """Run a short training loop with real data and return trainer + captured losses."""
    from oplm.training import Trainer

    cfg = OplmConfig(
        model=_small_config(**(model_overrides or {})),
        train=TrainConfig(
            max_steps=max_steps,
            batch_size=batch_size,
            lr=1e-3,
            warmup_steps=5,
            log_every=5,
            eval_every=100,
            save_every=100,
            wandb_enabled=False,
            mixed_precision=mixed_precision,
            output_dir=tempfile.mkdtemp(),
        ),
    )
    cfg.data.train = str(training_parquet)
    cfg.data.max_length = 128
    cfg.data.num_workers = 0

    trainer = Trainer(cfg)

    losses: list[float] = []
    orig_log = trainer._log_step

    def _capture_log(loss: float) -> None:
        losses.append(loss)
        orig_log(loss)

    trainer._log_step = _capture_log  # type: ignore[method-assign]
    trainer.train()

    return trainer, losses


# ---------------------------------------------------------------------------
# E2E training: real data through the full Trainer pipeline
# ---------------------------------------------------------------------------


class TestE2ETraining:
    """Full pipeline: real protein data -> DataLoader -> Trainer -> loss decreases."""

    def test_loss_decreases(self, training_parquet: Path) -> None:
        """Train a small model on real sequences, verify loss drops."""
        trainer, losses = _run_training(training_parquet)

        assert len(losses) >= 4
        assert trainer.global_step == 20
        assert trainer.tokens_seen > 0
        avg_first = sum(losses[:2]) / 2
        avg_last = sum(losses[-2:]) / 2
        assert avg_last < avg_first, (
            f"Loss did not decrease: first 2 avg={avg_first:.4f}, last 2 avg={avg_last:.4f}"
        )

    def test_loss_decreases_with_features(self, training_parquet: Path) -> None:
        """Train with value residuals, convolutions, and value embeddings."""
        _, losses = _run_training(
            training_parquet,
            model_overrides={
                "value_residual": True,
                "conv_positions": "AC",
                "num_value_embeds": 1,
                "qk_norm": True,
            },
        )

        assert len(losses) >= 4
        avg_first = sum(losses[:2]) / 2
        avg_last = sum(losses[-2:]) / 2
        assert avg_last < avg_first, (
            f"Loss did not decrease: first 2 avg={avg_first:.4f}, last 2 avg={avg_last:.4f}"
        )

    def test_loss_decreases_with_attn_residual(self, training_parquet: Path) -> None:
        """Train with attention residuals enabled."""
        _, losses = _run_training(
            training_parquet,
            model_overrides={"attn_residual": True, "attn_residual_block_size": 1},
        )

        assert len(losses) >= 4
        avg_first = sum(losses[:2]) / 2
        avg_last = sum(losses[-2:]) / 2
        assert avg_last < avg_first, (
            f"Loss did not decrease: first 2 avg={avg_first:.4f}, last 2 avg={avg_last:.4f}"
        )

    @pytest.mark.slow
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_loss_decreases_cuda_bf16(self, training_parquet: Path) -> None:
        """Train on GPU with bf16 mixed precision."""
        trainer, losses = _run_training(
            training_parquet,
            mixed_precision="bf16",
            batch_size=16,
        )

        assert len(losses) >= 4
        assert trainer.global_step == 20
        avg_first = sum(losses[:2]) / 2
        avg_last = sum(losses[-2:]) / 2
        assert avg_last < avg_first, (
            f"Loss did not decrease: first 2 avg={avg_first:.4f}, last 2 avg={avg_last:.4f}"
        )

    @pytest.mark.slow
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_loss_decreases_cuda_bf16_with_features(self, training_parquet: Path) -> None:
        """Train on GPU with bf16 and all model features enabled."""
        _, losses = _run_training(
            training_parquet,
            model_overrides={
                "value_residual": True,
                "conv_positions": "AC",
                "num_value_embeds": 1,
                "qk_norm": True,
                "attn_residual": True,
                "attn_residual_block_size": 1,
            },
            mixed_precision="bf16",
            batch_size=16,
        )

        assert len(losses) >= 4
        avg_first = sum(losses[:2]) / 2
        avg_last = sum(losses[-2:]) / 2
        assert avg_last < avg_first, (
            f"Loss did not decrease: first 2 avg={avg_first:.4f}, last 2 avg={avg_last:.4f}"
        )


# ---------------------------------------------------------------------------
# E2E: inference mode
# ---------------------------------------------------------------------------


class TestE2EInference:
    """Verify model works in eval mode without errors."""

    def test_eval_forward(self) -> None:
        cfg = _small_config()
        model = OplmForMLM(cfg)
        tokenizer = ProteinTokenizer()
        model.eval()

        sequences = ["MKWVTFISLLLLFSSAYS", "ACDEFGHIKLMNPQ"]
        batch = tokenizer.batch_encode(sequences)

        with torch.no_grad():
            result = model(input_ids=batch["input_ids"])

        assert result["logits"].shape[0] == 2
        assert result["logits"].shape[2] == cfg.vocab_size
        assert result["loss"] is None

    def test_eval_with_attention_weights(self) -> None:
        cfg = _small_config()
        model = OplmForMLM(cfg)
        tokenizer = ProteinTokenizer()
        model.eval()

        batch = tokenizer.batch_encode(["ACDEFGHIKLMNPQ"])
        with torch.no_grad():
            result = model(input_ids=batch["input_ids"], need_weights=True)

        assert "attention_weights" in result
        assert len(result["attention_weights"]) == cfg.num_layers


# ---------------------------------------------------------------------------
# E2E: torch.compile smoke test
# ---------------------------------------------------------------------------


class TestE2ECompile:
    """Verify model compiles without graph breaks (standard feature set)."""

    @pytest.mark.slow
    def test_torch_compile(self) -> None:
        cfg = _small_config()
        model = OplmForMLM(cfg)
        model.eval()

        compiled = torch.compile(model, fullgraph=True)
        input_ids = torch.randint(0, cfg.vocab_size, (1, 16))

        with torch.no_grad():
            result = compiled(input_ids)

        assert result["logits"].shape == (1, 16, cfg.vocab_size)
