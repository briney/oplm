"""End-to-end integration tests: real protein data, full Trainer pipeline.

Tests the complete training workflow with real sequences from the test fixtures
parquet shard. Covers CPU training, GPU training with bf16 mixed precision,
and various model feature combinations.
"""

from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING

import pytest
import torch

from oplm.config import DataConfig, ModelConfig, OplmConfig, TrainConfig
from oplm.data.tokenizer import ProteinTokenizer
from oplm.model.transformer import OplmForMLM

if TYPE_CHECKING:
    from pathlib import Path

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


# ---------------------------------------------------------------------------
# E2E: training with eval integration
# ---------------------------------------------------------------------------


class TestE2ETrainingWithEval:
    """Verify eval harness fires during training and produces metrics."""

    def test_training_with_sequence_eval(self, training_parquet: Path) -> None:
        """Train with sequence eval configured, verify eval metrics are logged."""
        from oplm.training import Trainer

        cfg = OplmConfig(
            model=_small_config(),
            train=TrainConfig(
                max_steps=20,
                batch_size=8,
                lr=1e-3,
                warmup_steps=5,
                log_every=5,
                eval_every=10,
                save_every=100,
                wandb_enabled=False,
                mixed_precision="no",
                output_dir=tempfile.mkdtemp(),
            ),
            data=DataConfig(
                train=str(training_parquet),
                max_length=128,
                num_workers=0,
                eval={
                    "test": {
                        "path": str(training_parquet),
                        "type": "sequence",
                    }
                },
            ),
        )

        trainer = Trainer(cfg)
        assert trainer.evaluator is not None
        assert trainer.evaluator.has_tasks

        # Capture eval metrics via accelerator.log
        logged_metrics: list[dict[str, float]] = []
        orig_log = trainer.accelerator.log

        def _capture_log(values: dict[str, float], **kwargs: object) -> None:
            logged_metrics.append(values)
            orig_log(values, **kwargs)

        trainer.accelerator.log = _capture_log
        trainer.train()

        # Eval should have fired at steps 10 and 20
        assert len(logged_metrics) >= 2
        # Check that eval metrics are present
        combined = {}
        for m in logged_metrics:
            combined.update(m)

        assert "eval/test/loss" in combined
        assert "eval/test/accuracy" in combined
        assert "eval/test/perplexity" in combined
        assert combined["eval/test/loss"] > 0.0
        assert 0.0 <= combined["eval/test/accuracy"] <= 1.0
        assert combined["eval/test/perplexity"] > 1.0

    def test_training_with_eval_custom_metrics(self, training_parquet: Path) -> None:
        """Only requested metrics should appear in logged output."""
        from oplm.training import Trainer

        cfg = OplmConfig(
            model=_small_config(),
            train=TrainConfig(
                max_steps=10,
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
            data=DataConfig(
                train=str(training_parquet),
                max_length=128,
                num_workers=0,
                eval={
                    "test": {
                        "path": str(training_parquet),
                        "type": "sequence",
                        "metrics": ["loss"],
                    }
                },
            ),
        )

        trainer = Trainer(cfg)
        logged_metrics: list[dict[str, float]] = []
        orig_log = trainer.accelerator.log

        def _capture_log(values: dict[str, float], **kwargs: object) -> None:
            logged_metrics.append(values)
            orig_log(values, **kwargs)

        trainer.accelerator.log = _capture_log
        trainer.train()

        assert len(logged_metrics) >= 1
        combined = {}
        for m in logged_metrics:
            combined.update(m)

        assert "eval/test/loss" in combined
        assert "eval/test/accuracy" not in combined
        assert "eval/test/perplexity" not in combined

    def test_training_multi_dataset_scheduling(self, training_parquet: Path) -> None:
        """Multiple eval datasets with different schedules fire at correct steps."""
        from oplm.training import Trainer

        cfg = OplmConfig(
            model=_small_config(),
            train=TrainConfig(
                max_steps=20,
                batch_size=8,
                lr=1e-3,
                warmup_steps=5,
                log_every=5,
                eval_every=10,
                save_every=100,
                wandb_enabled=False,
                mixed_precision="no",
                output_dir=tempfile.mkdtemp(),
            ),
            data=DataConfig(
                train=str(training_parquet),
                max_length=128,
                num_workers=0,
                eval={
                    "fast": {
                        "path": str(training_parquet),
                        "type": "sequence",
                        "eval_every": 10,
                    },
                    "slow": {
                        "path": str(training_parquet),
                        "type": "sequence",
                        "eval_every": 20,
                    },
                },
            ),
        )

        trainer = Trainer(cfg)
        assert trainer.evaluator is not None
        assert len(trainer.evaluator.tasks) == 2

        # Track which step each log call occurs at
        logged_at_step: list[tuple[int, dict[str, float]]] = []
        orig_log = trainer.accelerator.log

        def _capture_log(values: dict[str, float], step: int = 0, **kwargs: object) -> None:
            logged_at_step.append((step or trainer.global_step, values))
            orig_log(values, step=step, **kwargs)

        trainer.accelerator.log = _capture_log
        trainer.train()

        # Step 10: fast fires, slow doesn't
        step10_metrics = {}
        for step, metrics in logged_at_step:
            if step == 10:
                step10_metrics.update(metrics)

        assert "eval/fast/loss" in step10_metrics
        assert "eval/slow/loss" not in step10_metrics

        # Step 20: both fire
        step20_metrics = {}
        for step, metrics in logged_at_step:
            if step == 20:
                step20_metrics.update(metrics)

        assert "eval/fast/loss" in step20_metrics
        assert "eval/slow/loss" in step20_metrics

    def test_no_eval_config_no_evaluator(self, training_parquet: Path) -> None:
        """When no eval is configured, evaluator should be None."""
        from oplm.training import Trainer

        cfg = OplmConfig(
            model=_small_config(),
            train=TrainConfig(
                max_steps=5,
                batch_size=8,
                lr=1e-3,
                warmup_steps=2,
                log_every=5,
                eval_every=100,
                save_every=100,
                wandb_enabled=False,
                mixed_precision="no",
                output_dir=tempfile.mkdtemp(),
            ),
        )
        cfg.data.train = str(training_parquet)
        cfg.data.max_length = 128
        cfg.data.num_workers = 0

        trainer = Trainer(cfg)
        assert trainer.evaluator is None
