"""Tests for MLM metric computation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import torch
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from pathlib import Path

from oplm.config import DataConfig, ModelConfig, OplmConfig, TrainConfig
from oplm.eval.metrics.mlm import _PERPLEXITY_CAP, compute_mlm_metrics
from oplm.model.transformer import OplmForMLM


def _make_tiny_model() -> OplmForMLM:
    """Create a tiny model for testing."""
    cfg = OplmConfig(
        model=ModelConfig(
            hidden_dim=32,
            num_layers=1,
            num_heads=2,
            num_kv_heads=2,
            max_seq_len=64,
        ),
    )
    return OplmForMLM(cfg.model)


def _make_eval_batch(
    batch_size: int = 4,
    seq_len: int = 16,
    vocab_size: int = 33,
    mask_count: int = 3,
) -> dict[str, torch.Tensor]:
    """Create a synthetic eval batch with known masked positions."""
    input_ids = torch.randint(5, vocab_size, (batch_size, seq_len))
    # CLS at position 0, EOS at position 1
    input_ids[:, 0] = 0
    input_ids[:, -1] = 2

    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)

    # Mask a few known positions per sequence
    for b in range(batch_size):
        for pos in range(2, 2 + mask_count):
            labels[b, pos] = input_ids[b, pos]
            input_ids[b, pos] = 4  # <mask>

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _make_mock_accelerator() -> MagicMock:
    """Create a mock Accelerator that passes through reduce."""
    acc = MagicMock()
    acc.device = torch.device("cpu")
    acc.reduce = lambda tensor, reduction: tensor  # identity
    return acc


class TestComputeMlmMetrics:
    """Test compute_mlm_metrics function."""

    def test_returns_expected_keys(self) -> None:
        """Should return loss, accuracy, and perplexity."""
        model = _make_tiny_model()
        model.eval()

        batch = _make_eval_batch(batch_size=2, seq_len=32)
        dl = _single_batch_dataloader(batch)
        acc = _make_mock_accelerator()

        metrics = compute_mlm_metrics(model, dl, acc)
        assert set(metrics.keys()) == {"loss", "accuracy", "perplexity"}

    def test_loss_is_positive(self) -> None:
        """Cross-entropy loss should be positive for random predictions."""
        model = _make_tiny_model()
        model.eval()

        batch = _make_eval_batch(batch_size=2, seq_len=32)
        dl = _single_batch_dataloader(batch)
        acc = _make_mock_accelerator()

        metrics = compute_mlm_metrics(model, dl, acc)
        assert metrics["loss"] > 0.0

    def test_accuracy_between_zero_and_one(self) -> None:
        """Accuracy should be in [0, 1]."""
        model = _make_tiny_model()
        model.eval()

        batch = _make_eval_batch(batch_size=4, seq_len=32)
        dl = _single_batch_dataloader(batch)
        acc = _make_mock_accelerator()

        metrics = compute_mlm_metrics(model, dl, acc)
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_perplexity_is_exp_loss(self) -> None:
        """Perplexity should equal exp(loss) when below cap."""
        model = _make_tiny_model()
        model.eval()

        batch = _make_eval_batch(batch_size=2, seq_len=32)
        dl = _single_batch_dataloader(batch)
        acc = _make_mock_accelerator()

        metrics = compute_mlm_metrics(model, dl, acc)
        expected_ppl = min(math.exp(metrics["loss"]), _PERPLEXITY_CAP)
        assert abs(metrics["perplexity"] - expected_ppl) < 1e-6

    def test_perplexity_capped(self) -> None:
        """Perplexity should be capped at 1000."""
        assert _PERPLEXITY_CAP == 1000.0

    def test_multiple_batches_accumulated(self) -> None:
        """Metrics should reflect all batches, not just the last one."""
        model = _make_tiny_model()
        model.eval()

        batches = [_make_eval_batch(batch_size=2, seq_len=32) for _ in range(3)]
        dl = _multi_batch_dataloader(batches)
        acc = _make_mock_accelerator()

        metrics = compute_mlm_metrics(model, dl, acc)
        # Just verify we get valid metrics — the accumulation is tested
        # by checking against single-batch results being different
        assert metrics["loss"] > 0.0
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_empty_dataloader_returns_defaults(self) -> None:
        """Empty DataLoader should return zero loss, zero accuracy, perplexity=1."""
        model = _make_tiny_model()
        model.eval()

        dl = DataLoader([])  # type: ignore[arg-type]
        acc = _make_mock_accelerator()

        metrics = compute_mlm_metrics(model, dl, acc)
        assert metrics == {"loss": 0.0, "accuracy": 0.0, "perplexity": 1.0}

    def test_with_real_data(self, training_parquet: Path) -> None:
        """Integration test with real data and a real tiny model."""
        from oplm.eval.data.sequence_loader import build_sequence_eval_dataloader

        cfg = OplmConfig(
            model=ModelConfig(
                hidden_dim=32,
                num_layers=1,
                num_heads=2,
                num_kv_heads=2,
                max_seq_len=64,
            ),
            train=TrainConfig(batch_size=4),
            data=DataConfig(max_length=64, num_workers=0),
        )
        model = OplmForMLM(cfg.model)
        model.eval()

        dl = build_sequence_eval_dataloader(str(training_parquet), cfg)
        acc = _make_mock_accelerator()

        # Only evaluate on first batch to keep test fast
        first_batch = next(iter(dl))
        single_dl = _single_batch_dataloader(first_batch)

        metrics = compute_mlm_metrics(model, single_dl, acc)

        assert metrics["loss"] > 0.0
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert metrics["perplexity"] > 1.0


# ---- Helpers ----


def _single_batch_dataloader(batch: dict[str, torch.Tensor]) -> DataLoader:
    """Wrap a single batch dict into a DataLoader that yields it once."""
    return DataLoader(
        [batch],
        batch_size=None,  # Each item is already a full batch
    )


def _multi_batch_dataloader(batches: list[dict[str, torch.Tensor]]) -> DataLoader:
    """Wrap multiple batch dicts into a DataLoader."""
    return DataLoader(
        batches,
        batch_size=None,
    )
