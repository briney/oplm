"""Tests for the sequence eval data loader."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

if TYPE_CHECKING:
    from pathlib import Path

from oplm.config import DataConfig, OplmConfig, TrainConfig
from oplm.data.tokenizer import ProteinTokenizer
from oplm.eval.data.sequence_loader import (
    _EVAL_MASK_PROB,
    DeterministicMLMCollator,
    build_sequence_eval_dataloader,
)


@pytest.fixture()
def tokenizer() -> ProteinTokenizer:
    return ProteinTokenizer()


@pytest.fixture()
def sample_batch() -> list[dict[str, str]]:
    return [
        {"sequence": "MKWVTFISLLFLFSSAYS"},
        {"sequence": "ACDEFGHIKLMNPQRSTVWY"},
        {"sequence": "MVLSPADKTNVKAAWGKVGA"},
    ]


class TestDeterministicMLMCollator:
    """Test deterministic masking behavior."""

    def test_same_seed_same_masks(
        self, tokenizer: ProteinTokenizer, sample_batch: list[dict[str, str]]
    ) -> None:
        """Same seed and batch index should produce identical masks."""
        collator1 = DeterministicMLMCollator(tokenizer, max_length=64, seed=42)
        collator2 = DeterministicMLMCollator(tokenizer, max_length=64, seed=42)

        result1 = collator1(sample_batch)
        result2 = collator2(sample_batch)

        torch.testing.assert_close(result1["input_ids"], result2["input_ids"])
        torch.testing.assert_close(result1["labels"], result2["labels"])

    def test_different_seeds_different_masks(
        self, tokenizer: ProteinTokenizer, sample_batch: list[dict[str, str]]
    ) -> None:
        """Different seeds should produce different masks."""
        collator1 = DeterministicMLMCollator(tokenizer, max_length=64, seed=42)
        collator2 = DeterministicMLMCollator(tokenizer, max_length=64, seed=999)

        result1 = collator1(sample_batch)
        result2 = collator2(sample_batch)

        # Labels should differ (extremely unlikely to match by chance)
        assert not torch.equal(result1["labels"], result2["labels"])

    def test_reset_reproduces_masks(
        self, tokenizer: ProteinTokenizer, sample_batch: list[dict[str, str]]
    ) -> None:
        """After reset, the collator should produce the same masks again."""
        collator = DeterministicMLMCollator(tokenizer, max_length=64, seed=42)

        result1 = collator(sample_batch)
        collator.reset()
        result2 = collator(sample_batch)

        torch.testing.assert_close(result1["input_ids"], result2["input_ids"])
        torch.testing.assert_close(result1["labels"], result2["labels"])

    def test_consecutive_batches_differ(
        self, tokenizer: ProteinTokenizer, sample_batch: list[dict[str, str]]
    ) -> None:
        """Consecutive calls should produce different masks (different batch index)."""
        collator = DeterministicMLMCollator(tokenizer, max_length=64, seed=42)

        result1 = collator(sample_batch)
        result2 = collator(sample_batch)

        # Different batch indices should give different masks
        assert not torch.equal(result1["labels"], result2["labels"])

    def test_output_format(
        self, tokenizer: ProteinTokenizer, sample_batch: list[dict[str, str]]
    ) -> None:
        """Output should have the standard MLM batch format."""
        collator = DeterministicMLMCollator(tokenizer, max_length=64, seed=42)
        result = collator(sample_batch)

        assert set(result.keys()) == {"input_ids", "attention_mask", "labels"}
        B, T = 3, result["input_ids"].shape[1]
        assert result["input_ids"].shape == (B, T)
        assert result["attention_mask"].shape == (B, T)
        assert result["labels"].shape == (B, T)
        assert result["input_ids"].dtype == torch.long
        assert result["labels"].dtype == torch.long

    def test_uses_fixed_eval_mask_prob(self, tokenizer: ProteinTokenizer) -> None:
        """Default mask_prob should be the fixed eval value (0.15)."""
        collator = DeterministicMLMCollator(tokenizer)
        assert collator._mask_prob == _EVAL_MASK_PROB

    def test_does_not_corrupt_external_rng(
        self, tokenizer: ProteinTokenizer, sample_batch: list[dict[str, str]]
    ) -> None:
        """Collator should save and restore RNG state, not affecting external code."""
        torch.manual_seed(12345)
        before = torch.rand(5)

        torch.manual_seed(12345)
        collator = DeterministicMLMCollator(tokenizer, max_length=64, seed=42)
        collator(sample_batch)
        after = torch.rand(5)

        torch.testing.assert_close(before, after)


class TestBuildSequenceEvalDataloader:
    """Test the build_sequence_eval_dataloader function."""

    def test_builds_from_parquet(self, training_parquet: Path) -> None:
        """Should build a DataLoader from a real parquet file."""
        cfg = OplmConfig(
            train=TrainConfig(batch_size=4),
            data=DataConfig(max_length=128, num_workers=0),
        )
        dl = build_sequence_eval_dataloader(str(training_parquet), cfg)

        batch = next(iter(dl))
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch
        assert batch["input_ids"].shape[0] == 4
        assert batch["input_ids"].shape[1] == 128

    def test_deterministic_across_calls(self, training_parquet: Path) -> None:
        """Two DataLoaders from the same config should yield identical first batches."""
        cfg = OplmConfig(
            train=TrainConfig(batch_size=4),
            data=DataConfig(max_length=128, num_workers=0),
        )
        dl1 = build_sequence_eval_dataloader(str(training_parquet), cfg)
        dl2 = build_sequence_eval_dataloader(str(training_parquet), cfg)

        batch1 = next(iter(dl1))
        batch2 = next(iter(dl2))

        torch.testing.assert_close(batch1["input_ids"], batch2["input_ids"])
        torch.testing.assert_close(batch1["labels"], batch2["labels"])

    def test_no_shuffling(self, training_parquet: Path) -> None:
        """Eval DataLoader should iterate in the same order every time."""
        cfg = OplmConfig(
            train=TrainConfig(batch_size=2),
            data=DataConfig(max_length=64, num_workers=0),
        )

        def get_first_n_ids(n: int) -> list[torch.Tensor]:
            dl = build_sequence_eval_dataloader(str(training_parquet), cfg)
            batches = []
            for i, batch in enumerate(dl):
                if i >= n:
                    break
                batches.append(batch["input_ids"].clone())
            return batches

        run1 = get_first_n_ids(3)
        run2 = get_first_n_ids(3)

        for b1, b2 in zip(run1, run2, strict=True):
            torch.testing.assert_close(b1, b2)
