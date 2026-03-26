"""Tests for the MLM collator."""

from __future__ import annotations

import torch
import pytest

from oplm.data.collate import MLMCollator, _AA_MAX_ID, _AA_MIN_ID, _SPECIAL_IDS
from oplm.data.tokenizer import ProteinTokenizer


@pytest.fixture()
def tokenizer() -> ProteinTokenizer:
    return ProteinTokenizer()


@pytest.fixture()
def collator(tokenizer: ProteinTokenizer) -> MLMCollator:
    return MLMCollator(tokenizer, max_length=64, mask_prob=0.15)


class TestMLMCollator:
    def test_output_keys(self, collator: MLMCollator) -> None:
        batch = [{"sequence_id": "s1", "sequence": "MKWVTFISLLLLFSSAYS"}]
        result = collator(batch)
        assert set(result.keys()) == {"input_ids", "attention_mask", "labels"}

    def test_output_shapes(self, collator: MLMCollator) -> None:
        batch = [
            {"sequence_id": "s1", "sequence": "MKWVTFISLLLLFSSAYS"},
            {"sequence_id": "s2", "sequence": "MVHLTPEEKSAVTALWGKVNV"},
        ]
        result = collator(batch)
        assert result["input_ids"].shape[0] == 2
        assert result["attention_mask"].shape[0] == 2
        assert result["labels"].shape[0] == 2
        # All tensors same shape
        assert result["input_ids"].shape == result["attention_mask"].shape
        assert result["input_ids"].shape == result["labels"].shape

    def test_output_dtypes(self, collator: MLMCollator) -> None:
        batch = [{"sequence_id": "s1", "sequence": "MKWVTFISLLLLFSSAYS"}]
        result = collator(batch)
        assert result["input_ids"].dtype == torch.long
        assert result["attention_mask"].dtype == torch.long
        assert result["labels"].dtype == torch.long

    def test_special_tokens_never_masked(self, tokenizer: ProteinTokenizer) -> None:
        # Use high mask_prob to ensure most tokens are masked
        collator = MLMCollator(tokenizer, max_length=64, mask_prob=0.99)
        batch = [{"sequence_id": "s1", "sequence": "MKWVTFISLLLLFSSAYS"}]

        # Run multiple times to reduce randomness
        for _ in range(10):
            result = collator(batch)
            input_ids = result["input_ids"]
            labels = result["labels"]

            # Check that special token positions are never in labels
            for sid in _SPECIAL_IDS:
                special_positions = (
                    tokenizer.batch_encode(["MKWVTFISLLLLFSSAYS"], max_length=64)["input_ids"]
                    == sid
                )
                # Labels at special positions should always be -100
                assert (labels[special_positions] == -100).all()

    def test_labels_minus_100_at_unmasked(
        self, collator: MLMCollator, tokenizer: ProteinTokenizer
    ) -> None:
        batch = [{"sequence_id": "s1", "sequence": "MKWVTFISLLLLFSSAYS"}]
        result = collator(batch)
        labels = result["labels"]

        # Where labels != -100, that's a masked position
        masked = labels != -100
        # Labels at masked positions should be valid token IDs (not -100)
        assert (labels[masked] >= 0).all()
        assert (labels[masked] < tokenizer.vocab_size).all()

    def test_approximate_mask_rate(self, tokenizer: ProteinTokenizer) -> None:
        collator = MLMCollator(tokenizer, max_length=512, mask_prob=0.15)
        long_seq = "ACDEFGHIKLMNPQRSTVWY" * 25  # 500 AA

        # Run many times and check average mask rate
        total_eligible = 0
        total_masked = 0
        for _ in range(50):
            result = collator([{"sequence_id": "s", "sequence": long_seq}])
            attn = result["attention_mask"].bool()
            labels = result["labels"]
            input_ids = result["input_ids"]

            # Eligible = in attention mask and not special
            special = torch.zeros_like(input_ids, dtype=torch.bool)
            for sid in _SPECIAL_IDS:
                special |= (input_ids == sid) | (labels == sid)
            # Use original tokens to check eligibility
            orig = tokenizer.batch_encode([long_seq[:510]], max_length=512)["input_ids"]
            special_orig = torch.zeros_like(orig, dtype=torch.bool)
            for sid in _SPECIAL_IDS:
                special_orig |= orig == sid
            eligible = attn & ~special_orig
            masked = labels != -100

            total_eligible += eligible.sum().item()
            total_masked += masked.sum().item()

        mask_rate = total_masked / total_eligible
        # Should be approximately 0.15 (within tolerance)
        assert 0.10 < mask_rate < 0.20, f"Mask rate {mask_rate:.3f} outside expected range"

    def test_mask_strategy_breakdown(self, tokenizer: ProteinTokenizer) -> None:
        # Use high mask_prob for sufficient samples
        collator = MLMCollator(
            tokenizer, max_length=512, mask_prob=0.5,
            mask_token_prob=0.8, random_token_prob=0.1,
        )
        long_seq = "ACDEFGHIKLMNPQRSTVWY" * 25

        mask_token_count = 0
        random_token_count = 0
        keep_original_count = 0
        total_masked = 0

        for _ in range(100):
            orig = tokenizer.batch_encode([long_seq[:510]], max_length=512)["input_ids"]
            result = collator([{"sequence_id": "s", "sequence": long_seq}])
            labels = result["labels"]
            input_ids = result["input_ids"]

            masked = labels != -100
            n_masked = masked.sum().item()
            total_masked += n_masked

            # Count replacements
            mask_token_count += (input_ids[masked] == tokenizer.mask_token_id).sum().item()
            keep_original_count += (input_ids[masked] == labels[masked]).sum().item()
            random_token_count += (
                (input_ids[masked] != tokenizer.mask_token_id)
                & (input_ids[masked] != labels[masked])
            ).sum().item()

        # Check approximate breakdown
        if total_masked > 0:
            mask_rate = mask_token_count / total_masked
            random_rate = random_token_count / total_masked
            keep_rate = keep_original_count / total_masked
            assert 0.70 < mask_rate < 0.90, f"Mask token rate {mask_rate:.3f}"
            assert 0.05 < random_rate < 0.20, f"Random token rate {random_rate:.3f}"
            assert 0.03 < keep_rate < 0.20, f"Keep original rate {keep_rate:.3f}"

    def test_truncation(self, tokenizer: ProteinTokenizer) -> None:
        collator = MLMCollator(tokenizer, max_length=16)
        long_seq = "ACDEFGHIKLMNPQRSTVWY" * 5  # 100 AA
        result = collator([{"sequence_id": "s1", "sequence": long_seq}])
        # Should be truncated to max_length
        assert result["input_ids"].shape[1] <= 16

    def test_random_tokens_are_amino_acids(self, tokenizer: ProteinTokenizer) -> None:
        # Force all replacements to be random
        collator = MLMCollator(
            tokenizer, max_length=64, mask_prob=0.99,
            mask_token_prob=0.0, random_token_prob=1.0,
        )
        batch = [{"sequence_id": "s1", "sequence": "ACDEFGHIKLMNPQRSTVWY"}]

        for _ in range(10):
            result = collator(batch)
            input_ids = result["input_ids"]
            labels = result["labels"]
            masked = labels != -100
            if masked.any():
                replaced = input_ids[masked]
                assert (replaced >= _AA_MIN_ID).all()
                assert (replaced <= _AA_MAX_ID).all()
