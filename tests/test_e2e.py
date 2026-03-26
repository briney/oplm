"""End-to-end tests: small model trains a few steps, loss decreases.

NOTE: This module uses a vanilla training loop with synthetic protein sequences.
Once the real training infrastructure (Trainer, Dataset, DataLoader) is
implemented, these tests should be updated to exercise the actual training
workflow with real data, so that the e2e tests reproduce real-world usage.
"""

from __future__ import annotations

import random

import pytest
import torch
from torch.optim import AdamW

from oplm.config import ModelConfig
from oplm.data.tokenizer import ProteinTokenizer
from oplm.model.transformer import OplmForMLM

# Standard amino acid alphabet for synthetic data
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def _random_protein(min_len: int = 20, max_len: int = 50) -> str:
    """Generate a random protein sequence."""
    length = random.randint(min_len, max_len)
    return "".join(random.choice(AMINO_ACIDS) for _ in range(length))


def _make_mlm_batch(
    tokenizer: ProteinTokenizer,
    sequences: list[str],
    mask_prob: float = 0.15,
    max_length: int = 64,
) -> dict[str, torch.Tensor]:
    """Tokenize sequences and apply random MLM masking.

    Returns dict with input_ids, attention_mask, and labels.
    Labels are -100 for non-masked positions (ignored in cross-entropy).
    """
    batch = tokenizer.batch_encode(sequences, max_length=max_length)
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    # Build labels: clone original, set non-masked to -100
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    # Create mask: skip special tokens (cls=0, pad=1, eos=2)
    special_mask = (input_ids <= 2)
    maskable = attention_mask.bool() & ~special_mask

    # Random masking
    rand = torch.rand_like(input_ids, dtype=torch.float)
    mask = (rand < mask_prob) & maskable

    # Replace masked positions with <mask> token in input
    input_ids = input_ids.clone()
    input_ids[mask] = tokenizer.mask_token_id

    # Only compute loss on masked positions
    labels[~mask] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _small_config(**overrides: object) -> ModelConfig:
    """Create a minimal config for fast e2e testing."""
    defaults: dict[str, object] = {
        "hidden_dim": 64,
        "num_heads": 4,
        "num_kv_heads": 2,
        "num_layers": 2,
        "max_seq_len": 64,
    }
    defaults.update(overrides)
    return ModelConfig(**defaults)


# ---------------------------------------------------------------------------
# E2E training: loss decreases
# ---------------------------------------------------------------------------


class TestE2ETraining:
    """Verify a small model can train and learn from synthetic protein data.

    TODO: Replace vanilla training loop with the real Trainer once implemented.
    TODO: Replace synthetic data with real protein sequences and DataLoader.
    """

    def test_loss_decreases(self) -> None:
        """Train for 10 steps, verify final loss < initial loss."""
        torch.manual_seed(42)
        random.seed(42)

        cfg = _small_config()
        model = OplmForMLM(cfg)
        tokenizer = ProteinTokenizer()
        optimizer = AdamW(model.parameters(), lr=1e-3)

        # Generate synthetic protein sequences
        sequences = [_random_protein() for _ in range(32)]

        model.train()
        losses: list[float] = []
        num_steps = 10
        batch_size = 8

        for step in range(num_steps):
            # Sample a mini-batch
            batch_seqs = sequences[
                (step * batch_size) % len(sequences):
                (step * batch_size) % len(sequences) + batch_size
            ]
            batch = _make_mlm_batch(tokenizer, batch_seqs, mask_prob=0.15)

            result = model(
                input_ids=batch["input_ids"],
                labels=batch["labels"],
            )
            loss = result["loss"]
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Loss should decrease (compare average of first 3 vs last 3)
        avg_first = sum(losses[:3]) / 3
        avg_last = sum(losses[-3:]) / 3
        assert avg_last < avg_first, (
            f"Loss did not decrease: first 3 avg={avg_first:.4f}, "
            f"last 3 avg={avg_last:.4f}"
        )

    def test_loss_decreases_with_features(self) -> None:
        """Same test with value residuals, convolutions, and value embeddings."""
        torch.manual_seed(42)
        random.seed(42)

        cfg = _small_config(
            value_residual=True,
            conv_positions="AC",
            num_value_embeds=1,
            qk_norm=True,
        )
        model = OplmForMLM(cfg)
        tokenizer = ProteinTokenizer()
        optimizer = AdamW(model.parameters(), lr=1e-3)

        sequences = [_random_protein() for _ in range(32)]

        model.train()
        losses: list[float] = []

        for step in range(10):
            batch_seqs = sequences[
                (step * 8) % len(sequences):
                (step * 8) % len(sequences) + 8
            ]
            batch = _make_mlm_batch(tokenizer, batch_seqs, mask_prob=0.15)
            result = model(input_ids=batch["input_ids"], labels=batch["labels"])
            loss = result["loss"]
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_first = sum(losses[:3]) / 3
        avg_last = sum(losses[-3:]) / 3
        assert avg_last < avg_first, (
            f"Loss did not decrease: first 3 avg={avg_first:.4f}, "
            f"last 3 avg={avg_last:.4f}"
        )

    def test_loss_decreases_with_attn_residual(self) -> None:
        """Same test with attention residuals enabled."""
        torch.manual_seed(42)
        random.seed(42)

        cfg = _small_config(attn_residual=True, attn_residual_block_size=1)
        model = OplmForMLM(cfg)
        tokenizer = ProteinTokenizer()
        optimizer = AdamW(model.parameters(), lr=1e-3)

        sequences = [_random_protein() for _ in range(32)]

        model.train()
        losses: list[float] = []

        for step in range(10):
            batch_seqs = sequences[
                (step * 8) % len(sequences):
                (step * 8) % len(sequences) + 8
            ]
            batch = _make_mlm_batch(tokenizer, batch_seqs, mask_prob=0.15)
            result = model(input_ids=batch["input_ids"], labels=batch["labels"])
            loss = result["loss"]
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_first = sum(losses[:3]) / 3
        avg_last = sum(losses[-3:]) / 3
        assert avg_last < avg_first, (
            f"Loss did not decrease: first 3 avg={avg_first:.4f}, "
            f"last 3 avg={avg_last:.4f}"
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
