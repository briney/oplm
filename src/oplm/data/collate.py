"""MLM collation: tokenize protein sequences and apply masked-language-model masking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from oplm.data.tokenizer import ProteinTokenizer

# Special token IDs that must never be masked (cls, pad, eos, unk)
_SPECIAL_IDS = frozenset({0, 1, 2, 3})

# Standard amino acid token IDs (L..C, indices 5-24) for random replacement
_AA_MIN_ID = 5
_AA_MAX_ID = 24  # inclusive


class MLMCollator:
    """Collate raw protein sequence dicts into MLM training batches.

    Tokenizes sequences, pads to a fixed length, and applies BERT-style
    masked-language-model masking:

    * ~``mask_prob`` of non-special positions are selected for masking
    * 80% of selected positions → ``<mask>`` token
    * 10% → random amino acid token
    * 10% → keep original token

    Labels are set to the original token ID at masked positions and ``-100``
    elsewhere (ignored by cross-entropy loss).

    Args:
        tokenizer: Protein tokenizer instance.
        max_length: Maximum sequence length including special tokens.
        mask_prob: Fraction of eligible tokens to mask.
        mask_token_prob: Fraction of masked tokens replaced with ``<mask>``.
        random_token_prob: Fraction of masked tokens replaced with a random AA.
    """

    def __init__(
        self,
        tokenizer: ProteinTokenizer,
        max_length: int = 1024,
        mask_prob: float = 0.15,
        mask_token_prob: float = 0.8,
        random_token_prob: float = 0.1,
    ) -> None:
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._mask_prob = mask_prob
        self._mask_token_prob = mask_token_prob
        self._random_token_prob = random_token_prob
        self._mask_token_id = tokenizer.mask_token_id

    def __call__(self, batch: list[dict[str, str]]) -> dict[str, Tensor]:
        """Collate a batch of sequence dicts into MLM tensors.

        Args:
            batch: List of dicts with at least a ``"sequence"`` key (str).

        Returns:
            Dict with ``"input_ids"``, ``"attention_mask"``, and ``"labels"``
            tensors, each of shape ``(B, T)``.
        """
        sequences = [item["sequence"] for item in batch]

        # Pre-truncate raw sequences to leave room for <cls> and <eos>
        raw_max = self._max_length - 2
        sequences = [s[:raw_max] for s in sequences]

        # Tokenize and pad  # (B, T)
        encoded = self._tokenizer.batch_encode(sequences, max_length=self._max_length)
        input_ids: Tensor = encoded["input_ids"]  # (B, T)
        attention_mask: Tensor = encoded["attention_mask"]  # (B, T)

        labels = self._apply_mlm_masking(input_ids, attention_mask)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _apply_mlm_masking(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Apply MLM masking in-place on input_ids and return labels.

        Args:
            input_ids: Token IDs, shape ``(B, T)``. Modified in-place.
            attention_mask: Attention mask, shape ``(B, T)``.

        Returns:
            Labels tensor, shape ``(B, T)``. Original token at masked positions,
            ``-100`` at non-masked positions.
        """
        labels = torch.full_like(input_ids, -100)

        # Eligible positions: not special tokens and within attention mask
        special_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for sid in _SPECIAL_IDS:
            special_mask |= input_ids == sid
        eligible = attention_mask.bool() & ~special_mask  # (B, T)

        # Sample which eligible positions to mask
        rand = torch.rand_like(input_ids, dtype=torch.float)
        selected = eligible & (rand < self._mask_prob)  # (B, T)

        # Set labels at selected positions
        labels[selected] = input_ids[selected]

        # Decide replacement strategy for each selected position
        strategy_rand = torch.rand_like(input_ids, dtype=torch.float)

        # 80% → <mask> token
        mask_replace = selected & (strategy_rand < self._mask_token_prob)
        input_ids[mask_replace] = self._mask_token_id

        # 10% → random amino acid
        random_replace = selected & (
            (strategy_rand >= self._mask_token_prob)
            & (strategy_rand < self._mask_token_prob + self._random_token_prob)
        )
        num_random = int(random_replace.sum().item())
        if num_random > 0:
            random_tokens = torch.randint(
                _AA_MIN_ID, _AA_MAX_ID + 1, (num_random,), device=input_ids.device
            )
            input_ids[random_replace] = random_tokens

        # Remaining 10% → keep original (no action needed)

        return labels
