"""Eval DataLoader for held-out protein sequences with deterministic MLM masking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader

from oplm.data.collate import MLMCollator
from oplm.data.dataset import ShardedProteinDataset
from oplm.data.tokenizer import ProteinTokenizer

if TYPE_CHECKING:
    from oplm.config import OplmConfig

# Fixed eval masking parameters — independent of training config
_EVAL_MASK_PROB = 0.15
_EVAL_SEED = 42


class DeterministicMLMCollator(MLMCollator):
    """MLM collator with a fixed random seed for reproducible masking.

    Each call resets the RNG to a batch-index-dependent state derived from
    a base seed, so the same batch always receives the same mask pattern.
    This makes eval metrics comparable across training steps.

    Args:
        tokenizer: Protein tokenizer instance.
        max_length: Maximum sequence length including special tokens.
        mask_prob: Fraction of eligible tokens to mask.
        seed: Base seed for deterministic masking.
    """

    def __init__(
        self,
        tokenizer: ProteinTokenizer,
        max_length: int = 1024,
        mask_prob: float = _EVAL_MASK_PROB,
        seed: int = _EVAL_SEED,
    ) -> None:
        super().__init__(tokenizer, max_length=max_length, mask_prob=mask_prob)
        self._seed = seed
        self._batch_idx = 0

    def __call__(self, batch: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        """Collate with deterministic masking.

        Sets the torch RNG to a batch-specific state before masking so
        that the same batch always gets the same mask pattern.
        """
        # Save and restore RNG state to avoid affecting other randomness
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(self._seed + self._batch_idx)
        self._batch_idx += 1

        result = super().__call__(batch)

        torch.random.set_rng_state(rng_state)
        return result

    def reset(self) -> None:
        """Reset the batch counter so the next epoch produces identical masks."""
        self._batch_idx = 0


def build_sequence_eval_dataloader(
    path: str,
    cfg: OplmConfig,
) -> DataLoader:
    """Build an eval DataLoader from parquet files.

    Uses the same :class:`ShardedProteinDataset` and :class:`ProteinTokenizer`
    as training, but with:

    * **Fixed 15% mask probability** independent of ``data.mask_prob``.
    * **Deterministic masking** via a fixed random seed, so the same positions
      are masked on every eval run.
    * **No shuffling** — eval data is iterated in a fixed order.

    Args:
        path: Path to a parquet file or directory of shards.
        cfg: Full OPLM configuration.

    Returns:
        A DataLoader yielding batches of ``{input_ids, attention_mask, labels}``.
    """
    dataset = ShardedProteinDataset(
        path,
        shuffle_shards=False,
        shuffle_rows=False,
        seed=_EVAL_SEED,
    )

    tokenizer = ProteinTokenizer()
    collator = DeterministicMLMCollator(
        tokenizer,
        max_length=cfg.data.max_length,
        seed=_EVAL_SEED,
    )

    num_workers = cfg.data.num_workers
    return DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=cfg.data.pin_memory,
        prefetch_factor=cfg.data.prefetch_factor if num_workers > 0 else None,
    )
