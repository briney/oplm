"""Tests for the sequence dataloader builders (Phase 5).

Covers the shared §4.7 batch contract, the train/eval policy split (training
reshuffles across epochs and draws fresh masks; sequence eval is byte-identical
across passes and keeps natural order), multi-dataset interleaving, and the
``num_workers == 0`` → ``prefetch_factor is None`` DataLoader rule.

Tests run single-process with ``num_workers=0`` so iteration is deterministic and
free of multiprocessing flakiness; the underlying striping is exercised in
``test_dataset.py``. Sequences are the real human proteins from the shared
``sequence_shards`` fixtures (conftest).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from oplm.config import DataConfig, ModelConfig, OplmConfig, TrainConfig
from oplm.data.sequence.dataset import InterleavedDataset, ShardedProteinDataset
from oplm.data.sequence.loaders import build_sequence_eval_dataloader, build_train_dataloader
from oplm.data.tokenizer import get_tokenizer, pad_token_id

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from torch import Tensor
    from torch.utils.data import DataLoader

_IGNORE_INDEX = -100
_MAX_SEQ_LEN = 64
_BATCH_SIZE = 4


def _make_config(
    train: object,
    *,
    num_workers: int = 0,
    weighted_masking: bool = False,
    seed: int = 123,
) -> OplmConfig:
    """Build a tiny config wired to a fixture dataset for the loader tests."""
    return OplmConfig(
        model=ModelConfig(max_seq_len=_MAX_SEQ_LEN),
        train=TrainConfig(batch_size=_BATCH_SIZE, seed=seed),
        data=DataConfig(
            train=train,
            num_workers=num_workers,
            pin_memory=False,
            weighted_masking=weighted_masking,
            mask_prob=0.15,
        ),
    )


def _batches(loader: DataLoader[dict[str, Tensor]]) -> list[dict[str, Tensor]]:
    """Materialize every batch from one pass over ``loader``."""
    return list(loader)


def _originals_in_order(batches: list[dict[str, Tensor]]) -> list[tuple[int, ...]]:
    """Reconstruct each row's original (pre-masking) token ids, in dataloader order.

    Masking hides the input, but ``labels`` hold the original id at every masked
    position and ``-100`` elsewhere, so ``where(labels != -100, labels, input_ids)``
    recovers the pre-masking ids regardless of the 80/10/10 replacement. Padding is
    trimmed via the attention mask, leaving the full ``<cls> … <eos>`` token run.
    This lets row order be compared without being perturbed by dynamic masking.
    """
    originals: list[tuple[int, ...]] = []
    for batch in batches:
        ids = batch["input_ids"]
        labels = batch["labels"]
        attention = batch["attention_mask"]
        recovered = torch.where(labels != _IGNORE_INDEX, labels, ids)  # (B, T)
        for row in range(ids.shape[0]):
            keep = attention[row] == 1
            originals.append(tuple(recovered[row][keep].tolist()))
    return originals


# --------------------------------------------------------------------------- #
# Batch contract
# --------------------------------------------------------------------------- #


def test_train_batch_contract(sequence_shards: Path) -> None:
    """Train batches expose exactly the §4.7 keys/shapes/dtypes; T ≤ max_seq_len."""
    cfg = _make_config(str(sequence_shards))
    loader = build_train_dataloader(cfg)
    pad_id = pad_token_id()

    batches = _batches(loader)
    assert batches  # the fixture has rows
    seen_rows = 0
    for batch in batches:
        assert set(batch) == {"input_ids", "attention_mask", "labels"}
        ids = batch["input_ids"]
        attention = batch["attention_mask"]
        labels = batch["labels"]
        b, t = ids.shape
        assert t <= _MAX_SEQ_LEN
        assert attention.shape == (b, t)
        assert labels.shape == (b, t)
        for tensor in (ids, attention, labels):
            assert tensor.dtype == torch.long
        # Padding cells are exactly the zero-attention cells.
        assert torch.equal(ids == pad_id, attention == 0)
        # Masked positions (labels != -100) only ever land on real (non-pad) tokens.
        masked = labels != _IGNORE_INDEX
        assert torch.all(attention[masked] == 1)
        seen_rows += b

    # Every fixture row is served exactly once over the epoch.
    assert seen_rows == len(ShardedProteinDataset(sequence_shards))


# --------------------------------------------------------------------------- #
# Eval policy: deterministic, natural order
# --------------------------------------------------------------------------- #


def test_eval_is_deterministic_across_passes(sequence_shards: Path) -> None:
    """Two passes over the eval loader yield byte-identical batches."""
    cfg = _make_config(str(sequence_shards))
    loader = build_sequence_eval_dataloader(str(sequence_shards), cfg)

    first = _batches(loader)
    second = _batches(loader)
    assert len(first) == len(second)
    for a, b in zip(first, second, strict=True):
        assert set(a) == set(b)
        for key in a:
            assert torch.equal(a[key], b[key]), f"eval batch differs on {key!r}"


def test_eval_preserves_natural_order(sequence_shards: Path) -> None:
    """Eval disables shuffling: rows arrive in shard-then-row (natural) order."""
    cfg = _make_config(str(sequence_shards))
    loader = build_sequence_eval_dataloader(str(sequence_shards), cfg)
    tokenizer = get_tokenizer()

    natural = ShardedProteinDataset(sequence_shards, shuffle_shards=False, shuffle_rows=False)
    max_residues = _MAX_SEQ_LEN - 2  # collator truncates to leave room for <cls>/<eos>
    expected = [
        tuple(tokenizer(str(row["sequence"])[:max_residues])["input_ids"]) for row in natural
    ]

    assert _originals_in_order(_batches(loader)) == expected


# --------------------------------------------------------------------------- #
# Train policy: reshuffles across epochs
# --------------------------------------------------------------------------- #


def test_train_differs_across_epochs(sequence_shards: Path) -> None:
    """Row order changes between epochs while the underlying data is unchanged."""
    cfg = _make_config(str(sequence_shards))
    loader = build_train_dataloader(cfg)
    dataset = loader.dataset

    dataset.set_epoch(0)  # type: ignore[attr-defined]  # IterableDataset carries set_epoch
    epoch0 = _originals_in_order(_batches(loader))
    dataset.set_epoch(1)  # type: ignore[attr-defined]  # IterableDataset carries set_epoch
    epoch1 = _originals_in_order(_batches(loader))

    assert sorted(epoch0) == sorted(epoch1)  # same rows
    assert epoch0 != epoch1  # reshuffled order


# --------------------------------------------------------------------------- #
# Dataset assembly: single vs. interleaved
# --------------------------------------------------------------------------- #


def test_single_dataset_is_not_interleaved(sequence_shards: Path) -> None:
    """A single training entry is used directly (no InterleavedDataset wrapper)."""
    cfg = _make_config(str(sequence_shards))
    loader = build_train_dataloader(cfg)
    assert isinstance(loader.dataset, ShardedProteinDataset)


def test_multi_dataset_wraps_interleaved(
    tmp_path: Path,
    make_sequence_shards: Callable[..., Path],
    real_records: list[tuple[str, str]],
) -> None:
    """More than one training entry wraps the sources in an InterleavedDataset."""
    dir0 = (tmp_path / "ds0").resolve()
    dir1 = (tmp_path / "ds1").resolve()
    dir0.mkdir()
    dir1.mkdir()
    make_sequence_shards(dir0, real_records[:8], n_shards=2, id_prefix="ds0_")
    make_sequence_shards(dir1, real_records[8:16], n_shards=2, id_prefix="ds1_")

    cfg = _make_config(
        {
            "ds0": {"path": str(dir0), "fraction": 0.5},
            "ds1": {"path": str(dir1), "fraction": 0.5},
        }
    )
    loader = build_train_dataloader(cfg)
    assert isinstance(loader.dataset, InterleavedDataset)

    # The interleaved loader still produces the §4.7 contract.
    batch = next(iter(loader))
    assert set(batch) == {"input_ids", "attention_mask", "labels"}


def test_empty_train_raises() -> None:
    """An empty ``data.train`` spec is a configuration error, not a silent no-op."""
    cfg = _make_config(None)
    with pytest.raises(ValueError, match="no datasets"):
        build_train_dataloader(cfg)


# --------------------------------------------------------------------------- #
# DataLoader worker settings
# --------------------------------------------------------------------------- #


def test_prefetch_factor_none_without_workers(sequence_shards: Path) -> None:
    """``num_workers == 0`` forces ``prefetch_factor`` to ``None`` (PyTorch rule)."""
    cfg = _make_config(str(sequence_shards), num_workers=0)
    loader = build_train_dataloader(cfg)
    assert loader.num_workers == 0
    assert loader.prefetch_factor is None


def test_prefetch_factor_set_with_workers(sequence_shards: Path) -> None:
    """With workers, the configured ``prefetch_factor`` is passed through."""
    cfg = _make_config(str(sequence_shards), num_workers=2)
    loader = build_train_dataloader(cfg)
    assert loader.num_workers == 2
    assert loader.prefetch_factor == cfg.data.prefetch_factor
