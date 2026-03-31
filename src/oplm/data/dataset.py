"""Training dataset classes for sharded parquet protein sequences."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import IterableDataset, get_worker_info

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


# Golden ratio constant for seed mixing
_PHI = 0x9E3779B97F4A7C15
_PRIME = 0x100_0003
_SEED_MASK = 0xFFFF_FFFF

_PARQUET_SUFFIXES = {".parquet", ".parq", ".pq"}


class ShardedProteinDataset(IterableDataset[dict[str, str]]):
    """Iterable dataset over parquet files containing protein sequences.

    Handles both a single parquet file and a directory of parquet shards.
    Loads one shard at a time to bound memory usage. Supports deterministic
    per-epoch shuffling and worker striping. Distributed process sharding is
    delegated to the training launcher (for example, Accelerate).

    Parquet files must contain columns ``sequence_id`` (str) and
    ``sequence`` (str).

    Args:
        path: Path to a single ``.parquet`` file or a directory of shards.
        shuffle_shards: Shuffle shard order each epoch.
        shuffle_rows: Shuffle row indices within each shard each epoch.
        seed: Base seed for deterministic shuffling.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        shuffle_shards: bool = True,
        shuffle_rows: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self._path = Path(path)
        self._shuffle_shards = shuffle_shards
        self._shuffle_rows = shuffle_rows
        self._seed = seed
        self._epoch = -1

        # Enumerate shards and read metadata (no data loaded)
        if self._path.is_dir():
            shard_paths = sorted(
                p for p in self._path.iterdir() if p.suffix.lower() in _PARQUET_SUFFIXES
            )
            if not shard_paths:
                raise RuntimeError(f"No parquet shards found in {self._path}")
        elif self._path.is_file() and self._path.suffix.lower() in _PARQUET_SUFFIXES:
            shard_paths = [self._path]
        else:
            raise RuntimeError(f"Expected a parquet file or directory of shards, got {self._path}")

        self._shards: list[Path] = []
        self._rows_per_shard: list[int] = []
        for sp in shard_paths:
            pf = pq.ParquetFile(sp)
            self._shards.append(sp)
            self._rows_per_shard.append(pf.metadata.num_rows)

        self._total_rows = sum(self._rows_per_shard)

    def __len__(self) -> int:
        """Return the total number of raw examples in the dataset."""
        return self.total_length

    @property
    def total_length(self) -> int:
        """Return the total number of raw examples in the dataset."""
        return self._total_rows

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for deterministic shuffling.

        Args:
            epoch: Epoch number (0-indexed).
        """
        self._epoch = epoch - 1  # __iter__ increments before use

    def __iter__(self) -> Iterator[dict[str, str]]:
        self._epoch += 1
        seed_base = (_PHI ^ self._seed) + (self._epoch * _PRIME)

        wi = get_worker_info()
        if wi is None:
            num_workers, worker_id = 1, 0
        else:
            num_workers, worker_id = wi.num_workers, wi.id

        # Per-epoch shard order
        shard_indices = list(range(len(self._shards)))
        if self._shuffle_shards:
            rng = np.random.default_rng(seed_base & _SEED_MASK)
            rng.shuffle(shard_indices)

        for s_idx in shard_indices:
            nrows = self._rows_per_shard[s_idx]
            row_indices = list(range(nrows))
            if self._shuffle_rows:
                rng_rows = np.random.default_rng((seed_base + 1009 + s_idx) & _SEED_MASK)
                rng_rows.shuffle(row_indices)

            # Worker striping happens within a process; distributed sharding is
            # handled by the outer dataloader wrapper.
            worker_rows = row_indices[worker_id::num_workers]
            if not worker_rows:
                continue

            # Read only needed columns from this shard
            table = pq.read_table(self._shards[s_idx], columns=["sequence_id", "sequence"])
            seq_ids = table.column("sequence_id")
            sequences = table.column("sequence")

            for i in worker_rows:
                yield {
                    "sequence_id": seq_ids[i].as_py(),
                    "sequence": sequences[i].as_py(),
                }


class InterleavedDataset(IterableDataset[dict[str, str]]):
    """Interleave samples from multiple iterable datasets by fraction.

    Each iteration step selects a dataset probabilistically according to
    the provided fractions, then yields the next sample from that dataset.
    Exhausted datasets are re-initialized to allow continuous mixing.

    Args:
        datasets: List of iterable datasets to interleave.
        fractions: Sampling fractions per dataset (normalized internally).
        num_samples: Total samples per epoch. Defaults to sum of dataset lengths.
        seed: Base seed for deterministic dataset selection.

    Raises:
        ValueError: If datasets/fractions are empty or mismatched in length.
    """

    def __init__(
        self,
        datasets: Sequence[IterableDataset[dict[str, str]]],
        fractions: list[float],
        *,
        num_samples: int | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if not datasets:
            raise ValueError("InterleavedDataset requires at least one dataset")
        if len(datasets) != len(fractions):
            raise ValueError("datasets and fractions must have the same length")

        fr = np.asarray(fractions, dtype=np.float64)
        if np.any(fr < 0):
            raise ValueError("fractions must be non-negative")
        total = float(fr.sum())
        if total <= 0:
            raise ValueError("fractions must sum to a positive value")
        fr = fr / total

        self._datasets = datasets
        self._fractions = fr.tolist()
        self._seed = seed
        self._epoch = -1

        if num_samples is None:
            total_len = 0
            for ds in datasets:
                try:
                    total_len += len(ds)  # type: ignore[arg-type]
                except TypeError:
                    total_len = 0
                    break
            num_samples = total_len
        self._num_samples = num_samples

    def __len__(self) -> int:
        return self.total_length

    @property
    def total_length(self) -> int:
        """Return the nominal number of samples in one mixed-data epoch."""
        return self._num_samples

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for deterministic selection and propagate to sub-datasets.

        Args:
            epoch: Epoch number (0-indexed).
        """
        self._epoch = epoch - 1
        for ds in self._datasets:
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(epoch)

    def __iter__(self) -> Iterator[dict[str, str]]:
        self._epoch += 1

        wi = get_worker_info()
        if wi is None:
            num_workers, worker_id = 1, 0
        else:
            num_workers, worker_id = wi.num_workers, wi.id

        if self._num_samples <= 0:
            return

        seed_base = (_PHI ^ self._seed) + (self._epoch * _PRIME)
        rng = np.random.default_rng((seed_base + worker_id) & _SEED_MASK)

        iters = [iter(ds) for ds in self._datasets]
        fr = np.asarray(self._fractions, dtype=np.float64)

        for _pos in range(worker_id, self._num_samples, max(1, num_workers)):
            ds_idx = int(rng.choice(len(iters), p=fr))
            try:
                yield next(iters[ds_idx])
            except StopIteration:
                iters[ds_idx] = iter(self._datasets[ds_idx])
                yield next(iters[ds_idx])
