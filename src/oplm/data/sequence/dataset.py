"""Iterable sequence datasets.

:class:`ShardedProteinDataset` streams rows from one or more parquet shards with
reproducible, rank/worker-aware shuffling and striping. :class:`InterleavedDataset`
samples across several sources according to fixed mixing fractions.

Both are :class:`~torch.utils.data.IterableDataset` subclasses: tokenization
happens later, in the collator (``data/sequence/collate.py``). Rows are yielded as
``{"sequence_id": str, "sequence": str}`` dicts, optionally carrying a
``"masking_weights"`` field (see :func:`ShardedProteinDataset.__init__`).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow.parquet as pq
import torch
from torch import distributed as dist
from torch.utils.data import IterableDataset, get_worker_info

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

# Parquet shard discovery accepts these suffixes (case-insensitive).
_PARQUET_SUFFIXES = frozenset({".parquet", ".parq", ".pq"})

# Optional per-residue masking-weight column (read only when requested).
_WEIGHTS_COLUMN = "masking_weights"

# --------------------------------------------------------------------------- #
# Seed mixing (fixed constants — reproducibility contract, do not change)
# --------------------------------------------------------------------------- #

_PHI = 0x9E3779B97F4A7C15  # golden-ratio mix
_PRIME = 0x0100_0003
_MASK32 = 0xFFFF_FFFF


def _epoch_seed(base: int, epoch: int) -> int:
    """Mix the base seed with the epoch index into a 32-bit shuffle seed."""
    return ((_PHI ^ base) + epoch * _PRIME) & _MASK32


def _shard_row_seed(epoch_seed: int, s: int) -> int:
    """Derive a per-shard row-shuffle seed so different shards permute differently."""
    return (epoch_seed + 1009 + s) & _MASK32


# --------------------------------------------------------------------------- #
# Distributed / worker context
# --------------------------------------------------------------------------- #


def _resolve_distributed_context() -> tuple[int, int, int, int]:
    """Resolve the joint ``(rank, world_size, worker_id, num_workers)`` context.

    Rank/world-size come from ``torch.distributed`` when a process group is
    initialized, else from the ``RANK`` / ``WORLD_SIZE`` environment variables,
    else ``(0, 1)``. Worker id/count come from
    :func:`torch.utils.data.get_worker_info`, else ``(0, 1)``.

    Returns:
        ``(rank, world_size, worker_id, num_workers)``.
    """
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))

    info = get_worker_info()
    if info is None:
        worker_id, num_workers = 0, 1
    else:
        worker_id, num_workers = info.id, info.num_workers

    return rank, world_size, worker_id, num_workers


def _joint_stripe() -> tuple[int, int]:
    """Return ``(joint_index, stride)`` for the current ``(rank, worker)``.

    ``joint_index = rank * num_workers + worker_id`` and
    ``stride = world_size * num_workers`` together partition a row stream into
    disjoint, gap-free subsets — one per ``(rank, worker)``.
    """
    rank, world_size, worker_id, num_workers = _resolve_distributed_context()
    joint_index = rank * num_workers + worker_id
    stride = world_size * num_workers
    return joint_index, stride


class ShardedProteinDataset(IterableDataset[dict[str, object]]):
    """Iterable dataset over one or more parquet shards of protein sequences.

    Handles a single parquet file or a directory of shards, loading one shard at
    a time to bound memory. Shuffling is deterministic per ``(seed, epoch)`` and
    identical across runs and ranks; rank/worker striping is explicit (over the
    joint ``(rank, worker)`` index) so coverage does not depend on launcher
    behavior. See docs/DATA_TOOLING.md §4.2.

    Each shard must contain columns ``sequence_id`` (str) and ``sequence`` (str,
    raw one-letter amino acids). An optional ``masking_weights`` column
    (``list[float]``, one weight per residue) is read only when
    ``load_masking_weights`` is set.

    Args:
        path: A single ``.parquet``/``.parq``/``.pq`` file, or a directory of such
            shards.
        shuffle_shards: Shuffle shard *order* each epoch.
        shuffle_rows: Shuffle row order *within* each shard each epoch.
        seed: Base seed for deterministic shuffling.
        load_masking_weights: Read the optional ``masking_weights`` column and
            attach it to each row (``None`` for a row/shard without weights). When
            ``False``, the column is never read, even if present.

    Raises:
        FileNotFoundError: If ``path`` is neither a parquet file nor a directory
            containing parquet shards.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        shuffle_shards: bool = True,
        shuffle_rows: bool = True,
        seed: int = 0,
        load_masking_weights: bool = False,
    ) -> None:
        super().__init__()
        self._path = Path(path)
        self._shuffle_shards = shuffle_shards
        self._shuffle_rows = shuffle_rows
        self._seed = seed
        self._load_masking_weights = load_masking_weights
        self._epoch = 0

        shard_paths = self._discover_shards(self._path)

        self._shards: list[Path] = []
        self._rows_per_shard: list[int] = []
        self._shard_has_weights: list[bool] = []
        for shard in shard_paths:
            pf = pq.ParquetFile(shard)
            self._shards.append(shard)
            self._rows_per_shard.append(pf.metadata.num_rows)
            self._shard_has_weights.append(_WEIGHTS_COLUMN in pf.schema_arrow.names)

        self._total_rows = sum(self._rows_per_shard)

    @staticmethod
    def _discover_shards(path: Path) -> list[Path]:
        """Resolve ``path`` to a sorted list of parquet shard files."""
        if path.is_dir():
            shards = sorted(p for p in path.iterdir() if p.suffix.lower() in _PARQUET_SUFFIXES)
            if not shards:
                raise FileNotFoundError(f"no parquet shards found in directory {path}")
            return shards
        if path.is_file() and path.suffix.lower() in _PARQUET_SUFFIXES:
            return [path]
        raise FileNotFoundError(f"expected a parquet file or directory of shards, got {path}")

    def __len__(self) -> int:
        """Return the total number of rows across all shards."""
        return self._total_rows

    @property
    def total_length(self) -> int:
        """Total number of rows across all shards (the full, un-striped count)."""
        return self._total_rows

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used to seed shuffling.

        The shuffle seed is a pure function of ``(seed, epoch)``, so the same
        epoch yields the same order across runs and ranks. The trainer calls this
        on each epoch boundary (docs/DATA_TOOLING.md §10.1).

        Args:
            epoch: Epoch index (0-based).
        """
        self._epoch = epoch

    def _shard_order(self, epoch_seed: int) -> list[int]:
        """Return shard indices in this epoch's (optionally shuffled) order."""
        n = len(self._shards)
        if not self._shuffle_shards:
            return list(range(n))
        generator = torch.Generator()
        generator.manual_seed(epoch_seed)
        return torch.randperm(n, generator=generator).tolist()

    def _row_order(self, n_rows: int, epoch_seed: int, shard_idx: int) -> list[int]:
        """Return row indices for one shard in this epoch's (optional) row order."""
        if not self._shuffle_rows:
            return list(range(n_rows))
        generator = torch.Generator()
        generator.manual_seed(_shard_row_seed(epoch_seed, shard_idx))
        return torch.randperm(n_rows, generator=generator).tolist()

    def __iter__(self) -> Iterator[dict[str, object]]:
        epoch_seed = _epoch_seed(self._seed, self._epoch)
        joint_index, stride = _joint_stripe()

        # `global_idx` runs across all shards in iteration order; a row is served
        # to this (rank, worker) iff its global index is congruent to joint_index
        # mod stride. Unioned over joint_index in [0, stride) this covers every
        # row exactly once.
        global_idx = 0
        for shard_idx in self._shard_order(epoch_seed):
            n_rows = self._rows_per_shard[shard_idx]
            row_order = self._row_order(n_rows, epoch_seed, shard_idx)

            base = global_idx
            global_idx += n_rows
            selected = [
                row_idx
                for offset, row_idx in enumerate(row_order)
                if (base + offset) % stride == joint_index
            ]
            if not selected:
                continue  # no rows in this shard belong to this (rank, worker)

            yield from self._read_rows(shard_idx, selected)

    def _read_rows(self, shard_idx: int, row_indices: list[int]) -> Iterator[dict[str, object]]:
        """Read ``row_indices`` from one shard and yield them as row dicts."""
        has_weights = self._load_masking_weights and self._shard_has_weights[shard_idx]
        columns = ["sequence_id", "sequence"]
        if has_weights:
            columns.append(_WEIGHTS_COLUMN)

        table = pq.read_table(self._shards[shard_idx], columns=columns)
        seq_ids = table.column("sequence_id")
        sequences = table.column("sequence")
        weights = table.column(_WEIGHTS_COLUMN) if has_weights else None

        for row_idx in row_indices:
            row: dict[str, object] = {
                "sequence_id": seq_ids[row_idx].as_py(),
                "sequence": sequences[row_idx].as_py(),
            }
            if self._load_masking_weights:
                # Column-absent shards (and absent rows) surface None; the
                # collator falls back to uniform weights with a one-time warning.
                row["masking_weights"] = weights[row_idx].as_py() if weights is not None else None
            yield row


# Returned by `_next_or_refill` when a source yields nothing for this (rank,
# worker) even after re-iteration — distinct from any real row dict.
_EXHAUSTED = object()


class InterleavedDataset(IterableDataset[dict[str, object]]):
    """Interleave several iterable datasets by sampling fraction.

    Each step picks a source by its (normalized) fraction and pulls that source's
    next item; an exhausted source is re-initialized so sources of unequal size
    keep mixing at the requested ratio for the whole epoch. Sub-datasets perform
    their own ``(rank, worker)`` striping, so this class strides only the *number
    of steps* per worker. See docs/DATA_TOOLING.md §4.3.

    Args:
        datasets: Source iterable datasets (typically :class:`ShardedProteinDataset`).
        fractions: Per-source sampling weights; normalized to sum to 1.0.
        num_samples: Nominal samples per (full) epoch. Defaults to the sum of the
            sources' ``len()`` when available, else ``0``.
        seed: Base seed for deterministic source selection.

    Raises:
        ValueError: If ``datasets`` is empty, ``datasets``/``fractions`` lengths
            differ, a fraction is negative, or the fractions do not sum to a
            positive value.
    """

    def __init__(
        self,
        datasets: Sequence[IterableDataset[dict[str, object]]],
        fractions: Sequence[float],
        *,
        num_samples: int | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if not datasets:
            raise ValueError("InterleavedDataset requires at least one dataset")
        if len(datasets) != len(fractions):
            raise ValueError(
                f"datasets and fractions must have the same length: "
                f"{len(datasets)} != {len(fractions)}"
            )

        self._datasets = list(datasets)
        self._fractions = _normalize_fractions(fractions)
        self._seed = seed
        self._epoch = 0
        self._num_samples = self._default_num_samples() if num_samples is None else int(num_samples)

    def _default_num_samples(self) -> int:
        """Sum of source lengths, or 0 if any source has no defined length."""
        total = 0
        for ds in self._datasets:
            try:
                total += len(ds)  # ty: ignore[invalid-argument-type]  # IterableDataset len is optional
            except TypeError:
                return 0
        return total

    def __len__(self) -> int:
        """Return the nominal number of samples in one mixed epoch."""
        return self._num_samples

    @property
    def total_length(self) -> int:
        """Nominal number of samples in one mixed epoch."""
        return self._num_samples

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch and propagate it to every sub-dataset.

        Args:
            epoch: Epoch index (0-based).
        """
        self._epoch = epoch
        for ds in self._datasets:
            set_epoch = getattr(ds, "set_epoch", None)
            if callable(set_epoch):
                set_epoch(epoch)

    def __iter__(self) -> Iterator[dict[str, object]]:
        joint_index, stride = _joint_stripe()
        if self._num_samples <= 0 or joint_index >= self._num_samples:
            return

        # One generator per (epoch, rank, worker) so each worker draws an
        # independent source sequence while staying reproducible.
        generator = torch.Generator()
        generator.manual_seed((_epoch_seed(self._seed, self._epoch) + joint_index) & _MASK32)

        n_steps = len(range(joint_index, self._num_samples, stride))
        weights = torch.tensor(self._fractions, dtype=torch.float64)
        choices = torch.multinomial(weights, n_steps, replacement=True, generator=generator)

        iters = [iter(ds) for ds in self._datasets]
        for source_idx in choices.tolist():
            item = self._next_or_refill(iters, source_idx)
            if item is _EXHAUSTED:
                continue  # source produced nothing for this (rank, worker)
            yield item  # ty: ignore[invalid-yield]  # _EXHAUSTED sentinel filtered above

    def _next_or_refill(self, iters: list[Iterator[dict[str, object]]], source_idx: int) -> object:
        """Pull the next item from a source, re-iterating once if exhausted.

        Returns :data:`_EXHAUSTED` if the source yields nothing even after a fresh
        ``iter()`` (e.g. it serves no rows to this ``(rank, worker)``), avoiding a
        ``StopIteration`` escaping the generator (PEP 479).
        """
        try:
            return next(iters[source_idx])
        except StopIteration:
            iters[source_idx] = iter(self._datasets[source_idx])
            try:
                return next(iters[source_idx])
            except StopIteration:
                return _EXHAUSTED


def _normalize_fractions(fractions: Sequence[float]) -> list[float]:
    """Validate and normalize sampling fractions to sum to 1.0.

    Args:
        fractions: Per-source weights.

    Returns:
        Fractions scaled to sum to 1.0.

    Raises:
        ValueError: If any fraction is negative or the total is not positive.
    """
    values = [float(f) for f in fractions]
    if any(f < 0 for f in values):
        raise ValueError(f"fractions must be non-negative, got {values}")
    total = sum(values)
    if total <= 0:
        raise ValueError(f"fractions must sum to a positive value, got {total}")
    return [f / total for f in values]
