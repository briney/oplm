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

import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import distributed as dist
from torch.utils.data import IterableDataset, get_worker_info

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

logger = logging.getLogger(__name__)

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


def train_data_fingerprint(raw_train: Any) -> str:
    """Hex sha256 identifying the training data by content, not location.

    Hashes each ``data.train`` entry's fraction and the raw parquet footer of each of its
    shards (row counts, row-group layout, statistics), in config order then shard order.
    The same shards at another path hash equal; a re-cut or reordered set does not.
    """
    from oplm.data.config import parse_train_configs

    h = hashlib.sha256()
    for entry in parse_train_configs(raw_train):
        h.update(repr(entry.fraction).encode())
        for shard in ShardedProteinDataset._discover_shards(Path(entry.path)):
            with shard.open("rb") as f:
                f.seek(-8, os.SEEK_END)
                n = int.from_bytes(f.read(4), "little")
                f.seek(-8 - n, os.SEEK_END)
                h.update(f.read(n))
    return h.hexdigest()


def _epoch_seed(base: int, epoch: int) -> int:
    """Mix the base seed with the epoch index into a 32-bit shuffle seed."""
    return ((_PHI ^ base) + epoch * _PRIME) & _MASK32


def _pass_seed(epoch_seed: int, pass_index: int) -> int:
    """Seed of a refilled source's ``pass_index``-th pass within one epoch (pass 0 is the epoch's).

    Mixed so that no (epoch, pass) seed coincides with another epoch's pass-0 seed.
    """
    if pass_index == 0:
        return epoch_seed
    return ((epoch_seed ^ (pass_index * 0x85EB_CA6B)) + pass_index * 0xC2B2_AE35) & _MASK32


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


def _resolve_sample_skip(
    batches_in_epoch: int | None, per_rank_batch: int, num_workers: int, stream_length: int
) -> int:
    """Resolve an armed batch-count skip into this stream's sample-offset skip.

    Shared round-robin arithmetic behind both :meth:`ShardedProteinDataset._resolved_skip`
    and :class:`InterleavedDataset`'s ``__iter__``: each worker computes its own share of
    the globally-armed ``batches_in_epoch``, matching DataLoader's round-robin batch
    assignment across workers, then converts that batch count to a sample count and
    clamps it to ``stream_length``. It must not wrap: a stream that was fully consumed
    when the checkpoint was taken (a cursor exactly on the epoch boundary) resolves to
    a skip equal to ``stream_length``, i.e. an empty stream, so the Trainer's
    ``StopIteration`` path rolls to the next epoch just as the uninterrupted run did.
    Wrapping to 0 would replay the finished epoch, since the next epoch reshuffles.

    Args:
        batches_in_epoch: Globally-armed batch count already consumed this epoch, or
            ``None`` if unarmed.
        per_rank_batch: Samples per batch contributed by one (rank, worker) stream.
        num_workers: DataLoader ``num_workers`` the interrupted run used.
        stream_length: This stream's total sample count for the epoch.

    Returns:
        The resolved sample skip, in ``[0, stream_length]`` (``stream_length`` meaning
        the whole stream was already consumed); ``0`` when unarmed or when
        ``stream_length <= 0``.
    """
    if batches_in_epoch is None:
        return 0
    _, _, worker_id, _ = _resolve_distributed_context()
    batch_count = len(range(worker_id, batches_in_epoch, num_workers))
    skip = batch_count * per_rank_batch
    return min(skip, max(stream_length, 0))


def _resolve_rows_skip(rows_per_worker: Sequence[int], stream_length: int) -> int:
    """This worker's entry of an armed per-worker row-count skip, clamped to its stream.

    Token-mode resume (:meth:`ShardedProteinDataset.set_resume_rows`): the checkpoint records how
    many rows each worker had emitted, so no batch-count arithmetic is involved. Clamped, never
    wrapped, for the reason given in :func:`_resolve_sample_skip`.

    Raises:
        ValueError: ``rows_per_worker`` does not have one entry per DataLoader worker.
    """
    _, _, worker_id, num_workers = _resolve_distributed_context()
    if len(rows_per_worker) != num_workers:
        raise ValueError(
            f"resume rows have {len(rows_per_worker)} entries for {num_workers} DataLoader "
            "workers; the checkpoint's worker count must match data.num_workers"
        )
    return min(int(rows_per_worker[worker_id]), max(stream_length, 0))


@dataclass(frozen=True)
class DataCursor:
    """Position of the training stream, plus the layout it was recorded under.

    Captured at checkpoint time (one per run, not per rank — in sequence mode ranks step in
    lockstep so ``batches_in_epoch`` is the same value everywhere; token mode adds the
    per-rank ``rank_epochs`` and ``worker_rows`` tables) and replayed on resume via
    :meth:`ShardedProteinDataset.set_resume_skip` (sequence mode) or ``set_resume_rows``
    (token mode). A sequence-mode cursor can resume a token-mode run (its position maps onto
    per-worker rows); a token-mode cursor cannot resume a sequence-mode run. A sequence-mode
    cursor resumes only at its own layout (``world_size``, ``num_workers``,
    ``per_rank_batch``, ``seed``). A token-mode cursor needs only the same ``seed``: at another
    world size or worker count its position becomes a :class:`Reshard` event (``reshards``)
    and the live streams stripe the rows the old ones had not consumed.

    Attributes:
        epoch: Epoch index the cursor was captured in.
        batches_in_epoch: Count of batches consumed so far this epoch (per rank; identical
            across ranks in sequence mode, where training steps in lockstep; the rank-0
            count, informational only, in token mode).
        world_size: Number of ranks the interrupted run used.
        num_workers: DataLoader ``num_workers`` the interrupted run used.
        per_rank_batch: Samples per batch contributed by one (rank, worker) stream; 0 in
            token mode.
        seed: Base dataset seed the interrupted run used.
        rank_epochs: Token mode only: each rank's epoch at the save (ranks roll over on their
            own).
        worker_rows: Token mode only: a ranks x workers table of the rows each worker had
            emitted this epoch, the exact position whatever the batch size (draws, for a mix).
        worker_sources: Token mode only: a ranks x workers table of each worker's per-source
            ``[pass, offset]`` (a mix's position tag; ``None`` for a single source or a worker
            that had not produced a batch). ``None`` in cursors from before it existed.
        reshards: Token mode only: the :class:`Reshard` events (as dicts) that still apply to
            some rank's current or later epoch.
        data_fingerprint: Content hash of ``data.train``, or ``None``.
    """

    epoch: int
    batches_in_epoch: int
    world_size: int
    num_workers: int
    per_rank_batch: int
    seed: int
    rank_epochs: list[int] | None = None
    worker_rows: list[list[int]] | None = None
    worker_sources: list[list[list[list[int]] | None]] | None = None
    reshards: list[dict] = field(default_factory=list)
    data_fingerprint: str | None = None


@dataclass(frozen=True)
class Reshard:
    """A layout change: where every old ``(rank, worker)`` stream stood when it happened.

    Replayed as a layer (see :func:`_resolve_layers`), it removes what the old streams had
    consumed from the epoch's global order; the live stride then stripes what is left. Per-stream
    lists are in joint-index order (``rank * num_workers + worker``).

    Attributes:
        stride: The old stream count, ``world_size * num_workers``.
        epochs: Each old stream's epoch at the save.
        rows: Each old stream's rows emitted in that epoch.
        sources: Mix only: per old stream, per source ``[pass, offset]`` (``None`` for a stream
            that had not produced a batch).
    """

    stride: int
    epochs: list[int]
    rows: list[int]
    sources: list[list[list[int]] | None] | None = None


# One applicable reshard event: (stride, consumed rows per old stream), unclamped.
Layer = tuple[int, list[int]]

# "Consumed everything" sentinel; the layer math clamps it to each stream's length.
_FULL = 2**62


def _resolve_layers(
    reshards: Sequence[Reshard], epoch: int, *, source: int | None = None, pass_index: int = 0
) -> list[Layer]:
    """The layers that apply to ``epoch`` (or to one mix source's ``pass_index``-th pass of it).

    An event applies iff ``min(epochs) <= epoch <= max(epochs)``. An old stream already in a
    later epoch consumed all of ``epoch``; one still in an earlier epoch consumed none of it. For a
    stream in ``epoch``, the count is its ``rows`` (or, with ``source``, its ``[pass, offset]`` for
    that source: all of an earlier pass, ``offset`` of this one, none of a later one). Events that
    consumed nothing are dropped: an identity layer would trim the ``drop_remainder`` tail twice.

    Raises:
        ValueError: ``source`` is given but an applicable event has no per-source positions.
    """
    layers: list[Layer] = []
    for reshard in reshards:
        if not min(reshard.epochs) <= epoch <= max(reshard.epochs):
            continue
        if source is None:
            here = reshard.rows
        elif reshard.sources is None:
            raise ValueError("reshard event has no per-source positions for a mixed dataset")
        else:
            here = [
                _FULL if pass_ > pass_index else offset if pass_ == pass_index else 0
                for pass_, offset in (
                    positions[source] if positions is not None else (0, 0)
                    for positions in reshard.sources
                )
            ]
        consumed = [
            _FULL if stream_epoch > epoch else 0 if stream_epoch < epoch else count
            for stream_epoch, count in zip(reshard.epochs, here, strict=True)
        ]
        if any(consumed):
            layers.append((reshard.stride, consumed))
    return layers


def _stripe_lengths(size: int, stride: int, drop_remainder: bool) -> np.ndarray:
    """Length of each of the ``stride`` stripes of ``size`` items."""
    if drop_remainder:
        return np.full(stride, size // stride, dtype=np.int64)
    return (size - np.arange(stride) + stride - 1) // stride


def _layer_bounds(size: int, layer: Layer, drop_remainder: bool) -> tuple[np.ndarray, np.ndarray]:
    """``(lengths, consumed)`` of a layer's stripes over a level of ``size`` items, clamped."""
    stride, consumed = layer
    lengths = _stripe_lengths(size, stride, drop_remainder)
    return lengths, np.minimum(np.asarray(consumed, dtype=np.int64), lengths)


def _level_sizes(total: int, layers: Sequence[Layer], drop_remainder: bool) -> list[int]:
    """The size of each level, starting with ``total``: what each layer leaves of the last."""
    sizes = [total]
    for layer in layers:
        lengths, consumed = _layer_bounds(sizes[-1], layer, drop_remainder)
        sizes.append(int((lengths - consumed).sum()))
    return sizes


def _survivors_before(x: int, stride: int, lengths: np.ndarray, consumed: np.ndarray) -> int:
    """How many level indices below ``x`` survive a layer: ``x``'s index at the next level."""
    before = (x - np.arange(stride) + stride - 1) // stride  # stripe items below x
    return int(np.maximum(np.minimum(before, lengths) - consumed, 0).sum())


class ShardedProteinDataset(IterableDataset[dict[str, object]]):
    """Iterable dataset over one or more parquet shards of protein sequences.

    Handles a single parquet file or a directory of shards, loading one shard at
    a time to bound memory. Shuffling is deterministic per ``(seed, epoch)`` and
    identical across runs and ranks; rank/worker striping is explicit (over the
    joint ``(rank, worker)`` index) so coverage does not depend on launcher
    behavior. See docs/data.md, Dataset formats.

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
        drop_remainder: Give every ``(rank, worker)`` stream the same row count,
            ``total_rows // (world_size * num_workers)``, dropping the epoch's last
            few rows (fewer than one per stream) instead of handing some streams one
            extra row. Uneven streams make ranks finish an epoch at different steps.

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
        drop_remainder: bool = False,
    ) -> None:
        super().__init__()
        self._path = Path(path)
        self._shuffle_shards = shuffle_shards
        self._shuffle_rows = shuffle_rows
        self._seed = seed
        self._load_masking_weights = load_masking_weights
        self._drop_remainder = drop_remainder
        self._epoch = 0

        # Armed resume skip (Task 3.1): plain instance state, so it pickles into
        # DataLoader worker processes along with the dataset. `None` means
        # unarmed (`__iter__` behaves exactly as before). Set via
        # `set_resume_skip`, resolved per-worker at iteration time by
        # `_resolved_skip`.
        self._resume_batches_in_epoch: int | None = None
        self._resume_per_rank_batch = 0
        self._resume_num_workers = 1
        # Token-mode resume skip: rows already emitted, one entry per DataLoader worker.
        self._resume_rows: list[int] | None = None
        # Reshard events (layout changes) to replay; persist across epochs and resume skips.
        self._reshards: list[Reshard] = []

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
        on each epoch boundary (docs/data.md, Dataset formats).

        Args:
            epoch: Epoch index (0-based).
        """
        self._epoch = epoch

    def set_reshards(self, reshards: Sequence[Reshard]) -> None:
        """Replay these reshard events in every later epoch they apply to.

        Persists across epochs and is not cleared by :meth:`clear_resume_skip`.
        """
        self._reshards = list(reshards)

    def _layers(self, layers: Sequence[Layer] | None) -> Sequence[Layer]:
        return _resolve_layers(self._reshards, self._epoch) if layers is None else layers

    def stream_length(self, layers: Sequence[Layer] | None = None) -> int:
        """Return the number of rows this (rank, worker) stream serves in one epoch (or pass).

        The live stride's share of the last level left by ``layers`` (``None``: this epoch's
        reshard events); with no layers, ``len(range(joint_index, total_rows, stride))``.
        Order-invariant, so independent of shuffling. Must be called in-context (inside a
        DataLoader worker, or single-process): it resolves the current ``(rank, worker)`` via
        :func:`_joint_stripe`.
        """
        joint_index, stride = _joint_stripe()
        size = _level_sizes(self._total_rows, self._layers(layers), self._drop_remainder)[-1]
        if self._drop_remainder:
            return size // stride
        return len(range(joint_index, size, stride))

    def set_resume_skip(self, batches_in_epoch: int, per_rank_batch: int, num_workers: int) -> None:
        """Arm a one-epoch skip so the next ``__iter__`` (in every worker) resumes mid-epoch.

        Stored as plain instance attributes, which pickle into DataLoader worker
        processes along with the dataset — that is how the skip reaches them. Each
        worker resolves its own share of ``batches_in_epoch`` at iteration time
        (see :meth:`_resolved_skip`), using its own ``worker_id``. The skip applies
        to the very next epoch iterated (usually the epoch it was captured in);
        call :meth:`clear_resume_skip` once it has been consumed so later epochs
        are unaffected.

        Args:
            batches_in_epoch: Count of batches this rank's DataLoader has already
                consumed this epoch (from the interrupted run's cursor).
            per_rank_batch: Samples per batch contributed by one (rank, worker)
                stream (the per-device batch size).
            num_workers: DataLoader ``num_workers`` the interrupted run used. Must
                match the resuming run's actual ``num_workers`` for the skip
                arithmetic to be correct; validated by the caller (Task 3.3's
                layout guard), not here.

        Note:
            With ``num_workers > 1``, a resumed ``DataLoader`` always restarts its
            worker round-robin at worker 0 (a fresh iterator/process has no memory
            of which worker was "next"). Each worker's own substream still resumes
            with no row lost or duplicated regardless of where the interruption
            landed, but the *global* interleaving order across workers only
            reproduces the uninterrupted run's batch order bit-for-bit when
            ``batches_in_epoch % num_workers == 0`` (the interruption landed on a
            full worker-cycle boundary). Off that boundary, data content is still
            exact; only which worker's batch appears at which global position can
            shift.
        """
        self._resume_batches_in_epoch = batches_in_epoch
        self._resume_per_rank_batch = per_rank_batch
        self._resume_num_workers = num_workers

    def set_resume_rows(self, rows_per_worker: Sequence[int]) -> None:
        """Arm a one-epoch skip of ``rows_per_worker[worker_id]`` rows in each worker's stream.

        The token-mode counterpart of :meth:`set_resume_skip`: batches hold a varying number of
        rows, so the cursor records rows per worker directly. Pickles into workers the same way;
        cleared by :meth:`clear_resume_skip`.
        """
        self._resume_rows = list(rows_per_worker)

    def clear_resume_skip(self) -> None:
        """Disarm both resume skips; the next ``__iter__`` starts each stream at row 0."""
        self._resume_batches_in_epoch = None
        self._resume_rows = None

    def _resolved_skip(self) -> int:
        """Resolve the armed batch-count (or per-worker row-count) skip into a sample-offset skip.

        Must be called in-context (inside a DataLoader worker, or single-process):
        delegates to :func:`_resolve_sample_skip`, which uses the current worker's
        own ``worker_id`` (via :func:`_resolve_distributed_context`) so each worker
        computes its own share of the globally-armed ``batches_in_epoch``, matching
        DataLoader's round-robin batch assignment across workers. Returns ``0`` when
        unarmed.
        """
        if self._resume_rows is not None:
            return _resolve_rows_skip(self._resume_rows, self.stream_length())
        return _resolve_sample_skip(
            self._resume_batches_in_epoch,
            self._resume_per_rank_batch,
            self._resume_num_workers,
            self.stream_length(),
        )

    def _shard_order(self, epoch_seed: int) -> list[int]:
        """Return shard indices in this epoch's (optionally shuffled) order."""
        n = len(self._shards)
        if not self._shuffle_shards:
            return list(range(n))
        generator = torch.Generator()
        generator.manual_seed(epoch_seed)
        return torch.randperm(n, generator=generator).tolist()

    def _row_order(self, n_rows: int, epoch_seed: int, shard_idx: int) -> np.ndarray:
        """Return row indices for one shard in this epoch's (optional) row order."""
        if not self._shuffle_rows:
            return np.arange(n_rows)
        generator = torch.Generator()
        generator.manual_seed(_shard_row_seed(epoch_seed, shard_idx))
        return torch.randperm(n_rows, generator=generator).numpy()

    def __iter__(self) -> Iterator[dict[str, object]]:
        # Straight to the stream, unlike a mixed source's per-pass arming: a skip equal to the
        # stream length (checkpoint taken on the epoch boundary) must stay an empty stream,
        # not wrap to the next pass.
        yield from self._iter_stream(self._resolved_skip())

    def _iter_pass(
        self, pass_index: int, skip: int = 0, layers: Sequence[Layer] | None = None
    ) -> Iterator[dict[str, object]]:
        """This stream's ``pass_index``-th pass of the epoch, in that pass's own order.

        A source sampled above its row share in an :class:`InterleavedDataset` runs
        out mid-epoch and is read again; each pass gets a fresh shard and row order
        (pass 0 is the epoch's ordinary order), so repeats do not replay one sequence.
        ``layers`` are the pass's reshard layers (``None``: this epoch's own events).
        """
        return self._iter_stream(skip, pass_index=pass_index, layers=layers)

    def _iter_stream(
        self, skip: int, *, pass_index: int = 0, layers: Sequence[Layer] | None = None
    ) -> Iterator[dict[str, object]]:
        """Yield this (rank, worker) stream, skipping its first ``skip`` selected rows.

        Each shard holds a contiguous run of global positions. Every layer removes the
        positions its old streams consumed (and, with ``drop_remainder``, its tail); survivors
        keep their order, so a shard's survivors are again a contiguous run at the next level.
        At the last level a row is this stream's iff its index is ``joint_index`` mod the live
        stride, up to :meth:`stream_length`. Unioned over every stream and layer this covers
        each row exactly once. Selection is index arithmetic only (no ``pq.read_table``), so a
        shard fully consumed by ``skip`` is never read. ``pass_index`` selects the order of a
        refilled pass. With no layers this is the pre-layer stream, byte for byte.
        """
        epoch_seed = _pass_seed(_epoch_seed(self._seed, self._epoch), pass_index)
        joint_index, stride = _joint_stripe()
        layers = self._layers(layers)
        limit = self.stream_length(layers)
        sizes = _level_sizes(self._total_rows, layers, self._drop_remainder)
        bounds = [
            _layer_bounds(size, layer, self._drop_remainder)
            for size, layer in zip(sizes[:-1], layers, strict=True)
        ]
        consumed = 0  # selected rows passed so far, skipped or served

        global_idx = 0
        remaining_skip = skip
        for shard_idx in self._shard_order(epoch_seed):
            if consumed >= limit:
                return  # this stream has served its share
            n_rows = self._rows_per_shard[shard_idx]
            base = global_idx
            global_idx += n_rows

            offsets = np.arange(n_rows)  # shard offsets still in play
            start = base  # their first index at the current level
            for (layer_stride, _), (lengths, used) in zip(layers, bounds, strict=True):
                idx = np.arange(start, start + len(offsets))
                quot = idx // layer_stride
                rem = idx % layer_stride
                offsets = offsets[(quot >= used[rem]) & (quot < lengths[rem])]
                start = _survivors_before(start, layer_stride, lengths, used)
            k = np.arange(start, start + len(offsets))
            offsets = offsets[(k % stride == joint_index) & (k // stride < limit)]

            shard_count = len(offsets)
            if shard_count == 0:
                continue  # no rows in this shard belong to this (rank, worker)
            consumed += shard_count

            if remaining_skip >= shard_count:
                remaining_skip -= shard_count
                continue  # fully skipped by arithmetic — no read_table call

            selected = self._row_order(n_rows, epoch_seed, shard_idx)[offsets[remaining_skip:]]
            remaining_skip = 0
            yield from self._read_rows(shard_idx, selected.tolist())

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

# Row key of an `InterleavedDataset` row's stream position: `(draws_done, ((pass, offset), ...))`
# after the draw that served it, the offset counting this stream's rows of the source's current
# (layered) pass. The token batcher pops it into its batch's `stream_position`.
POSITION_KEY = "_stream_position"


class InterleavedDataset(IterableDataset[dict[str, object]]):
    """Interleave several iterable datasets by sampling fraction.

    Each step picks a source by its (normalized) fraction and pulls that source's
    next item; an exhausted source is re-initialized so sources of unequal size
    keep mixing at the requested ratio for the whole epoch. Sub-datasets perform
    their own ``(rank, worker)`` striping, so this class strides only the *number
    of steps* per worker. See docs/data.md, Dataset formats.

    Every row carries its stream position under :data:`POSITION_KEY`: ``(draws_done,
    ((pass, offset), ...))``, this stream's draws so far and, per source, its pass and the rows
    this stream has served of that pass. A checkpoint that stores it resumes exactly at any
    layout (:meth:`set_resume_sources`, :meth:`set_reshards`).

    Args:
        datasets: Source iterable datasets (typically :class:`ShardedProteinDataset`).
        fractions: Per-source sampling weights; normalized to sum to 1.0.
        num_samples: Nominal samples per (full) epoch. Defaults to the sum of the
            sources' ``len()`` when available, else ``0``.
        seed: Base seed for deterministic source selection.
        reshuffle_on_refill: Read each refill of an exhausted source in a fresh order
            (sources that support it, e.g. :class:`ShardedProteinDataset`). Off, every
            pass within an epoch replays the epoch's order, so an oversampled source is
            seen as the same sequence of rows again and again.
        drop_remainder: Give every ``(rank, worker)`` stream the same number of draws,
            ``num_samples // (world_size * num_workers)``, so all ranks run the same
            number of batches per epoch.

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
        reshuffle_on_refill: bool = False,
        drop_remainder: bool = False,
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
        self._reshuffle_on_refill = reshuffle_on_refill
        self._drop_remainder = drop_remainder
        self._epoch = 0
        self._num_samples = self._default_num_samples() if num_samples is None else int(num_samples)

        # Armed resume skip (Task 3.2): mirrors ShardedProteinDataset's plain
        # instance-attribute state (Task 3.1), so it pickles into DataLoader worker
        # processes along with the dataset. `None` means unarmed. Resolved per-worker,
        # per-source at iteration time in `__iter__` — see `_arm_source_skip`.
        self._resume_batches_in_epoch: int | None = None
        self._resume_per_rank_batch = 0
        self._resume_num_workers = 1
        # Token-mode resume skip: rows already emitted, one entry per DataLoader worker.
        self._resume_rows: list[int] | None = None
        # Per worker, per source `[pass, offset]` (from POSITION_KEY); with it, `_resume_rows`
        # counts draws.
        self._resume_sources: list[Sequence[Sequence[int]] | None] | None = None
        # Reshard events (layout changes) to replay; persist across epochs and resume skips.
        self._reshards: list[Reshard] = []

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

    def set_reshards(self, reshards: Sequence[Reshard]) -> None:
        """Replay these reshard events in every later epoch they apply to.

        Each event layers the epoch's draw budget (its ``rows`` are draws) and every source pass
        (its ``sources``). Persists across epochs and is not cleared by :meth:`clear_resume_skip`.
        """
        self._reshards = list(reshards)

    def set_resume_skip(self, batches_in_epoch: int, per_rank_batch: int, num_workers: int) -> None:
        """Arm a one-epoch skip so the next ``__iter__`` (in every worker) resumes mid-epoch.

        Same contract as :meth:`ShardedProteinDataset.set_resume_skip`: stored as plain
        instance attributes (pickled into DataLoader worker processes), resolved
        per-worker at iteration time. Here, the resolved skip additionally determines
        *which* of this worker's already-drawn source ``choices`` were already
        consumed, so each source can be armed with its own sample-count skip — see
        :meth:`_arm_source_skip`. Call :meth:`clear_resume_skip` once consumed so later
        epochs are unaffected.

        Data-exact resume (skipping exactly the already-consumed rows, no loss or
        duplication) holds only for sources that expose the internal sample-skip path
        — currently :class:`ShardedProteinDataset`. Any other ``IterableDataset``
        source instead restarts from position 0 on resume (a logged warning, not an
        error); see :meth:`_arm_source_skip`.

        Args:
            batches_in_epoch: Count of batches this rank's DataLoader has already
                consumed this epoch (from the interrupted run's cursor).
            per_rank_batch: Samples per batch contributed by one (rank, worker) stream.
            num_workers: DataLoader ``num_workers`` the interrupted run used.
        """
        self._resume_batches_in_epoch = batches_in_epoch
        self._resume_per_rank_batch = per_rank_batch
        self._resume_num_workers = num_workers

    def set_resume_rows(self, rows_per_worker: Sequence[int]) -> None:
        """Arm a one-epoch skip of ``rows_per_worker[worker_id]`` rows in each worker's stream.

        The token-mode counterpart of :meth:`set_resume_skip`. A worker's rows are draws from
        the mix, so the count resolves through :meth:`_arm_source_skip` exactly like the
        batch-count skip. Cleared by :meth:`clear_resume_skip`.
        """
        self._resume_rows = list(rows_per_worker)

    def set_resume_sources(self, sources: Sequence[Sequence[Sequence[int]] | None]) -> None:
        """Arm each worker's per-source ``[pass, offset]`` (its last row's :data:`POSITION_KEY`).

        Armed together with :meth:`set_resume_rows`, whose counts then mean draws: each worker
        skips its draws and starts every source exactly where its tag says, so the resume is
        exact even inside a resharded epoch. A ``None`` entry is a worker that had drawn
        nothing. Cleared by :meth:`clear_resume_skip`.
        """
        self._resume_sources = list(sources)

    def clear_resume_skip(self) -> None:
        """Disarm every resume skip; the next ``__iter__`` starts each stream at step 0.

        Since per-source sample skips are computed fresh inside ``__iter__`` from the
        (now-unarmed) skip rather than stored on the sub-datasets, this also implicitly
        clears any per-source arming — there is nothing further to undo.
        """
        self._resume_batches_in_epoch = None
        self._resume_rows = None
        self._resume_sources = None

    def _arm_source_skip(
        self, source_idx: int, n_samples: int
    ) -> tuple[Iterator[dict[str, object]], int, int]:
        """Return ``(iterator, pass_index, offset)`` for source ``source_idx``, ``n_samples`` in.

        ``n_samples`` counts this worker's draws of the source before the interruption,
        across however many passes (refills) it had already made this epoch. With
        ``reshuffle_on_refill`` each pass has its own order, so the source resumes at pass
        ``n_samples // stream_length``, offset ``n_samples % stream_length``, and later
        refills continue from the following pass; without it every pass is the epoch's
        order and only the offset matters (the returned ``pass_index`` still counts passes,
        for the position tag). Applies to the first iterator only:
        :meth:`_next_or_refill` starts later passes unskipped. Sources without
        ``_iter_pass`` (anything but :class:`ShardedProteinDataset`) restart from position
        0 instead, logged via :data:`logger`, since data-exact resume is unavailable there.

        Args:
            source_idx: Index into ``self._datasets``.
            n_samples: Number of this source's samples already consumed this epoch.
        """
        source = self._datasets[source_idx]
        iter_pass = getattr(source, "_iter_pass", None)
        if callable(iter_pass):
            length = source.stream_length()  # ty: ignore[unresolved-attribute]  # duck-typed with _iter_pass
            pass_index, offset = divmod(n_samples, length) if length > 0 else (0, 0)
            order = pass_index if self._reshuffle_on_refill else 0
            return iter_pass(order, offset), pass_index, offset
        if n_samples > 0:
            logger.warning(
                "InterleavedDataset resume skip: source %d (%s) has no sample-skip "
                "support; data-exact resume is unavailable for it, restarting this "
                "source from position 0 instead of skipping %d already-consumed sample(s).",
                source_idx,
                type(source).__name__,
                n_samples,
            )
        return iter(source), 0, 0

    def _open(
        self, source_idx: int, pass_index: int, offset: int = 0
    ) -> Iterator[dict[str, object]]:
        """Source ``source_idx``'s ``pass_index``-th pass, ``offset`` rows in, with its layers.

        The pass is read in its own order when ``reshuffle_on_refill`` is set (else the epoch's)
        and with the reshard layers of that pass. Sources without ``_iter_pass`` (anything but
        :class:`ShardedProteinDataset`) just restart.
        """
        source = self._datasets[source_idx]
        iter_pass = getattr(source, "_iter_pass", None)
        if not callable(iter_pass):
            return iter(source)
        layers = _resolve_layers(
            self._reshards, self._epoch, source=source_idx, pass_index=pass_index
        )
        return iter_pass(pass_index if self._reshuffle_on_refill else 0, offset, layers)

    def _start_pass(self, source_idx: int) -> int:
        """The first pass of source ``source_idx`` the reshard layers left any rows of."""
        source = self._datasets[source_idx]
        last = 1 + max(
            (
                positions[source_idx][0]
                for reshard in self._reshards
                if min(reshard.epochs) <= self._epoch <= max(reshard.epochs)
                for positions in reshard.sources or ()
                if positions is not None
            ),
            default=-1,
        )
        for pass_index in range(last):
            layers = _resolve_layers(
                self._reshards, self._epoch, source=source_idx, pass_index=pass_index
            )
            total = source.total_length  # ty: ignore[unresolved-attribute]  # ShardedProteinDataset
            drop = source._drop_remainder  # ty: ignore[unresolved-attribute]
            if _level_sizes(total, layers, drop)[-1] > 0:
                return pass_index
        return last

    def __iter__(self) -> Iterator[dict[str, object]]:
        joint_index, stride = _joint_stripe()
        layers = _resolve_layers(self._reshards, self._epoch)
        # The draw budget: what the old streams' draws left of `num_samples`.
        budget = _level_sizes(self._num_samples, layers, self._drop_remainder)[-1]
        if budget <= 0 or joint_index >= budget:
            return

        # One generator per (epoch, rank, worker) so each worker draws an
        # independent source sequence while staying reproducible.
        generator = torch.Generator()
        generator.manual_seed((_epoch_seed(self._seed, self._epoch) + joint_index) & _MASK32)

        if self._drop_remainder:
            n_steps = budget // stride
        else:
            n_steps = len(range(joint_index, budget, stride))
        if n_steps <= 0:
            return
        weights = torch.tensor(self._fractions, dtype=torch.float64)
        choices = torch.multinomial(weights, n_steps, replacement=True, generator=generator)

        n_sources = len(self._datasets)
        if self._resume_sources is not None:
            # Exact per-source positions: skip the draws, start each source where its tag says.
            if self._resume_rows is None:
                raise ValueError("set_resume_sources needs set_resume_rows (the draw counts)")
            skip = _resolve_rows_skip(self._resume_rows, n_steps)
            _, _, worker_id, _ = _resolve_distributed_context()
            tag = self._resume_sources[worker_id]
            if tag is None:  # this worker had drawn nothing
                tag = [(self._start_pass(idx), 0) for idx in range(n_sources)]
            passes = [int(pass_index) for pass_index, _ in tag]
            offsets = [int(offset) for _, offset in tag]
            iters = [self._open(idx, passes[idx], offsets[idx]) for idx in range(n_sources)]
        else:
            if self._resume_rows is not None:
                skip = _resolve_rows_skip(self._resume_rows, n_steps)
            else:
                skip = _resolve_sample_skip(
                    self._resume_batches_in_epoch,
                    self._resume_per_rank_batch,
                    self._resume_num_workers,
                    n_steps,
                )
            armed_skip = self._resume_rows is not None or self._resume_batches_in_epoch is not None
            if layers and armed_skip:
                raise ValueError(
                    "resuming a resharded mix needs per-source positions (set_resume_sources); "
                    "draw counts alone cannot place each source inside a layered pass"
                )
            if skip:
                # Per-source draw counts among the choices already consumed before the
                # interruption; each count is a sample count for that source alone, so
                # `_arm_source_skip` reduces it modulo that source's own stream length.
                counts = torch.bincount(choices[:skip], minlength=n_sources)
                armed = [
                    self._arm_source_skip(idx, int(counts[idx].item())) for idx in range(n_sources)
                ]
                iters = [it for it, _, _ in armed]
                passes = [pass_index for _, pass_index, _ in armed]
                offsets = [offset for _, _, offset in armed]
            else:
                passes = [self._start_pass(idx) for idx in range(n_sources)]
                offsets = [0] * n_sources
                iters = [self._open(idx, passes[idx]) for idx in range(n_sources)]

        draws = skip
        for source_idx in choices[skip:].tolist():
            pass_before = passes[source_idx]
            item = self._next_or_refill(iters, passes, source_idx)
            draws += 1
            if passes[source_idx] != pass_before:
                offsets[source_idx] = 0  # refilled: the new pass's rows count from 0
            if item is _EXHAUSTED:
                continue  # source produced nothing for this (rank, worker)
            offsets[source_idx] += 1
            item[POSITION_KEY] = (draws, tuple(zip(passes, offsets, strict=True)))  # ty: ignore[invalid-assignment]  # a row dict
            yield item  # ty: ignore[invalid-yield]  # _EXHAUSTED sentinel filtered above

    def _next_or_refill(
        self, iters: list[Iterator[dict[str, object]]], passes: list[int], source_idx: int
    ) -> object:
        """Pull the next item from a source, starting its next pass once if exhausted.

        Returns :data:`_EXHAUSTED` if the source yields nothing even after a fresh
        pass (e.g. it serves no rows to this ``(rank, worker)``), avoiding a
        ``StopIteration`` escaping the generator (PEP 479).
        """
        try:
            return next(iters[source_idx])
        except StopIteration:
            passes[source_idx] += 1
            iters[source_idx] = self._open(source_idx, passes[source_idx])
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
