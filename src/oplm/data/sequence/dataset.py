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

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

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
    """Position of the training stream, plus the layout it is only valid under.

    Captured at checkpoint time (one per run, not per rank — in sequence mode ranks step in
    lockstep so ``batches_in_epoch`` is the same value everywhere; token mode adds the
    per-rank ``rank_epochs`` and ``worker_rows`` tables) and replayed on resume via
    :meth:`ShardedProteinDataset.set_resume_skip` (sequence mode) or ``set_resume_rows``
    (token mode). A sequence-mode cursor can resume a token-mode run (its position maps onto
    per-worker rows); a token-mode cursor cannot resume a sequence-mode run. The layout fields
    (``world_size``, ``num_workers``, ``per_rank_batch``, ``seed``) let the
    resuming run detect a changed layout before trusting ``batches_in_epoch`` — see
    Task 3.3's guard.

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
            emitted this epoch, the exact position whatever the batch size.
    """

    epoch: int
    batches_in_epoch: int
    world_size: int
    num_workers: int
    per_rank_batch: int
    seed: int
    rank_epochs: list[int] | None = None
    worker_rows: list[list[int]] | None = None


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
        on each epoch boundary (docs/data.md, Why the data is shuffled).

        Args:
            epoch: Epoch index (0-based).
        """
        self._epoch = epoch

    def stream_length(self) -> int:
        """Return the number of rows this (rank, worker) stream serves in one epoch.

        Order-invariant: shard order only redistributes which rows land in which
        shard, not which global row indices this ``(rank, worker)`` owns, so this
        equals ``len(range(joint_index, total_rows, stride))`` — independent of
        shuffling and the epoch. Must be called in-context (inside a DataLoader
        worker, or single-process): it resolves the current ``(rank, worker)``
        via :func:`_joint_stripe`.
        """
        joint_index, stride = _joint_stripe()
        if self._drop_remainder:
            return self._total_rows // stride
        return len(range(joint_index, self._total_rows, stride))

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

    def _row_order(self, n_rows: int, epoch_seed: int, shard_idx: int) -> list[int]:
        """Return row indices for one shard in this epoch's (optional) row order."""
        if not self._shuffle_rows:
            return list(range(n_rows))
        generator = torch.Generator()
        generator.manual_seed(_shard_row_seed(epoch_seed, shard_idx))
        return torch.randperm(n_rows, generator=generator).tolist()

    def __iter__(self) -> Iterator[dict[str, object]]:
        # Straight to the stream, unlike a mixed source's per-pass arming: a skip equal to the
        # stream length (checkpoint taken on the epoch boundary) must stay an empty stream,
        # not wrap to the next pass.
        yield from self._iter_stream(self._resolved_skip())

    def _iter_pass(self, pass_index: int, skip: int = 0) -> Iterator[dict[str, object]]:
        """This stream's ``pass_index``-th pass of the epoch, in that pass's own order.

        A source sampled above its row share in an :class:`InterleavedDataset` runs
        out mid-epoch and is read again; each pass gets a fresh shard and row order
        (pass 0 is the epoch's ordinary order), so repeats do not replay one sequence.
        """
        return self._iter_stream(skip, pass_index=pass_index)

    def _iter_stream(self, skip: int, *, pass_index: int = 0) -> Iterator[dict[str, object]]:
        """Yield this (rank, worker) stream, skipping its first ``skip`` selected rows.

        Walks the shard order computing each shard's selected-row COUNT from index
        arithmetic alone (no ``pq.read_table``): a shard fully consumed by
        ``skip`` is passed over without ever being read. Reading starts at the
        shard where the running count first reaches ``skip``, applying the
        remainder of the skip to that shard's (already-shuffled) selection only;
        every later shard is read in full, exactly as when unarmed. With
        ``skip=0`` this is behavior-identical to the pre-refactor ``__iter__``.
        The stream stops after :meth:`stream_length` rows (which only binds with
        ``drop_remainder``). ``pass_index`` selects the order of a refilled pass.
        """
        epoch_seed = _pass_seed(_epoch_seed(self._seed, self._epoch), pass_index)
        joint_index, stride = _joint_stripe()
        limit = self.stream_length()
        consumed = 0  # selected rows passed so far, skipped or served

        # `global_idx` runs across all shards in iteration order; a row is served
        # to this (rank, worker) iff its global index is congruent to joint_index
        # mod stride. Unioned over joint_index in [0, stride) this covers every
        # row exactly once.
        global_idx = 0
        remaining_skip = skip
        for shard_idx in self._shard_order(epoch_seed):
            n_rows = self._rows_per_shard[shard_idx]
            base = global_idx
            global_idx += n_rows

            # Smallest offset o in [0, n_rows) with (base + o) % stride == joint_index,
            # i.e. o ≡ (joint_index - base) (mod stride); the selected-row count for
            # this shard follows without materializing row_order or reading it.
            first_hit_offset = (joint_index - base) % stride
            shard_count = min(len(range(first_hit_offset, n_rows, stride)), limit - consumed)
            if consumed >= limit:
                return  # drop_remainder: this stream has served its equal share
            if shard_count == 0:
                continue  # no rows in this shard belong to this (rank, worker)
            consumed += shard_count

            if remaining_skip >= shard_count:
                remaining_skip -= shard_count
                continue  # fully skipped by arithmetic — no read_table call

            row_order = self._row_order(n_rows, epoch_seed, shard_idx)
            selected = [
                row_idx
                for offset, row_idx in enumerate(row_order)
                if (base + offset) % stride == joint_index
            ][:shard_count]
            if remaining_skip:
                selected = selected[remaining_skip:]
                remaining_skip = 0

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
    of steps* per worker. See docs/data.md, Dataset formats.

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

    def clear_resume_skip(self) -> None:
        """Disarm both resume skips; the next ``__iter__`` starts each stream at step 0.

        Since per-source sample skips are computed fresh inside ``__iter__`` from the
        (now-unarmed) skip rather than stored on the sub-datasets, this also implicitly
        clears any per-source arming — there is nothing further to undo.
        """
        self._resume_batches_in_epoch = None
        self._resume_rows = None

    def _arm_source_skip(
        self, source_idx: int, n_samples: int
    ) -> tuple[Iterator[dict[str, object]], int]:
        """Return ``(iterator, pass_index)`` for source ``source_idx``, ``n_samples`` in.

        ``n_samples`` counts this worker's draws of the source before the interruption,
        across however many passes (refills) it had already made this epoch. With
        ``reshuffle_on_refill`` each pass has its own order, so the source resumes at pass
        ``n_samples // stream_length``, offset ``n_samples % stream_length``, and later
        refills continue from the following pass; without it every pass is the epoch's
        order and only the offset matters. Applies to the first iterator only:
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
            if not self._reshuffle_on_refill:
                pass_index = 0
            return iter_pass(pass_index, offset), pass_index
        if n_samples > 0:
            logger.warning(
                "InterleavedDataset resume skip: source %d (%s) has no sample-skip "
                "support; data-exact resume is unavailable for it, restarting this "
                "source from position 0 instead of skipping %d already-consumed sample(s).",
                source_idx,
                type(source).__name__,
                n_samples,
            )
        return iter(source), 0

    def _refill(self, source_idx: int, pass_index: int) -> Iterator[dict[str, object]]:
        """A fresh pass over source ``source_idx`` (reshuffled when enabled and supported)."""
        source = self._datasets[source_idx]
        iter_pass = getattr(source, "_iter_pass", None)
        if self._reshuffle_on_refill and callable(iter_pass):
            return iter_pass(pass_index)
        return iter(source)

    def __iter__(self) -> Iterator[dict[str, object]]:
        joint_index, stride = _joint_stripe()
        if self._num_samples <= 0 or joint_index >= self._num_samples:
            return

        # One generator per (epoch, rank, worker) so each worker draws an
        # independent source sequence while staying reproducible.
        generator = torch.Generator()
        generator.manual_seed((_epoch_seed(self._seed, self._epoch) + joint_index) & _MASK32)

        if self._drop_remainder:
            n_steps = self._num_samples // stride
        else:
            n_steps = len(range(joint_index, self._num_samples, stride))
        weights = torch.tensor(self._fractions, dtype=torch.float64)
        choices = torch.multinomial(weights, n_steps, replacement=True, generator=generator)

        if self._resume_rows is not None:
            skip = _resolve_rows_skip(self._resume_rows, n_steps)
        else:
            skip = _resolve_sample_skip(
                self._resume_batches_in_epoch,
                self._resume_per_rank_batch,
                self._resume_num_workers,
                n_steps,
            )
        if skip:
            # Per-source draw counts among the choices already consumed before the
            # interruption; each count is a sample count for that source alone, so
            # `_arm_source_skip` reduces it modulo that source's own stream length.
            counts = torch.bincount(choices[:skip], minlength=len(self._datasets))
            armed = [
                self._arm_source_skip(idx, int(counts[idx].item()))
                for idx in range(len(self._datasets))
            ]
            iters = [it for it, _ in armed]
            passes = [pass_index for _, pass_index in armed]
            choices = choices[skip:]
        else:
            iters = [iter(ds) for ds in self._datasets]
            passes = [0] * len(self._datasets)

        for source_idx in choices.tolist():
            item = self._next_or_refill(iters, passes, source_idx)
            if item is _EXHAUSTED:
                continue  # source produced nothing for this (rank, worker)
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
            iters[source_idx] = self._refill(source_idx, passes[source_idx])
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
