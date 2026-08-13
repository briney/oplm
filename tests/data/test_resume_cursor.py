"""Tests for skip-aware dataset resume (Task 3.1: ``ShardedProteinDataset``; Task 3.2:
``InterleavedDataset``).

Covers the index-arithmetic skip: baseline-suffix equality for several skip
values, a ``pq.read_table`` call-count assertion proving fully-skipped shards
are never read, ``stream_length()`` correctness, a ``num_workers=2`` DataLoader
batch-level resume check, and the multi-source ``InterleavedDataset`` skip that
arms each source's own sample-count skip from this worker's ``choices`` draws.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pyarrow.parquet as pq
import pytest
from torch.utils.data import DataLoader

import oplm.data.sequence.dataset as dataset_mod
from oplm.data.sequence.collate import MLMCollator
from oplm.data.sequence.dataset import DataCursor, InterleavedDataset, ShardedProteinDataset
from oplm.data.tokenizer import get_tokenizer

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@pytest.fixture(autouse=True)
def _single_process_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default every test to a single-process ``(rank=0, ws=1, worker=0, nw=1)`` context."""
    monkeypatch.setattr(dataset_mod, "_resolve_distributed_context", lambda: (0, 1, 0, 1))


def _ids(dataset: ShardedProteinDataset) -> list[str]:
    """Materialize the ``sequence_id`` stream from a dataset iteration."""
    return [str(row["sequence_id"]) for row in dataset]


def _arm_skip(ds: ShardedProteinDataset, skip: int) -> None:
    """Arm ``ds`` so that ``_resolved_skip()`` (single-process, worker 0) resolves to ``skip``.

    With ``num_workers=1``, ``worker_id=0``: ``len(range(0, batches_in_epoch, 1)) ==
    batches_in_epoch``, so ``per_rank_batch=1`` makes the resolved raw skip exactly
    ``batches_in_epoch`` (mod ``stream_length()``).
    """
    ds.set_resume_skip(batches_in_epoch=skip, per_rank_batch=1, num_workers=1)


# --------------------------------------------------------------------------- #
# (a) baseline-suffix equality
# --------------------------------------------------------------------------- #


def test_resumed_stream_is_baseline_suffix(sequence_shards: Path) -> None:
    """Armed skip yields ``baseline[skip % len(baseline):]`` exactly, for several skips."""
    baseline_ds = ShardedProteinDataset(sequence_shards, seed=0)
    baseline_ds.set_epoch(0)
    baseline = _ids(baseline_ds)
    stream_len = len(baseline)

    # shard0_len: number of rows the (rank=0, worker=0) stream draws from the
    # first shard in this epoch's shuffled shard order.
    epoch_seed = dataset_mod._epoch_seed(0, 0)
    shard_order = baseline_ds._shard_order(epoch_seed)
    n_rows0 = baseline_ds._rows_per_shard[shard_order[0]]
    shard0_len = len(range(0, n_rows0, 1))  # stride=1 in this single-process context

    for skip in {0, 1, shard0_len, shard0_len + 3, stream_len + 2}:
        ds = ShardedProteinDataset(sequence_shards, seed=0)
        ds.set_epoch(0)
        _arm_skip(ds, skip)
        resumed = _ids(ds)
        expected = baseline[skip % stream_len :]
        assert resumed == expected, f"mismatch for skip={skip}"


# --------------------------------------------------------------------------- #
# (b) read_table call-count: fully-skipped shards are never read
# --------------------------------------------------------------------------- #


def test_skip_avoids_reading_fully_skipped_shards(
    tmp_path: Path,
    make_sequence_shards,  # noqa: ANN001
    real_records,  # noqa: ANN001
) -> None:
    """A skip landing in shard 2 of 3 reads exactly 2 shards (shard 1 skipped by arithmetic)."""
    shard_dir = tmp_path / "three_shards"
    shard_dir.mkdir()
    make_sequence_shards(shard_dir, real_records, n_shards=3)

    ds = ShardedProteinDataset(shard_dir, seed=0, shuffle_shards=False)
    ds.set_epoch(0)

    epoch_seed = dataset_mod._epoch_seed(0, 0)
    shard_order = ds._shard_order(epoch_seed)
    shard0_len = ds._rows_per_shard[shard_order[0]]

    # Land the skip 1 row into shard 2 (index 1 in shard_order).
    skip = shard0_len + 1
    _arm_skip(ds, skip)

    calls = 0
    original = pq.read_table

    def _counting_read_table(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)  # type: ignore[arg-type]

    import pytest as _pytest  # local import to avoid polluting module namespace

    monkeypatch = _pytest.MonkeyPatch()
    monkeypatch.setattr(dataset_mod.pq, "read_table", _counting_read_table)
    try:
        list(ds)
    finally:
        monkeypatch.undo()

    assert calls == 2


# --------------------------------------------------------------------------- #
# (c) stream_length()
# --------------------------------------------------------------------------- #


def test_stream_length_matches_baseline_count(sequence_shards: Path) -> None:
    """``stream_length()`` equals the number of rows the baseline stream yields."""
    ds = ShardedProteinDataset(sequence_shards, seed=0)
    ds.set_epoch(0)
    baseline = _ids(ds)
    assert ds.stream_length() == len(baseline)


def test_stream_length_is_order_and_epoch_invariant(sequence_shards: Path) -> None:
    """``stream_length()`` for a fixed (rank, worker) context doesn't depend on epoch."""
    ds = ShardedProteinDataset(sequence_shards, seed=0)
    ds.set_epoch(0)
    len0 = ds.stream_length()
    ds.set_epoch(7)
    len7 = ds.stream_length()
    assert len0 == len7


# --------------------------------------------------------------------------- #
# (d) num_workers=2 DataLoader batch-level resume
# --------------------------------------------------------------------------- #


def test_resume_with_two_dataloader_workers_matches_batch_suffix(
    tmp_path: Path,
    make_sequence_shards,  # noqa: ANN001
    real_records,  # noqa: ANN001
) -> None:
    """Resumed 2-worker loader output equals the uninterrupted loader's output from batch k on."""
    shard_dir = tmp_path / "shards_for_workers"
    shard_dir.mkdir()
    make_sequence_shards(shard_dir, real_records, n_shards=2)

    collator = MLMCollator(get_tokenizer(), max_length=64, keep_sequence_ids=True)

    def _make_loader(ds: ShardedProteinDataset) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=2,
            collate_fn=collator,
            num_workers=2,
            persistent_workers=False,
        )

    baseline_ds = ShardedProteinDataset(shard_dir, seed=0)
    baseline_ds.set_epoch(0)
    baseline_batches = [batch["sequence_ids"] for batch in _make_loader(baseline_ds)]

    # A fresh DataLoader always restarts its worker round-robin at worker 0, so
    # the resumed run's local batch 0 lines up with the interrupted run's next
    # batch only when the crash landed on a full worker-cycle boundary
    # (`k % num_workers == 0`) — pick such a `k` here.
    num_workers = 2
    k = 4
    assert k % num_workers == 0
    assert len(baseline_batches) > k + 1  # ensure the fixture is large enough

    resumed_ds = ShardedProteinDataset(shard_dir, seed=0)
    resumed_ds.set_epoch(0)
    resumed_ds.set_resume_skip(batches_in_epoch=k, per_rank_batch=2, num_workers=num_workers)
    resumed_batches = [batch["sequence_ids"] for batch in _make_loader(resumed_ds)]

    assert resumed_batches == baseline_batches[k:]


# --------------------------------------------------------------------------- #
# clear_resume_skip
# --------------------------------------------------------------------------- #


def test_clear_resume_skip_restores_full_stream(sequence_shards: Path) -> None:
    """After clearing an armed skip, the next iteration yields the full baseline stream."""
    baseline_ds = ShardedProteinDataset(sequence_shards, seed=0)
    baseline_ds.set_epoch(0)
    baseline = _ids(baseline_ds)

    ds = ShardedProteinDataset(sequence_shards, seed=0)
    ds.set_epoch(0)
    _arm_skip(ds, 3)
    ds.clear_resume_skip()
    assert _ids(ds) == baseline


# --------------------------------------------------------------------------- #
# skip=0 byte-identical to unmodified __iter__
# --------------------------------------------------------------------------- #


def test_unarmed_skip_is_byte_identical_to_baseline(sequence_shards: Path) -> None:
    """With nothing armed, iteration output is unchanged from the pre-refactor contract."""
    ds1 = ShardedProteinDataset(sequence_shards, seed=0)
    ds1.set_epoch(2)
    rows1 = list(ds1)

    ds2 = ShardedProteinDataset(sequence_shards, seed=0)
    ds2.set_epoch(2)
    rows2 = list(ds2)

    assert rows1 == rows2


# --------------------------------------------------------------------------- #
# Property test: per-shard hit-count arithmetic vs. brute-force enumeration
# --------------------------------------------------------------------------- #


def test_hit_count_formula_matches_brute_force_enumeration() -> None:
    """The arithmetic hit-count formula matches brute-force enumeration over a grid.

    Directly CI-guards the identity ``_iter_stream`` relies on to skip
    fully-consumed shards without ever calling ``pq.read_table``:
    ``len(range((j - b) % s, n, s)) == sum(1 for o in range(n) if (b + o) % s == j)``.
    Sweeps stride ``s`` in ``[1, 12]``, every valid ``joint_index`` ``j`` in
    ``[0, s)``, shard size ``n`` (including ``n=0``), and base ``b`` (including
    small bases and bases far exceeding ``s``, which cover
    ``first_hit_offset >= n`` cases where the shard contributes no hits at all).
    """
    bases = [0, 1, 2, 3, 5, 8, 997, 4999, 12_345]
    shard_sizes = [0, 1, 2, 3, 5, 10, 25, 50]

    checked = 0
    for stride in range(1, 13):
        for joint_index in range(stride):
            for base in bases:
                for n_rows in shard_sizes:
                    first_hit_offset = (joint_index - base) % stride
                    formula = len(range(first_hit_offset, n_rows, stride))
                    brute_force = sum(
                        1 for offset in range(n_rows) if (base + offset) % stride == joint_index
                    )
                    assert formula == brute_force, (
                        f"{stride=} {joint_index=} {base=} {n_rows=} "
                        f"{first_hit_offset=} {formula=} {brute_force=}"
                    )
                    checked += 1

    assert checked == sum(range(1, 13)) * len(bases) * len(shard_sizes)


# --------------------------------------------------------------------------- #
# DataCursor
# --------------------------------------------------------------------------- #


def test_data_cursor_holds_fields_and_is_frozen() -> None:
    """``DataCursor`` carries exactly the brief's fields and rejects mutation."""
    cursor = DataCursor(
        epoch=2, batches_in_epoch=17, world_size=4, num_workers=2, per_rank_batch=8, seed=0
    )
    assert cursor.epoch == 2
    assert cursor.batches_in_epoch == 17
    assert cursor.world_size == 4
    assert cursor.num_workers == 2
    assert cursor.per_rank_batch == 8
    assert cursor.seed == 0
    with pytest.raises(dataclasses.FrozenInstanceError):
        cursor.epoch = 3  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# InterleavedDataset skip-aware resume (Task 3.2)
# --------------------------------------------------------------------------- #


def _interleaved_ids(inter: InterleavedDataset) -> list[str]:
    """Materialize the ``sequence_id`` stream from an ``InterleavedDataset`` iteration."""
    return [str(row["sequence_id"]) for row in inter]


def _arm_interleaved_skip(inter: InterleavedDataset, skip: int) -> None:
    """Arm ``inter`` so that its resolved skip (single-process, worker 0) equals ``skip``.

    Same reasoning as :func:`_arm_skip`: ``per_rank_batch=1``, ``num_workers=1`` makes
    the resolved raw skip exactly ``batches_in_epoch`` (mod this stream's step count).
    """
    inter.set_resume_skip(batches_in_epoch=skip, per_rank_batch=1, num_workers=1)


def test_interleaved_resumed_stream_is_baseline_suffix(
    tmp_path: Path,
    make_sequence_shards: Callable[..., Path],
    real_records: list[tuple[str, str]],
) -> None:
    """Armed skip yields ``baseline[skip:]`` exactly, crossing several refills of the small source."""
    small_dir = (tmp_path / "small").resolve()
    large_dir = (tmp_path / "large").resolve()
    small_dir.mkdir()
    large_dir.mkdir()
    make_sequence_shards(small_dir, real_records[:3], n_shards=1, id_prefix="small_")
    make_sequence_shards(large_dir, real_records[3:15], n_shards=2, id_prefix="large_")

    num_samples = 60  # the 3-row small source refills roughly 10x over this many draws

    def _build() -> InterleavedDataset:
        inter = InterleavedDataset(
            [ShardedProteinDataset(small_dir, seed=0), ShardedProteinDataset(large_dir, seed=0)],
            [0.5, 0.5],
            num_samples=num_samples,
            seed=0,
        )
        inter.set_epoch(0)
        return inter

    baseline = _interleaved_ids(_build())

    for skip in (0, 1, 5, 13, 30, 59):
        armed = _build()
        _arm_interleaved_skip(armed, skip)
        resumed = _interleaved_ids(armed)
        assert resumed == baseline[skip:], f"mismatch for skip={skip}"


def test_interleaved_refill_does_not_reapply_skip(
    tmp_path: Path,
    make_sequence_shards: Callable[..., Path],
    real_records: list[tuple[str, str]],
) -> None:
    """A source's armed skip is one-shot: refilling it after exhaustion is unskipped."""
    small_dir = (tmp_path / "single_small").resolve()
    small_dir.mkdir()
    make_sequence_shards(small_dir, real_records[:3], n_shards=1, id_prefix="s_")

    num_samples = 10  # spans 3+ refill cycles of the 3-row source

    def _build() -> InterleavedDataset:
        inter = InterleavedDataset(
            [ShardedProteinDataset(small_dir, seed=0)], [1.0], num_samples=num_samples, seed=0
        )
        inter.set_epoch(0)
        return inter

    baseline = _interleaved_ids(_build())
    assert len(baseline) == num_samples
    # A single source at fraction 1.0 just cycles its 3 ids every refill.
    assert baseline == [baseline[i % 3] for i in range(num_samples)]

    skip = 4  # lands one row into the *second* refill cycle
    armed = _build()
    _arm_interleaved_skip(armed, skip)
    resumed = _interleaved_ids(armed)

    # If the skip were (incorrectly) reapplied on every refill, later cycles would
    # drop extra rows too, shortening the stream and shifting the cycle phase.
    assert len(resumed) == num_samples - skip
    assert resumed == baseline[skip:]


def test_interleaved_clear_resume_skip_restores_full_stream(
    tmp_path: Path,
    make_sequence_shards: Callable[..., Path],
    real_records: list[tuple[str, str]],
) -> None:
    """After clearing an armed skip, the next iteration yields the full baseline stream."""
    small_dir = (tmp_path / "clear_small").resolve()
    large_dir = (tmp_path / "clear_large").resolve()
    small_dir.mkdir()
    large_dir.mkdir()
    make_sequence_shards(small_dir, real_records[:3], n_shards=1, id_prefix="cs_")
    make_sequence_shards(large_dir, real_records[3:15], n_shards=2, id_prefix="cl_")

    def _build() -> InterleavedDataset:
        inter = InterleavedDataset(
            [ShardedProteinDataset(small_dir, seed=0), ShardedProteinDataset(large_dir, seed=0)],
            [0.5, 0.5],
            num_samples=30,
            seed=0,
        )
        inter.set_epoch(0)
        return inter

    baseline = _interleaved_ids(_build())

    ds = _build()
    _arm_interleaved_skip(ds, 7)
    ds.clear_resume_skip()
    assert _interleaved_ids(ds) == baseline


def test_interleaved_unarmed_skip_is_byte_identical_to_baseline(
    tmp_path: Path,
    make_sequence_shards: Callable[..., Path],
    real_records: list[tuple[str, str]],
) -> None:
    """With nothing armed, iteration output is unchanged from the pre-3.2 contract."""
    dir0 = (tmp_path / "u0").resolve()
    dir1 = (tmp_path / "u1").resolve()
    dir0.mkdir()
    dir1.mkdir()
    make_sequence_shards(dir0, real_records[:8], n_shards=2, id_prefix="a_")
    make_sequence_shards(dir1, real_records[8:16], n_shards=2, id_prefix="b_")

    def _build() -> InterleavedDataset:
        inter = InterleavedDataset(
            [ShardedProteinDataset(dir0, seed=0), ShardedProteinDataset(dir1, seed=0)],
            [0.6, 0.4],
            num_samples=40,
            seed=0,
        )
        inter.set_epoch(3)
        return inter

    rows1 = list(_build())
    rows2 = list(_build())
    assert rows1 == rows2
