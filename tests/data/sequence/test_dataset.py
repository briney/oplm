"""Tests for the iterable sequence datasets (Phase 3).

Covers the reproducibility contract (same ``(seed, epoch)`` ⇒ identical order),
explicit rank/worker striping (disjoint, gap-free coverage), interleaved
sampling ratios + source refill, and the optional ``masking_weights`` column.

The ``(rank, worker)`` context is normally resolved from
``torch.distributed``/env/worker-info; tests monkeypatch
:func:`oplm.data.sequence.dataset._resolve_distributed_context` to drive specific
contexts deterministically in a single process.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import oplm.data.sequence.dataset as dataset_mod
from oplm.data import DataCursor
from oplm.data.sequence.dataset import InterleavedDataset, ShardedProteinDataset
from tests.data.conftest import write_sequence_shards

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path


@pytest.fixture(autouse=True)
def _single_process_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default every test to a single-process ``(rank=0, ws=1, worker=0, nw=1)`` context."""
    monkeypatch.setattr(dataset_mod, "_resolve_distributed_context", lambda: (0, 1, 0, 1))


def _set_context(monkeypatch: pytest.MonkeyPatch, rank: int, ws: int, worker: int, nw: int) -> None:
    """Pin the resolved sharding context for the next dataset iteration."""
    monkeypatch.setattr(
        dataset_mod,
        "_resolve_distributed_context",
        lambda: (rank, ws, worker, nw),
    )


def _all_sequence_ids(shard_dir: Path) -> list[str]:
    """Ground-truth set of every ``sequence_id`` in a shard directory."""
    ids: list[str] = []
    for shard in sorted(shard_dir.iterdir()):
        ids.extend(pq.read_table(shard, columns=["sequence_id"]).column("sequence_id").to_pylist())
    return ids


def _ids(dataset: ShardedProteinDataset) -> list[str]:
    """Materialize the ``sequence_id`` stream from a dataset iteration."""
    return [str(row["sequence_id"]) for row in dataset]


# --------------------------------------------------------------------------- #
# Construction & metadata
# --------------------------------------------------------------------------- #


def test_total_length_matches_parquet(sequence_shards: Path) -> None:
    """``len`` / ``total_length`` equal the parquet row count (read from metadata)."""
    ds = ShardedProteinDataset(sequence_shards)
    expected = len(_all_sequence_ids(sequence_shards))
    assert len(ds) == expected
    assert ds.total_length == expected


def test_missing_path_raises(tmp_path: Path) -> None:
    """A path that is neither a parquet file nor a shard directory errors clearly."""
    with pytest.raises(FileNotFoundError):
        ShardedProteinDataset(tmp_path / "does_not_exist.parquet")


def test_empty_directory_raises(tmp_path: Path) -> None:
    """A directory with no parquet shards errors."""
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        ShardedProteinDataset(empty)


def test_single_file_is_one_shard(sequence_shards: Path) -> None:
    """A single parquet file resolves to a one-shard dataset."""
    shard = sorted(sequence_shards.iterdir())[0]
    ds = ShardedProteinDataset(shard)
    n = pq.ParquetFile(shard).metadata.num_rows
    assert len(ds) == n
    assert len(_ids(ds)) == n


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #


def test_same_seed_epoch_identical_order(sequence_shards: Path) -> None:
    """Two iterations at the same ``(seed, epoch)`` yield identical row order."""
    ds = ShardedProteinDataset(sequence_shards, seed=0)
    ds.set_epoch(0)
    first = _ids(ds)
    ds.set_epoch(0)
    second = _ids(ds)
    assert first == second


def test_different_epochs_differ(sequence_shards: Path) -> None:
    """Different epochs reorder the same rows (same multiset, different order)."""
    ds = ShardedProteinDataset(sequence_shards, seed=0)
    ds.set_epoch(0)
    epoch0 = _ids(ds)
    ds.set_epoch(1)
    epoch1 = _ids(ds)
    assert sorted(epoch0) == sorted(epoch1)  # same rows
    assert epoch0 != epoch1  # different order


def test_no_shuffle_is_natural_order(sequence_shards: Path) -> None:
    """With shuffling off, rows come out in shard-then-row order."""
    ds = ShardedProteinDataset(sequence_shards, shuffle_shards=False, shuffle_rows=False)
    assert _ids(ds) == _all_sequence_ids(sequence_shards)


# --------------------------------------------------------------------------- #
# Striping coverage
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("world_size", "num_workers"), [(1, 1), (1, 2), (2, 2)])
def test_striping_disjoint_and_complete(
    sequence_shards: Path, monkeypatch: pytest.MonkeyPatch, world_size: int, num_workers: int
) -> None:
    """Union of all ``(rank, worker)`` shards equals the dataset, with no duplicates."""
    full = _all_sequence_ids(sequence_shards)

    collected: list[str] = []
    for rank in range(world_size):
        for worker in range(num_workers):
            _set_context(monkeypatch, rank, world_size, worker, num_workers)
            ds = ShardedProteinDataset(sequence_shards, seed=0)
            ds.set_epoch(3)
            collected.extend(_ids(ds))

    assert len(collected) == len(set(collected))  # no duplicates
    assert sorted(collected) == sorted(full)  # complete coverage


# --------------------------------------------------------------------------- #
# masking_weights column
# --------------------------------------------------------------------------- #


def test_masking_weights_present(sequence_shards_weighted: Path) -> None:
    """When requested and present, weights surface aligned to the raw sequence."""
    ds = ShardedProteinDataset(sequence_shards_weighted, load_masking_weights=True)
    rows = list(ds)
    assert rows
    for row in rows:
        weights = row["masking_weights"]
        assert isinstance(weights, list)
        assert len(weights) == len(str(row["sequence"]))


def test_masking_weights_absent_yields_none(sequence_shards: Path) -> None:
    """Requested but column-absent shards surface ``masking_weights=None``."""
    ds = ShardedProteinDataset(sequence_shards, load_masking_weights=True)
    rows = list(ds)
    assert rows
    for row in rows:
        assert "masking_weights" in row
        assert row["masking_weights"] is None


def test_masking_weights_not_read_when_disabled(sequence_shards_weighted: Path) -> None:
    """With ``load_masking_weights=False`` the key is absent even if the column exists."""
    ds = ShardedProteinDataset(sequence_shards_weighted, load_masking_weights=False)
    for row in ds:
        assert "masking_weights" not in row


# --------------------------------------------------------------------------- #
# InterleavedDataset
# --------------------------------------------------------------------------- #


def test_interleaving_validation() -> None:
    """Constructor guards on empty/mismatched/negative/zero fractions."""
    dummy = _ListDataset([{"sequence_id": "a", "sequence": "AA"}])
    with pytest.raises(ValueError, match="at least one dataset"):
        InterleavedDataset([], [])
    with pytest.raises(ValueError, match="same length"):
        InterleavedDataset([dummy], [0.5, 0.5])
    with pytest.raises(ValueError, match="non-negative"):
        InterleavedDataset([dummy, dummy], [-1.0, 2.0])
    with pytest.raises(ValueError, match="positive"):
        InterleavedDataset([dummy, dummy], [0.0, 0.0])


def test_interleaving_ratio_matches_fractions(
    tmp_path: Path, make_sequence_shards: Callable[..., Path], real_records: list[tuple[str, str]]
) -> None:
    """Empirical source-sampling ratio over many draws matches the fractions."""
    dir0 = (tmp_path / "ds0").resolve()
    dir1 = (tmp_path / "ds1").resolve()
    dir0.mkdir()
    dir1.mkdir()
    make_sequence_shards(dir0, real_records[:8], n_shards=2, id_prefix="ds0_")
    make_sequence_shards(dir1, real_records[8:16], n_shards=2, id_prefix="ds1_")

    inter = InterleavedDataset(
        [ShardedProteinDataset(dir0, seed=0), ShardedProteinDataset(dir1, seed=0)],
        [0.75, 0.25],
        num_samples=4000,
        seed=0,
    )
    inter.set_epoch(0)

    from_ds0 = sum(1 for row in inter if str(row["sequence_id"]).startswith("ds0_"))
    total = inter.total_length
    assert total == 4000
    assert abs(from_ds0 / total - 0.75) < 0.04


def test_interleaving_refills_exhausted_source(
    tmp_path: Path, make_sequence_shards: Callable[..., Path], real_records: list[tuple[str, str]]
) -> None:
    """A small source is re-iterated so the mix keeps producing for the full epoch."""
    small = (tmp_path / "small").resolve()
    small.mkdir()
    make_sequence_shards(small, real_records[:3], n_shards=1)

    inter = InterleavedDataset(
        [ShardedProteinDataset(small, seed=0)], [1.0], num_samples=20, seed=0
    )
    inter.set_epoch(0)
    rows = list(inter)
    assert len(rows) == 20  # refilled well past the 3 underlying rows


def test_interleaving_set_epoch_propagates(
    tmp_path: Path, make_sequence_shards: Callable[..., Path], real_records: list[tuple[str, str]]
) -> None:
    """``set_epoch`` reaches sub-datasets (their order changes with the epoch)."""
    shard_dir = (tmp_path / "src").resolve()
    shard_dir.mkdir()
    make_sequence_shards(shard_dir, real_records[:12], n_shards=2)
    source = ShardedProteinDataset(shard_dir, seed=0)

    inter = InterleavedDataset([source], [1.0], num_samples=12, seed=0)
    inter.set_epoch(0)
    assert source._epoch == 0
    inter.set_epoch(5)
    assert source._epoch == 5


class _ListDataset:
    """Minimal in-memory iterable used only for constructor-validation tests."""

    def __init__(self, items: list[dict[str, str]]) -> None:
        self._items = items

    def __iter__(self) -> Iterator[dict[str, str]]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)


# --------------------------------------------------------------------------- #
# Token-mode resume: per-worker row counts
# --------------------------------------------------------------------------- #


def _rows_dataset(tmp_path: Path) -> ShardedProteinDataset:
    shard_dir = tmp_path / "rows"
    shard_dir.mkdir()
    records = [(f"{i}", "ACDEFGHIKL"[: 3 + i % 7]) for i in range(40)]
    return ShardedProteinDataset(write_sequence_shards(shard_dir, records, n_shards=3), seed=0)


def test_set_resume_rows_skips_each_workers_own_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _rows_dataset(tmp_path)
    baseline = {}
    for worker in range(2):
        _set_context(monkeypatch, 0, 1, worker, 2)
        baseline[worker] = _ids(ds)
    ds.set_resume_rows([3, 7])
    for worker, skip in enumerate((3, 7)):
        _set_context(monkeypatch, 0, 1, worker, 2)
        assert _ids(ds) == baseline[worker][skip:]
    ds.clear_resume_skip()
    _set_context(monkeypatch, 0, 1, 1, 2)
    assert _ids(ds) == baseline[1]


def test_set_resume_rows_clamps_to_the_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _rows_dataset(tmp_path)
    ds.set_resume_rows([1000, 0])
    _set_context(monkeypatch, 0, 1, 0, 2)
    assert _ids(ds) == []


def test_set_resume_rows_needs_one_entry_per_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _rows_dataset(tmp_path)
    ds.set_resume_rows([1, 2, 3])
    _set_context(monkeypatch, 0, 1, 0, 2)
    with pytest.raises(ValueError, match="3 entries for 2 DataLoader workers"):
        _ids(ds)


def test_data_cursor_loads_old_dicts() -> None:
    old = {
        "epoch": 1,
        "batches_in_epoch": 5,
        "world_size": 2,
        "num_workers": 1,
        "per_rank_batch": 4,
        "seed": 0,
    }
    cursor = DataCursor(**old)
    assert cursor.rank_epochs is None and cursor.worker_rows is None
    assert cursor.worker_sources is None and cursor.reshards == []
    assert cursor.data_fingerprint is None


def test_data_cursor_round_trips_token_fields() -> None:
    cursor = DataCursor(
        epoch=0,
        batches_in_epoch=3,
        world_size=2,
        num_workers=2,
        per_rank_batch=0,
        seed=0,
        rank_epochs=[0, 1],
        worker_rows=[[4, 5], [0, 2]],
    )
    assert DataCursor(**asdict(cursor)) == cursor


# --------------------------------------------------------------------------- #
# train_data_fingerprint
# --------------------------------------------------------------------------- #


def test_fingerprint_ignores_location(sequence_shards: Path, tmp_path: Path) -> None:
    import shutil

    from oplm.data import train_data_fingerprint

    copy = tmp_path / "copy"
    shutil.copytree(sequence_shards, copy)
    assert train_data_fingerprint(str(sequence_shards)) == train_data_fingerprint(str(copy))


def test_fingerprint_sees_content_and_fraction(tmp_path: Path) -> None:
    from oplm.data import train_data_fingerprint

    records = [(f"id{i}", "ACDEFGHIKLMNPQRSTVWY"[: 5 + i]) for i in range(10)]
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    write_sequence_shards(a, records, n_shards=1)
    write_sequence_shards(b, [*records[1:], records[0]], n_shards=1)
    assert train_data_fingerprint(str(a)) != train_data_fingerprint(str(b))
    one = train_data_fingerprint(
        {"x": {"path": str(a), "fraction": 0.5}, "y": {"path": str(b), "fraction": 0.5}}
    )
    two = train_data_fingerprint(
        {"x": {"path": str(a), "fraction": 0.25}, "y": {"path": str(b), "fraction": 0.75}}
    )
    assert one != two


# --------------------------------------------------------------------------- #
# teacher_logits column and Neff gating
# --------------------------------------------------------------------------- #


def _write_distill_shard(
    path: Path,
    rows: list[tuple[str, str, bytes | None, float | None, float | None]],
    *,
    large_binary: bool = False,
    include_neff: bool = True,
) -> Path:
    cols: dict[str, pa.Array] = {
        "sequence_id": pa.array([r[0] for r in rows], type=pa.string()),
        "sequence": pa.array([r[1] for r in rows], type=pa.string()),
        "teacher_logits": pa.array(
            [r[2] for r in rows],
            type=pa.large_binary() if large_binary else pa.binary(),
        ),
    }
    if include_neff:
        cols["teacher_neff"] = pa.array([r[3] for r in rows], type=pa.float32())
        cols["teacher_neff_sqrt_l"] = pa.array([r[4] for r in rows], type=pa.float32())
    pq.write_table(pa.table(cols), path)
    return path


def test_teacher_logits_ignored_when_disabled(tmp_path: Path) -> None:
    shard = _write_distill_shard(
        tmp_path / "s.parquet", [("a", "ACDE", b"\x00" * 160, 50.0, 25.0)]
    )
    ds = ShardedProteinDataset(shard, shuffle_shards=False, shuffle_rows=False)
    row = next(iter(ds))
    assert "teacher_logits" not in row


@pytest.mark.parametrize("large_binary", [False, True])
def test_teacher_logits_loaded_and_neff_gated(tmp_path: Path, large_binary: bool) -> None:
    payload = b"\x01" * 160
    shard = _write_distill_shard(
        tmp_path / "s.parquet",
        [
            ("pass_both", "ACDE", payload, 30.0, 5.0),
            ("pass_boundary", "ACDE", payload, 20.0, 2.0),
            ("fail_neff", "ACDE", payload, 10.0, 5.0),
            ("fail_sqrt", "ACDE", payload, 30.0, 1.5),
            ("nan_neff", "ACDE", payload, float("nan"), 5.0),
            ("nan_sqrt", "ACDE", payload, 30.0, float("nan")),
            ("none_neff", "ACDE", payload, None, 5.0),
            ("none_sqrt", "ACDE", payload, 30.0, None),
            ("null_logits", "ACDE", None, 50.0, 10.0),
        ],
        large_binary=large_binary,
    )
    ds = ShardedProteinDataset(
        shard,
        shuffle_shards=False,
        shuffle_rows=False,
        load_teacher_logits=True,
        min_teacher_neff=20.0,
        min_teacher_neff_sqrt_l=2.0,
    )
    got = {str(r["sequence_id"]): r["teacher_logits"] for r in ds}
    assert got["pass_both"] == payload
    assert got["pass_boundary"] == payload
    for key in (
        "fail_neff",
        "fail_sqrt",
        "nan_neff",
        "nan_sqrt",
        "none_neff",
        "none_sqrt",
        "null_logits",
    ):
        assert got[key] is None


def test_teacher_logits_without_gates_does_not_require_neff_columns(tmp_path: Path) -> None:
    payload = b"\x03" * 160
    shard = _write_distill_shard(
        tmp_path / "s.parquet",
        [("a", "ACDE", payload, None, None)],
        include_neff=False,
    )
    ds = ShardedProteinDataset(
        shard, shuffle_shards=False, shuffle_rows=False, load_teacher_logits=True
    )
    row = next(iter(ds))
    assert row["teacher_logits"] == payload


def test_teacher_logits_missing_neff_column_raises(tmp_path: Path) -> None:
    shard = _write_distill_shard(
        tmp_path / "s.parquet", [("a", "ACDE", b"\x00" * 160, None, None)], include_neff=False
    )
    with pytest.raises(ValueError, match="teacher_neff"):
        ShardedProteinDataset(shard, load_teacher_logits=True, min_teacher_neff=20.0)
    with pytest.raises(ValueError, match="teacher_neff_sqrt_l"):
        ShardedProteinDataset(shard, load_teacher_logits=True, min_teacher_neff_sqrt_l=2.0)


def test_mixed_interleaved_distilled_plain_and_sample_members(tmp_path: Path) -> None:
    distilled = _write_distill_shard(
        tmp_path / "d.parquet", [("d0", "ACDE", b"\x02" * 160, 40.0, 6.0)]
    )
    plain = tmp_path / "plain.parquet"
    pq.write_table(
        pa.table({"sequence_id": ["p0", "p1"], "sequence": ["ACDE", "ACDF"]}),
        plain,
    )
    members_shard = tmp_path / "members.parquet"
    pq.write_table(
        pa.table(
            {
                "sequence_id": ["m0"],
                "sequence": ["ACDE"],
                "member_ids": [["m0_a", "m0_b"]],
                "member_sequences": [["ACDE", "ACDF"]],
                "subcluster_offsets": [[0, 1, 2]],
            }
        ),
        members_shard,
    )
    mix = InterleavedDataset(
        [
            ShardedProteinDataset(distilled, load_teacher_logits=True, min_teacher_neff_sqrt_l=2.0),
            ShardedProteinDataset(plain, load_teacher_logits=True, min_teacher_neff_sqrt_l=2.0),
            ShardedProteinDataset(
                members_shard,
                sample_members=True,
                load_teacher_logits=True,
                min_teacher_neff_sqrt_l=2.0,
            ),
            ShardedProteinDataset(
                members_shard,
                sample_members="subcluster",
                load_teacher_logits=True,
                min_teacher_neff_sqrt_l=2.0,
            ),
        ],
        [0.25, 0.25, 0.25, 0.25],
        num_samples=24,
        seed=0,
    )
    rows = list(mix)
    assert all("teacher_logits" in r for r in rows)
    assert any(r["teacher_logits"] is not None for r in rows)
    assert any(r["teacher_logits"] is None for r in rows)


@pytest.mark.parametrize("mode", [True, "uniform", "subcluster"])
def test_sample_members_with_teacher_logits_column_raises(
    tmp_path: Path, mode: bool | str
) -> None:
    shard = tmp_path / "bad.parquet"
    pq.write_table(
        pa.table(
            {
                "sequence_id": ["m0"],
                "sequence": ["ACDE"],
                "member_ids": [["m0_a"]],
                "member_sequences": [["ACDE"]],
                "subcluster_offsets": [[0, 1]],
                "teacher_logits": pa.array([b"\x00" * 160], type=pa.binary()),
            }
        ),
        shard,
    )
    with pytest.raises(ValueError, match="sample_members"):
        ShardedProteinDataset(shard, sample_members=mode, load_teacher_logits=True)


def test_dataset_load_teacher_contacts_shared_neff_gate_and_guards(tmp_path: Path) -> None:
    import numpy as np

    from oplm.data.sequence.collate import encode_teacher_contacts

    c0 = encode_teacher_contacts(np.zeros((10, 10), dtype=np.float32), min_sep=6)
    c1 = encode_teacher_contacts(np.ones((10, 10), dtype=np.float32), min_sep=6)
    shard = tmp_path / "distill_1d2d.parquet"
    pq.write_table(
        pa.table(
            {
                "sequence_id": ["s0", "s1"],
                "sequence": ["ACDEFGHIKL", "MNPQRSTVWY"],
                "teacher_logits": pa.array([b"\x01" * 400, b"\x02" * 400], type=pa.large_binary()),
                "teacher_contacts": pa.array([c0, c1], type=pa.large_binary()),
                "teacher_neff": pa.array([50.0, 5.0], type=pa.float32()),
                "teacher_neff_sqrt_l": pa.array([4.0, 1.0], type=pa.float32()),
            }
        ),
        shard,
    )
    ds = ShardedProteinDataset(
        shard,
        shuffle_shards=False,
        shuffle_rows=False,
        load_teacher_logits=True,
        load_teacher_contacts=True,
        min_teacher_neff_sqrt_l=2.0,
    )
    rows = list(ds)
    assert rows[0]["teacher_logits"] is not None
    assert rows[0]["teacher_contacts"] == c0
    assert rows[1]["teacher_logits"] is None
    assert rows[1]["teacher_contacts"] is None

    # Plain shard missing teacher_contacts yields None gracefully
    plain = tmp_path / "plain.parquet"
    pq.write_table(pa.table({"sequence_id": ["p0"], "sequence": ["ACDEFGHIKL"]}), plain)
    plain_rows = list(
        ShardedProteinDataset(
            plain,
            shuffle_shards=False,
            shuffle_rows=False,
            load_teacher_contacts=True,
            min_teacher_neff_sqrt_l=2.0,
        )
    )
    assert plain_rows[0]["teacher_contacts"] is None

    # sample_members rejected on shards carrying teacher_contacts
    bad_members = tmp_path / "bad_members.parquet"
    pq.write_table(
        pa.table(
            {
                "sequence_id": ["m0"],
                "sequence": ["ACDEFGHIKL"],
                "member_ids": [["m0_a"]],
                "member_sequences": [["ACDEFGHIKL"]],
                "teacher_contacts": pa.array([c0], type=pa.large_binary()),
            }
        ),
        bad_members,
    )
    with pytest.raises(ValueError, match="sample_members"):
        ShardedProteinDataset(bad_members, sample_members=True, load_teacher_contacts=True)
