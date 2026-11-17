"""Tests for phase-2 member sampling in :class:`ShardedProteinDataset`.

With ``sample_members`` each served cluster row yields one member, chosen by a pure hash of
``(pass seed, shard index, row index)``, so cursors, reshards and topologies are untouched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import oplm.data.sequence.dataset as dataset_mod
from oplm.data.sequence.dataset import (
    MEMBER_IDS_COLUMN,
    MEMBER_SEQUENCES_COLUMN,
    SUBCLUSTER_OFFSETS_COLUMN,
    Reshard,
    ShardedProteinDataset,
    _member_index,
)

if TYPE_CHECKING:
    from pathlib import Path

# Two shards of clusters with 1, 3 and 5 members; the representative is member 0.
_SIZES = [[1, 3, 5], [5, 3, 1]]


def _write(tmp_path: Path) -> Path:
    d = tmp_path / "clusters"
    d.mkdir()
    for s, sizes in enumerate(_SIZES):
        reps, seqs, mids, mseqs = [], [], [], []
        for c, n in enumerate(sizes):
            ids = [f"s{s}c{c}m{j}" for j in range(n)]
            ms = ["ACDEFGHIKL"[: 3 + j] + "A" * c for j in range(n)]
            reps.append(ids[0])
            seqs.append(ms[0])
            mids.append(ids)
            mseqs.append(ms)
        table = pa.table(
            {
                "sequence_id": reps,
                "sequence": seqs,
                MEMBER_IDS_COLUMN: mids,
                MEMBER_SEQUENCES_COLUMN: mseqs,
            }
        )
        pq.write_table(table, d / f"shard-{s}.parquet")
    return d


def _rows(
    ds: ShardedProteinDataset, rank: int = 0, world: int = 1, pass_index: int = 0
) -> list[dict[str, object]]:
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(dataset_mod, "_resolve_distributed_context", lambda: (rank, world, 0, 1))
        return list(ds._iter_pass(pass_index))


def _by_cluster(rows: list[dict[str, object]]) -> dict[str, str]:
    """Physical row (cluster) -> served member id."""
    return {str(r["sequence_id"]).rsplit("m", 1)[0]: str(r["sequence_id"]) for r in rows}


def test_flag_off_is_unchanged(tmp_path: Path) -> None:
    d = _write(tmp_path)
    rows = _rows(ShardedProteinDataset(d, seed=3))
    assert len(rows) == 6
    assert all(set(r) == {"sequence_id", "sequence"} for r in rows)
    assert all(str(r["sequence_id"]).endswith("m0") for r in rows)


def test_member_rows_come_from_their_cluster(tmp_path: Path) -> None:
    d = _write(tmp_path)
    for row in _rows(ShardedProteinDataset(d, seed=3, sample_members=True)):
        cluster = str(row["sequence_id"]).rsplit("m", 1)[0]
        n = _SIZES[int(cluster[1])][int(cluster[3])]
        j = int(str(row["sequence_id"]).rsplit("m", 1)[1])
        assert j < n
        assert row["sequence"] == "ACDEFGHIKL"[: 3 + j] + "A" * int(cluster[3])


def test_singleton_yields_representative(tmp_path: Path) -> None:
    d = _write(tmp_path)
    rows = _rows(ShardedProteinDataset(d, sample_members=True))
    singles = [r for r in rows if str(r["sequence_id"]).startswith(("s0c0", "s1c2"))]
    assert len(singles) == 2
    assert all(str(r["sequence_id"]).endswith("m0") for r in singles)


def _served(ds: ShardedProteinDataset, world: int) -> dict[str, str]:
    out: dict[str, str] = {}
    for rank in range(world):
        for r in _rows(ds, rank, world):
            sid = str(r["sequence_id"])
            out[sid.rsplit("m", 1)[0]] = sid
    return out


def test_same_member_across_topologies(tmp_path: Path) -> None:
    d = _write(tmp_path)
    ds = ShardedProteinDataset(d, seed=5, sample_members=True)
    one = _served(ds, 1)
    assert len(one) == 6
    assert _served(ds, 2) == one
    assert _served(ds, 3) == one


def test_members_change_across_epochs(tmp_path: Path) -> None:
    d = _write(tmp_path)
    ds = ShardedProteinDataset(d, sample_members=True)
    seen = set()
    for epoch in range(10):
        ds.set_epoch(epoch)
        seen |= {
            str(r["sequence_id"]) for r in _rows(ds) if str(r["sequence_id"]).startswith("s0c2")
        }
    assert len(seen) > 1


def test_resume_and_reshard_serve_the_same_member(tmp_path: Path) -> None:
    d = _write(tmp_path)
    ds = ShardedProteinDataset(d, seed=7, sample_members=True)
    fresh = _rows(ds)
    ds.set_resume_rows([2])
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(dataset_mod, "_resolve_distributed_context", lambda: (0, 1, 0, 1))
        resumed = list(ds)
    assert resumed == fresh[2:]
    ds.clear_resume_skip()
    ds.set_reshards([Reshard(stride=2, epochs=[0, 0], rows=[1, 2])])
    layered = _by_cluster(_rows(ds))
    assert 0 < len(layered) < len(fresh)
    full = _by_cluster(fresh)
    assert all(full[c] == m for c, m in layered.items())


def test_refill_pass_redraws_members(tmp_path: Path) -> None:
    d = _write(tmp_path)
    ds = ShardedProteinDataset(d, sample_members=True)
    seen = {_by_cluster(_rows(ds, pass_index=p))["s0c2"] for p in range(10)}
    assert len(seen) > 1


def test_member_choice_is_uniform() -> None:
    counts = [0, 0, 0]
    for shard in range(300):
        for row in range(100):
            counts[_member_index(12345, shard, row, 3)] += 1
    assert all(abs(c - 10_000) < 300 for c in counts)


def test_missing_member_columns_raise(tmp_path: Path) -> None:
    pq.write_table(pa.table({"sequence_id": ["a"], "sequence": ["AC"]}), tmp_path / "x.parquet")
    with pytest.raises(ValueError, match=MEMBER_IDS_COLUMN):
        ShardedProteinDataset(tmp_path, sample_members=True)


def test_masking_weights_with_members_raise(tmp_path: Path) -> None:
    d = _write(tmp_path)
    with pytest.raises(ValueError, match="masking"):
        ShardedProteinDataset(d, sample_members=True, load_masking_weights=True)


@pytest.mark.parametrize(
    ("ids", "seqs"),
    [(["a", "b"], ["AC", "AD", "AE"]), ([], []), (["a"], None)],
    ids=["mismatch", "empty", "null-seqs"],
)
def test_mismatched_member_lists_raise(
    tmp_path: Path, ids: list[str], seqs: list[str] | None
) -> None:
    table = pa.table(
        {
            "sequence_id": ["r0", "r1"],
            "sequence": ["AC", "AC"],
            MEMBER_IDS_COLUMN: pa.array([["r0"], ids], type=pa.list_(pa.string())),
            MEMBER_SEQUENCES_COLUMN: pa.array([["AC"], seqs], type=pa.list_(pa.string())),
        }
    )
    path = tmp_path / "bad.parquet"
    pq.write_table(table, path)
    ds = ShardedProteinDataset(path, shuffle_rows=False, sample_members=True)
    with pytest.raises(ValueError, match=r"bad\.parquet.*row 1"):
        _rows(ds)


# One 50% cluster: subcluster 0 has 1 member, subcluster 1 has 9 (offsets [0, 1, 10]).
_SUB_IDS = [f"k{0 if j == 0 else 1}m{j}" for j in range(10)]


def _write_subclusters(tmp_path: Path, offsets: list[list[int]] | None = None) -> Path:
    d = tmp_path / "sub"
    d.mkdir()
    n_rows = 400
    table = pa.table(
        {
            "sequence_id": [f"r{i}" for i in range(n_rows)],
            "sequence": ["AC"] * n_rows,
            MEMBER_IDS_COLUMN: [_SUB_IDS] * n_rows,
            MEMBER_SEQUENCES_COLUMN: [["AC" * (j + 1) for j in range(10)]] * n_rows,
            SUBCLUSTER_OFFSETS_COLUMN: offsets or [[0, 1, 10]] * n_rows,
        }
    )
    pq.write_table(table, d / "shard-0.parquet")
    return d


def test_subcluster_mode_is_uniform_over_subclusters(tmp_path: Path) -> None:
    rows = _rows(ShardedProteinDataset(_write_subclusters(tmp_path), sample_members="subcluster"))
    assert len(rows) == 400
    singleton = sum(r["sequence_id"] == "k0m0" for r in rows)
    assert 160 < singleton < 240  # ~1/2, vs ~1/10 under uniform member sampling
    big = {r["sequence_id"] for r in rows} - {"k0m0"}
    assert len(big) == 9  # every member of the big subcluster gets drawn
    for r in rows:
        j = _SUB_IDS.index(str(r["sequence_id"]))
        assert r["sequence"] == "AC" * (j + 1)


def test_true_and_uniform_are_the_same_mode(tmp_path: Path) -> None:
    d = _write(tmp_path)
    assert _rows(ShardedProteinDataset(d, seed=2, sample_members=True)) == _rows(
        ShardedProteinDataset(d, seed=2, sample_members="uniform")
    )


def test_subcluster_same_member_across_topologies(tmp_path: Path) -> None:
    ds = ShardedProteinDataset(_write_subclusters(tmp_path), seed=9, sample_members="subcluster")
    one = sorted((str(r["sequence_id"]), str(r["sequence"])) for r in _rows(ds))
    for world in (2, 3):
        multi = sorted(
            (str(r["sequence_id"]), str(r["sequence"]))
            for rank in range(world)
            for r in _rows(ds, rank, world)
        )
        assert multi == one


def test_subcluster_draws_are_jointly_uniform() -> None:
    counts = [[0] * 3 for _ in range(2)]
    for shard in range(200):
        for row in range(120):
            k = _member_index(777, shard, row, 2)
            counts[k][_member_index(777, shard, row, 3, dataset_mod._SUBCLUSTER_FOLD)] += 1
    assert all(abs(c - 4_000) < 250 for block in counts for c in block)


def test_subcluster_needs_offsets_column(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=SUBCLUSTER_OFFSETS_COLUMN):
        ShardedProteinDataset(_write(tmp_path), sample_members="subcluster")


def test_unknown_mode_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sample_members"):
        ShardedProteinDataset(_write(tmp_path), sample_members="subclusters")


@pytest.mark.parametrize(
    "bad",
    [[0, 1, 9], [1, 10], [0, 1, 1, 10], [0], []],
    ids=["short", "nonzero", "empty", "k0", "none"],
)
def test_bad_offsets_raise(tmp_path: Path, bad: list[int]) -> None:
    d = _write_subclusters(tmp_path, [[0, 1, 10]] + [bad] * 399)
    ds = ShardedProteinDataset(d, shuffle_rows=False, sample_members="subcluster")
    with pytest.raises(ValueError, match=r"row 1: subcluster_offsets"):
        _rows(ds)
