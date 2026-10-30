"""Seeded randomized check of the token-mode reshard cursor over a 2-source mix (#63).

Each phase picks a stride (world=stride, workers=1, so every stream rolls epochs on its own, the
general token-mode case). Streams consume random row counts (rolling over epochs), a cursor is
built as ``Trainer._save_checkpoint`` does, and the next phase resumes it as
``Trainer._resume_data_cursor`` does: same layout arms rows/sources; otherwise a ``Reshard`` is
appended and every stream restarts at ``min(rank_epochs)``. Checks:

- no row twice within one (epoch, source, pass);
- per finished epoch, served + EXHAUSTED draws == num_samples (drop False) / within slack (True);
- at each reshard, served-so-far + the live streams' full layered pass == the source (drop False);
- a same-layout resume continues the uninterrupted layered stream exactly.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any

import pytest

import oplm.data.sequence.dataset as dm
from oplm.data.sequence.dataset import (
    _EXHAUSTED,
    POSITION_KEY,
    InterleavedDataset,
    Reshard,
    ShardedProteinDataset,
)
from tests.data.conftest import _load_real_records, write_sequence_shards

if TYPE_CHECKING:
    from pathlib import Path

_N = 90
_SEEDS = 40

CTX = [(0, 1, 0, 1)]
EXH: Counter[int] = Counter()
COUNT = [True]


@pytest.fixture(scope="module")
def mix_root(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[Counter[str]]]:
    """Source ``a`` (37 rows, 3 shards) and ``b`` (53 rows, 4 shards), plus each one's ids."""
    root = tmp_path_factory.mktemp("reshard_stress")
    recs = _load_real_records(_N)
    (root / "a").mkdir()
    (root / "b").mkdir()
    write_sequence_shards(root / "a", recs[:37], n_shards=3, id_prefix="a:")
    write_sequence_shards(root / "b", recs[37:], n_shards=4, id_prefix="b:")
    ids = [Counter(f"a:{r}" for r, _ in recs[:37]), Counter(f"b:{r}" for r, _ in recs[37:])]
    return root, ids


@pytest.fixture(autouse=True)
def _patched(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drive the distributed context from ``CTX`` and count EXHAUSTED draws per epoch."""
    monkeypatch.setattr(dm, "_resolve_distributed_context", lambda: CTX[0])
    orig = InterleavedDataset._next_or_refill

    def _counting(self, iters, passes, source_idx):  # noqa: ANN001, ANN202
        out = orig(self, iters, passes, source_idx)
        if out is _EXHAUSTED and COUNT[0]:
            EXH[self._epoch] += 1
        return out

    monkeypatch.setattr(InterleavedDataset, "_next_or_refill", _counting)


def _factory(root: Path, drop: bool, reshuffle: bool, fracs: list[float]) -> InterleavedDataset:
    srcs = [ShardedProteinDataset(root / n, seed=5, drop_remainder=drop) for n in ("a", "b")]
    return InterleavedDataset(
        srcs, fracs, seed=5, reshuffle_on_refill=reshuffle, drop_remainder=drop
    )


def _full_epoch(ds: InterleavedDataset, j: int, s: int) -> list[str]:
    COUNT[0] = False
    CTX[0] = (j, s, 0, 1)
    out = [str(r["sequence_id"]) for r in ds]
    COUNT[0] = True
    return out


def _sim(root: Path, all_ids: list[Counter[str]], seed: int, drop: bool, reshuffle: bool) -> None:
    rng = random.Random(seed)
    fracs = rng.choice([[0.6, 0.4], [0.9, 0.1], [0.2, 0.8]])
    EXH.clear()
    served: defaultdict[tuple[int, int, int], Counter[str]] = defaultdict(Counter)
    per_epoch: Counter[int] = Counter()
    reshards: list[Reshard] = []
    cursor: dict[str, Any] | None = None
    strides_used: list[int] = []
    n_phases = rng.randint(2, 5)
    for phase in range(n_phases + 1):
        final = phase == n_phases
        if cursor is not None and rng.random() < 0.3:
            s = cursor["S"]
        else:
            s = rng.choice([1, 2, 3, 4, 5, 7])
        strides_used.append(s)
        same = cursor is not None and s == cursor["S"]
        if cursor is None:
            epochs, resume, counts_in_epoch = [0] * s, [None] * s, [0] * s
        elif same:
            reshards = [Reshard(**r) for r in cursor["reshards"]]
            epochs = list(cursor["epochs"])
            resume = list(zip(cursor["rows"], cursor["sources"], strict=True))
            counts_in_epoch = list(cursor["counts"])
        else:
            reshards = [Reshard(**r) for r in cursor["reshards"]]
            reshards.append(
                Reshard(
                    stride=cursor["S"],
                    epochs=list(cursor["epochs"]),
                    rows=list(cursor["rows"]),
                    sources=list(cursor["sources"]),
                )
            )
            m = min(cursor["epochs"])
            epochs, resume, counts_in_epoch = [m] * s, [None] * s, [0] * s
            # Coverage oracle on the latest epoch any old stream is in (no stream ahead of it).
            x = max(cursor["epochs"])
            ahead = any(max(r.epochs) > x for r in reshards if min(r.epochs) <= x)
            if not ahead:
                ds = _factory(root, drop, reshuffle, fracs)
                ds.set_reshards(reshards)
                ds.set_epoch(x)
                for src in range(2):
                    top = max((p for (e, sx, p) in served if e == x and sx == src), default=0)
                    for p in range(top + 3):
                        union: Counter[str] = Counter()
                        for j in range(s):
                            CTX[0] = (j, s, 0, 1)
                            union.update(str(r["sequence_id"]) for r in ds._open(src, p))
                        tot = served[(x, src, p)] + union
                        assert max(tot.values(), default=0) <= 1, ("dup-oracle", x, src, p)
                        if not drop:
                            assert tot == all_ids[src], ("cover", x, src, p)
                        else:
                            missing = sum((all_ids[src] - tot).values())
                            assert missing < sum(strides_used), ("drop-miss", x, src, p)
        target = min(epochs) + 2
        new_rows, new_sources = [], []
        for j in range(s):
            e = epochs[j]
            ds = _factory(root, drop, reshuffle, fracs)
            ds.set_reshards(reshards)
            ds.set_epoch(e)
            tag_rows, tag_src = 0, None
            if same and resume[j] is not None:
                rows, src_tag = resume[j]
                tag_rows, tag_src = rows, src_tag
                ds.set_resume_rows([rows])
                ds.set_resume_sources([src_tag])
            ref = None
            if same:
                ref_ds = _factory(root, drop, reshuffle, fracs)
                ref_ds.set_reshards(reshards)
                ref_ds.set_epoch(e)
                ref = _full_epoch(ref_ds, j, s)[counts_in_epoch[j] :]
            budget = 10**9 if final else rng.randint(0, 50)
            CTX[0] = (j, s, 0, 1)
            it = iter(ds)
            got: list[str] = []
            empties = 0
            while budget > 0:
                if final and e >= target:
                    break
                try:
                    CTX[0] = (j, s, 0, 1)
                    row = next(it)
                except StopIteration:
                    if ref is not None:
                        assert got == ref, ("same-layout", j, e)
                        ref = None
                    e += 1
                    ds.set_epoch(e)
                    ds.clear_resume_skip()
                    tag_rows, tag_src, counts_in_epoch[j] = 0, None, 0
                    CTX[0] = (j, s, 0, 1)
                    it = iter(ds)
                    empties += 1
                    assert empties < 10, "stuck"
                    continue
                empties = 0
                sid = str(row["sequence_id"])
                tag = row[POSITION_KEY]
                src = 0 if sid.startswith("a:") else 1
                served[(e, src, tag[1][src][0])][sid] += 1
                per_epoch[e] += 1
                tag_rows, tag_src = tag[0], [list(p) for p in tag[1]]
                counts_in_epoch[j] += 1
                got.append(sid)
                budget -= 1
            if ref is not None:
                assert got == ref[: len(got)], ("same-layout-prefix", j, e)
            epochs[j] = e
            new_rows.append(tag_rows)
            new_sources.append(tag_src)
        kept = [r for r in reshards if max(r.epochs) >= min(epochs)]
        cursor = {
            "S": s,
            "epochs": epochs,
            "rows": new_rows,
            "sources": new_sources,
            "counts": counts_in_epoch,
            "reshards": [
                {"stride": r.stride, "epochs": r.epochs, "rows": r.rows, "sources": r.sources}
                for r in kept
            ],
        }
    assert cursor is not None
    for key, c in served.items():
        assert max(c.values(), default=0) <= 1, ("dup", key)
    for e in range(min(cursor["epochs"])):
        total = per_epoch[e] + EXH[e]
        if drop:
            assert _N - sum(strides_used) <= total <= _N, ("draws-drop", e, total)
        else:
            assert total == _N, ("draws", e, total, per_epoch[e], EXH[e])


@pytest.mark.parametrize("reshuffle", [True, False], ids=["reshuffle", "no-reshuffle"])
@pytest.mark.parametrize("drop", [False, True], ids=["keep", "drop"])
@pytest.mark.parametrize("seed", range(_SEEDS))
def test_random_reshard_sequence(
    mix_root: tuple[Path, list[Counter[str]]], seed: int, drop: bool, reshuffle: bool
) -> None:
    """Random reshard/resume sequences never repeat a row and keep each epoch's draw budget."""
    _sim(*mix_root, seed, drop, reshuffle)
