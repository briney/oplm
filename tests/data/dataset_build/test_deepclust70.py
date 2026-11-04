import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import oplm.data.build.deepclust70 as d


def _src(tmp_path, n=6):
    src = tmp_path / "src"
    src.mkdir()
    for split in ("eval", "test"):
        for k in ("singletons", "gte2"):
            ids = [f"{split}-{k}-{i}" for i in range(n)]
            t = pa.table(
                {"sequence_id": ids, "sequence": ["ACDE"] * n},
                schema=pa.schema(
                    [("sequence_id", pa.large_string()), ("sequence", pa.large_string())]
                ),
            )
            pq.write_table(t, src / f"{split}_{k}.parquet")
    return src


def _run(tmp_path, monkeypatch, deep, u70, **kw):
    def fake(cands, reference, workdir, *, members=False, **_):
        assert all(c["member_ids"] == [c["sequence_id"]] for c in cands) and len(cands) == 24
        return u70 if members else deep

    monkeypatch.setattr(d, "screen_against", fake)
    monkeypatch.setattr(d, "mmseqs_version", lambda m: "x")
    args = dict(
        deepclust70=tmp_path, uniref70=tmp_path, workdir=tmp_path / "w", mmseqs="m", threads=1
    )
    return d.screen_heldout(_src(tmp_path), tmp_path / "out", **args, **kw)


def test_screen_heldout_keeps_survivors_in_pool_order(tmp_path, monkeypatch):
    deep = {"eval-singletons-0", "eval-gte2-1", "test-gte2-0"}
    u70 = {"eval-singletons-2", "test-singletons-5"}
    s = _run(tmp_path, monkeypatch, deep, u70, n_eval=3, n_test=4)
    out = tmp_path / "out"

    def ids(name):
        return pq.read_table(out / f"{name}.parquet")["sequence_id"].to_pylist()

    assert ids("eval_singletons") == ["eval-singletons-1", "eval-singletons-3", "eval-singletons-4"]
    assert ids("test_singletons") == [
        "eval-singletons-5",
        *(f"test-singletons-{i}" for i in range(3)),
    ]
    assert ids("eval_gte2") == ["eval-gte2-0", "eval-gte2-2", "eval-gte2-3"]
    assert ids("test_gte2") == ["eval-gte2-4", "eval-gte2-5", "test-gte2-1", "test-gte2-2"]
    assert not list(out.glob("test_extra_*"))
    assert pq.read_schema(out / "eval_gte2.parquet").field("sequence").type == pa.large_string()
    assert s["stats"]["singletons"]["survivors"] == 9 and s["stats"]["gte2"]["survivors"] == 10
    assert json.loads((out / "heldout.json").read_text()) == s


def test_screen_heldout_too_few_survivors(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="fewer than"):
        _run(tmp_path, monkeypatch, set(), {f"eval-gte2-{i}" for i in range(6)}, n_eval=3, n_test=4)
    assert not (tmp_path / "out").exists()


def test_return_heldout_merges_unused_into_last_shard(tmp_path, monkeypatch):
    src = _src(tmp_path)
    out = tmp_path / "dc"
    (out / "train").mkdir(parents=True)
    u70 = tmp_path / "u70"
    u70.mkdir()
    schema = pq.read_schema(src / "eval_gte2.parquet")
    for name, n in (("shard-00000", 4), ("shard-00001", 2)):
        ids = [f"{name}-{i}" for i in range(n)]
        pq.write_table(
            pa.table({"sequence_id": ids, "sequence": ["AC"] * n}, schema=schema),
            out / "train" / f"{name}.parquet",
        )
    for s in ("eval", "test"):
        for k in ("singletons", "gte2"):
            t = pq.read_table(src / f"{s}_{k}.parquet").slice(0, 2)  # 2 held out per file
            pq.write_table(t, out / f"{s}_{k}.parquet")
            pq.write_table(pa.table({"x": [1]}), u70 / f"{s}_{k}.parquet")
    pq.write_table(pa.table({"x": [1]}), out / "stray.parquet")  # must not be a screen ref
    seen = {}

    def fake(cands, reference, workdir, *, members=False, **_):
        seen[members] = (sorted(p.name for p in reference.iterdir()), len(cands))
        return {"eval-gte2-5"} if members else {"test-singletons-2"}

    monkeypatch.setattr(d, "screen_against", fake)
    monkeypatch.setattr(d, "mmseqs_version", lambda m: "x")
    kw = dict(uniref70=u70, workdir=tmp_path / "w", mmseqs="m", threads=1)
    s = d.return_heldout(src, out, **kw)

    refs = sorted(f"{a}_{b}.parquet" for a in ("eval", "test") for b in ("gte2", "singletons"))
    assert seen == {False: (refs, 16), True: (refs, 16)}  # 24 pool - 8 held out
    assert pq.read_table(out / "train" / "shard-00000.parquet").num_rows == 4
    last = pq.read_table(out / "train" / "shard-00001.parquet")["sequence_id"].to_pylist()
    want = {"shard-00001-0", "shard-00001-1"} | {
        f"{sp}-{k}-{i}"
        for sp in ("eval", "test")
        for k in ("singletons", "gte2")
        for i in range(2, 6)
    } - {"eval-gte2-5", "test-singletons-2"}
    assert sorted(last) == sorted(want) and len(last) == 16
    assert s["stats"]["shard"] == {
        "name": "shard-00001.parquet",
        "rows_before": 2,
        "rows_after": 16,
    }
    assert s["stats"]["singletons"] == {"unused": 8, "returned": 7}
    assert json.loads((out / "returned.json").read_text()) == s
    with pytest.raises(FileExistsError):
        d.return_heldout(src, out, **kw)
