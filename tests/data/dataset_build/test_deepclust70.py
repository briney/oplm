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
    assert pq.read_schema(out / "eval_gte2.parquet").field("sequence").type == pa.large_string()
    assert s["stats"]["singletons"]["survivors"] == 9 and s["stats"]["gte2"]["survivors"] == 10
    assert json.loads((out / "heldout.json").read_text()) == s


def test_screen_heldout_too_few_survivors(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="fewer than"):
        _run(tmp_path, monkeypatch, set(), {f"eval-gte2-{i}" for i in range(6)}, n_eval=3, n_test=4)
    assert not (tmp_path / "out").exists()
