import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from oplm.cli import app
from oplm.data.build.reshard import reshard, stream_digest


def _src(tmp_path, weights=False):
    src = tmp_path / "src"
    src.mkdir()
    for i in range(7):
        cols = {
            "sequence_id": [f"s{i}_{j}" for j in range(3)],
            "sequence": ["ACD" * (j + 1) for j in range(3)],
        }
        if weights:
            cols["masking_weights"] = [[float(i), float(j)] for j in range(3)]
        pq.write_table(pa.table(cols), src / f"part-{i:03d}.parquet")
    return src


def _rows(d):
    return [r for p in sorted(d.glob("*.parquet")) for r in pq.read_table(p).to_pylist()]


def test_reshard_preserves_rows_and_order(tmp_path):
    src = _src(tmp_path)
    s = reshard(src, tmp_path / "out", rows_per_shard=5)
    sizes = [
        pq.ParquetFile(p).metadata.num_rows for p in sorted((tmp_path / "out").glob("*.parquet"))
    ]
    assert sizes == [5, 5, 5, 5, 1]
    assert _rows(tmp_path / "out") == _rows(src)
    assert s["rows"] == 21 and s["shards_out"] == 5 and s["sha256_in"] == s["sha256_out"]
    assert json.loads((tmp_path / "out/summary.json").read_text())["sha256_out"] == s["sha256_out"]
    assert stream_digest(sorted(src.glob("*.parquet"))) == (21, s["sha256_in"])
    assert not list((tmp_path / "out").glob("*.tmp"))


def test_reshard_keeps_extra_columns(tmp_path):
    src = _src(tmp_path, weights=True)
    reshard(src, tmp_path / "out", rows_per_shard=5)
    assert _rows(tmp_path / "out") == _rows(src)


def test_reshard_refuses_nonempty_out(tmp_path):
    src = _src(tmp_path)
    out = tmp_path / "out"
    reshard(src, out, rows_per_shard=5)
    with pytest.raises(FileExistsError):
        reshard(src, out, rows_per_shard=5)


def test_cli_reshard(tmp_path):
    src = _src(tmp_path)
    r = CliRunner().invoke(
        app, ["data", "reshard", str(src), str(tmp_path / "o"), "--rows-per-shard", "5"]
    )
    assert r.exit_code == 0, r.output
    assert (tmp_path / "o/shard-00004.parquet").exists()


def test_reshard_rejects_nonpositive_rows_per_shard(tmp_path):
    with pytest.raises(ValueError, match="rows_per_shard"):
        reshard(_src(tmp_path), tmp_path / "out", rows_per_shard=0)


def test_reshard_checks_free_space(tmp_path, monkeypatch):
    import shutil
    from collections import namedtuple

    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda p: usage(0, 0, 10))
    with pytest.raises(RuntimeError, match="free"):
        reshard(_src(tmp_path), tmp_path / "out", rows_per_shard=5)
    assert not list((tmp_path / "out").iterdir())


def test_reshard_verify_failure_leaves_marker(tmp_path, monkeypatch):
    import oplm.data.build.reshard as r

    monkeypatch.setattr(r, "stream_digest", lambda paths: (21, "bad"))
    with pytest.raises(RuntimeError, match="VERIFY_FAILED"):
        reshard(_src(tmp_path), tmp_path / "out", rows_per_shard=5)
    assert "bad" in (tmp_path / "out" / "VERIFY_FAILED").read_text()
    assert not (tmp_path / "out" / "summary.json").exists()


def test_cli_rejects_zero_rows_per_shard(tmp_path):
    r = CliRunner().invoke(
        app, ["data", "reshard", str(_src(tmp_path)), str(tmp_path / "o"), "--rows-per-shard", "0"]
    )
    assert r.exit_code != 0
