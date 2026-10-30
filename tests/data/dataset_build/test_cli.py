import json

import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

import oplm.data.build.uniref70 as u
from oplm.data.build.cli import app
from oplm.data.build.manifest import write_manifest

runner = CliRunner()


def _fasta(path, n=40):
    path.write_text("".join(f">R{i}\n>R{i}\nACDEFGHIKL{'A' * i}\n" for i in range(n)))


@pytest.fixture
def calls(monkeypatch):
    calls = {"cluster": 0, "shuffle": 0, "screen": 0}

    def fake_cluster(fasta, workdir, *, mmseqs, threads, extra_args=()):
        calls["cluster"] += 1
        p = workdir / "uniref70_all_seqs.fasta"
        _fasta(p)
        (workdir / "tmp").mkdir(exist_ok=True)
        return p

    real_shuffle = u.shuffle_clusters

    def counting_shuffle(*a, **k):
        calls["shuffle"] += 1
        return real_shuffle(*a, **k)

    def fake_screen(cand, ref, wd, **k):
        calls["screen"] += 1
        return set()

    monkeypatch.setattr(u, "cluster_uniref90", fake_cluster)
    monkeypatch.setattr(u, "shuffle_clusters", counting_shuffle)
    monkeypatch.setattr(u, "screen_against", fake_screen)
    monkeypatch.setattr(u, "_SCREEN_MIN_FREE", 0)
    monkeypatch.setattr(u, "mmseqs_version", lambda m="mmseqs": "fake-1")
    return calls


def _args(tmp_path, deep=True, **extra):
    (tmp_path / "u90.fa").write_text(">a\nAC\n")
    if deep:
        (tmp_path / "deep").mkdir(exist_ok=True)
    a = {
        "--uniref90": tmp_path / "u90.fa",
        "--release": "2026_03",
        "--deepclust70": tmp_path / "deep",
        "--workdir": tmp_path / "wd",
        "--out": tmp_path / "out",
        "--pool": 30,
        "--n-eval": 5,
        "--rows-per-shard": 10,
        "--n-buckets": 4,
        **extra,
    }
    return ["uniref70", *[str(x) for kv in a.items() for x in kv]]


def _rows(out):
    return [
        r["sequence_id"]
        for f in sorted((out / "train").glob("*.parquet"))
        for r in pq.read_table(f).to_pylist()
    ]


def test_uniref70_stages_resume(tmp_path, calls):
    r = runner.invoke(app, _args(tmp_path))
    assert r.exit_code == 0, r.output
    first = _rows(tmp_path / "out")
    assert len(first) == 35
    assert not list((tmp_path / "wd").glob("uniref70_*"))  # mmseqs outputs cleaned
    assert not (tmp_path / "wd" / "tmp").exists()
    assert (tmp_path / "wd" / "shuffled").is_dir()
    (tmp_path / "wd" / "write.done").unlink()
    import shutil

    shutil.rmtree(tmp_path / "out")
    r = runner.invoke(app, _args(tmp_path))
    assert r.exit_code == 0, r.output
    assert calls == {"cluster": 1, "shuffle": 1, "screen": 1}
    assert _rows(tmp_path / "out") == first


def test_uniref70_waits_for_deepclust70(tmp_path, calls):
    r = runner.invoke(app, _args(tmp_path, deep=False, **{"--deepclust70": tmp_path / "nope"}))
    assert r.exit_code == 0, r.output
    assert "waits" in r.output
    assert (tmp_path / "wd" / "shuffle.done").exists()
    assert not (tmp_path / "wd" / "screen.done").exists()


def test_uniref70_accepts_handwritten_cluster_done(tmp_path, calls):
    wd = tmp_path / "wd"
    wd.mkdir()
    _fasta(wd / "uniref70_all_seqs.fasta")
    (wd / "cluster.done").write_text(
        json.dumps(
            {
                "all_seqs_fasta": str(wd / "uniref70_all_seqs.fasta"),
                "mmseqs_version": "hand",
                "threads": 32,
                "extra_args": ["--split-memory-limit", "100G"],
                "seconds": 1,
            }
        )
    )
    r = runner.invoke(app, _args(tmp_path))
    assert r.exit_code == 0, r.output
    assert calls["cluster"] == 0
    assert json.loads((tmp_path / "out" / "summary.json").read_text())["mmseqs_version"] == "hand"


def test_uniref70_refuses_changed_params(tmp_path, calls):
    assert runner.invoke(app, _args(tmp_path)).exit_code == 0
    r = runner.invoke(app, _args(tmp_path, **{"--seed": 1}))
    assert r.exit_code != 0 and "different parameters" in str(r.exception)


def test_uniref70_summary_fields(tmp_path, calls):
    r = runner.invoke(app, _args(tmp_path, **{"--seed": 3}))
    assert r.exit_code == 0, r.output
    s = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert s["uniref90_release"] == "2026_03"
    assert len(s["input_sha256"]) == 64
    assert s["parameters"]["seed"] == 3 and s["mmseqs_version"] == "fake-1"
    assert s["stages"]["screen"]["survivors"] == 30
    assert s["stages"]["write"]["eval"]["rows"] == 5 and s["stages"]["write"]["train"]["rows"] == 35
    assert "oplm_version" in s


def test_manifest_counts_and_limit(tmp_path):
    ds = tmp_path / "ds"
    (ds / "train").mkdir(parents=True)
    import hashlib

    import pyarrow as pa

    pq.write_table(pa.table({"x": [1, 2, 3]}), ds / "train" / "a.parquet")
    (ds / "README.md").write_text("hi")
    (ds / "manifest.parquet").write_text("stale")
    m = pq.read_table(write_manifest(ds)).to_pylist()
    by = {r["path"]: r for r in m}
    assert set(by) == {"train/a.parquet", "README.md"}
    assert by["train/a.parquet"]["rows"] == 3 and by["README.md"]["rows"] is None
    assert by["README.md"]["sha256"] == hashlib.sha256(b"hi").hexdigest()
    assert by["README.md"]["bytes"] == 2
    for i in range(10_001):
        (ds / "train" / f"f{i}.txt").touch()
    with pytest.raises(ValueError, match="train"):
        write_manifest(ds)
    small = tmp_path / "small"
    small.mkdir()
    (small / "a").write_text("x")
    assert runner.invoke(app, ["manifest", str(small)]).exit_code == 0
