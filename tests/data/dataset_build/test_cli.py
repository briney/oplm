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
    if deep:  # a finished, verified `oplm data reshard` output
        (tmp_path / "deep").mkdir(exist_ok=True)
        (tmp_path / "deep" / "summary.json").write_text(
            json.dumps({"rows": 7, "sha256_out": "d" * 64})
        )
    a = {
        "--uniref90": tmp_path / "u90.fa",
        "--release": "2026_03",
        "--deepclust70": tmp_path / "deep",
        "--workdir": tmp_path / "wd",
        "--out": tmp_path / "out",
        "--pool": 30,
        "--n-eval": 5,
        "--n-test": 3,
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
    assert len(first) == 32
    assert not list((tmp_path / "wd").glob("uniref70_*"))  # mmseqs outputs cleaned
    assert not (tmp_path / "wd" / "tmp").exists()
    assert (tmp_path / "wd" / "shuffled").is_dir()
    (tmp_path / "wd" / "write.done").unlink()
    import shutil

    shutil.rmtree(tmp_path / "out")
    r = runner.invoke(app, _args(tmp_path))
    assert r.exit_code == 0, r.output
    assert calls == {"cluster": 1, "shuffle": 1, "screen": 2}  # deepclust70 + uniref70, once
    assert _rows(tmp_path / "out") == first


def test_uniref70_waits_for_deepclust70(tmp_path, calls):
    r = runner.invoke(app, _args(tmp_path, deep=False, **{"--deepclust70": tmp_path / "nope"}))
    assert r.exit_code == 0, r.output
    assert "waits" in r.output
    assert (tmp_path / "wd" / "shuffle.done").exists()
    assert not (tmp_path / "wd" / "screen.done").exists()
    (tmp_path / "mid").mkdir()  # reshard still running: shards but no summary.json yet
    r = runner.invoke(app, _args(tmp_path, deep=False, **{"--deepclust70": tmp_path / "mid"}))
    assert r.exit_code == 0 and "waits" in r.output
    assert calls["screen"] == 0


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
    assert s["stages"]["write"]["eval"]["rows"] == 5 and s["stages"]["write"]["train"]["rows"] == 32
    assert s["stages"]["write"]["test"]["rows"] == 3 and s["parameters"]["n_test"] == 3
    assert "oplm_version" in s
    assert s["uniref90_sequences_in"] == 40 and s["clusters_out"] == 40
    assert s["screen_mmseqs_version"] == "fake-1"
    assert s["deepclust70"]["rows"] == 7 and s["deepclust70"]["sha256_out"] == "d" * 64


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


def test_uniref70_rerun_after_crash_before_write_done(tmp_path, calls):
    assert runner.invoke(app, _args(tmp_path)).exit_code == 0
    first = _rows(tmp_path / "out")
    assert not (tmp_path / "wd" / "write.started").exists()
    # the state a crash mid-write leaves: write.started, partial train/, no write.done
    (tmp_path / "wd" / "write.done").unlink()
    (tmp_path / "wd" / "write.started").write_text(str((tmp_path / "out").resolve()))
    (tmp_path / "out" / "train" / "stale.parquet.tmp").write_text("x")
    r = runner.invoke(app, _args(tmp_path))
    assert r.exit_code == 0, r.output
    assert _rows(tmp_path / "out") == first
    assert not list((tmp_path / "out" / "train").glob("*.tmp"))


def test_uniref70_write_keeps_foreign_train(tmp_path, calls):
    train = tmp_path / "out" / "train"
    train.mkdir(parents=True)
    (train / "precious.parquet").write_text("x")
    r = runner.invoke(app, _args(tmp_path))
    assert isinstance(r.exception, FileExistsError)
    assert (train / "precious.parquet").read_text() == "x"
    assert not (tmp_path / "wd" / "write.started").exists()
    # a marker left for a different --out does not license cleaning this one
    (tmp_path / "wd" / "write.started").write_text(str(tmp_path / "elsewhere"))
    assert isinstance(runner.invoke(app, _args(tmp_path)).exception, FileExistsError)
    assert (train / "precious.parquet").exists()


def test_uniref70_refuses_out_over_deepclust70(tmp_path, calls):
    a = _args(tmp_path, **{"--deepclust70": tmp_path / "out" / "train"})
    r = runner.invoke(app, a)
    assert isinstance(r.exception, ValueError) and "overwrite" in str(r.exception)
    assert calls["cluster"] == 0


@pytest.fixture
def tsv_lines(calls, monkeypatch):
    lines = {"n": 40}
    fake = u.cluster_uniref90

    def with_tsv(fasta, workdir, **k):
        p = fake(fasta, workdir, **k)
        (workdir / "uniref70_cluster.tsv").write_text("R\tR\n" * lines["n"])
        return p

    monkeypatch.setattr(u, "cluster_uniref90", with_tsv)
    return lines


def test_uniref70_reconciles_cluster_tsv(tmp_path, tsv_lines):
    assert runner.invoke(app, _args(tmp_path)).exit_code == 0
    sh = json.loads((tmp_path / "wd" / "shuffle.done").read_text())
    assert (sh["clusters"], sh["members"], sh["cluster_tsv_lines"]) == (40, 40, 40)


def test_uniref70_tsv_mismatch_raises(tmp_path, tsv_lines):
    tsv_lines["n"] = 41
    r = runner.invoke(app, _args(tmp_path))
    assert isinstance(r.exception, RuntimeError) and "41 lines" in str(r.exception)
    assert not (tmp_path / "wd" / "shuffle.done").exists()


@pytest.mark.parametrize("opt", ["--rows-per-shard", "--n-buckets", "--pool", "--chunk-rows"])
def test_uniref70_rejects_nonpositive_counts(tmp_path, calls, opt):
    assert runner.invoke(app, _args(tmp_path, **{opt: 0})).exit_code != 0
    assert calls["cluster"] == 0


def test_uniref70_screen_unions_deepclust70_and_uniref70(tmp_path, calls, monkeypatch):
    refs = {}

    def fake_screen(cand, ref, wd, *, members=False, **k):
        refs[members] = (ref, wd)
        return {cand[0 if members else 1]["sequence_id"]}  # different hit per screen

    monkeypatch.setattr(u, "screen_against", fake_screen)
    r = runner.invoke(app, _args(tmp_path))
    assert r.exit_code == 0, r.output
    wd = tmp_path / "wd"
    assert refs == {
        False: (tmp_path / "deep", wd / "screen"),
        True: (wd / "shuffled", wd / "screen-uniref70"),
    }
    sc = json.loads((wd / "screen.done").read_text())
    assert sc["params"]["screen_uniref70"] is True
    assert sc["stats"]["hits"] == 2 and sc["stats"]["survivors"] == 28
    assert sc["stats"]["hits_deepclust70"] == 1 and sc["stats"]["hits_uniref70"] == 1
    pool = [r["sequence_id"] for r in u.iter_rows(sorted((wd / "shuffled").glob("*.parquet")))]
    assert not set(sc["eval_ids"]) & set(pool[:2])


def _read(f):
    return [r["sequence_id"] for r in pq.read_table(f).to_pylist()]


def test_uniref70_test_split_follows_eval(tmp_path, calls, monkeypatch):
    wd, out = tmp_path / "wd", tmp_path / "out"
    hit = {}

    def fake_screen(cand, ref, wd_, *, members=False, **k):
        hit.setdefault("ids", {cand[1]["sequence_id"], cand[4]["sequence_id"]})
        return set(hit["ids"]) if not members else set()

    monkeypatch.setattr(u, "screen_against", fake_screen)
    r = runner.invoke(app, _args(tmp_path, **{"--n-eval": 3, "--n-test": 2}))
    assert r.exit_code == 0, r.output
    pool = [r["sequence_id"] for r in u.iter_rows(sorted((wd / "shuffled").glob("*.parquet")))]
    surv = [i for i in pool[:30] if i not in hit["ids"]]
    sc = json.loads((wd / "screen.done").read_text())
    assert sc["params"]["n_test"] == 2
    assert sc["eval_ids"] == surv[:3] and sc["test_ids"] == surv[3:5]

    def held(name):
        return {
            i
            for k in ("singletons", "gte2")
            if (out / f"{name}_{k}.parquet").exists()
            for i in _read(out / f"{name}_{k}.parquet")
        }

    assert held("eval") == set(surv[:3]) and held("test") == set(surv[3:5])
    assert not (out / "eval.parquet").exists()
    assert not set(_rows(out)) & set(surv[:5])
    s = json.loads((out / "summary.json").read_text())
    assert s["stages"]["write"]["test"]["rows"] == 2


def test_uniref70_n_test_zero_writes_no_test(tmp_path, calls):
    r = runner.invoke(app, _args(tmp_path, **{"--n-test": 0}))
    assert r.exit_code == 0, r.output
    assert not list((tmp_path / "out").glob("test*"))
    assert len(_rows(tmp_path / "out")) == 35


def test_uniref70_restart_removes_stale_test(tmp_path, calls):
    assert runner.invoke(app, _args(tmp_path)).exit_code == 0
    (tmp_path / "wd" / "write.done").unlink()
    (tmp_path / "wd" / "write.started").write_text(str((tmp_path / "out").resolve()))
    stale = [tmp_path / "out" / n for n in ("test_gte2.parquet.tmp", "eval_gte2.parquet.tmp")]
    for f in stale:
        f.write_text("x")
    final = sorted((tmp_path / "out").glob("test_*.parquet"))
    assert final
    final[0].unlink()
    assert runner.invoke(app, _args(tmp_path)).exit_code == 0
    assert not any(f.exists() for f in stale) and final[0].exists()


def test_uniref70_write_raises_on_missing_test_id(tmp_path, calls):
    assert runner.invoke(app, _args(tmp_path)).exit_code == 0
    wd = tmp_path / "wd"
    sc = json.loads((wd / "screen.done").read_text())
    sc["test_ids"].append("BOGUS")
    (wd / "screen.done").write_text(json.dumps(sc))
    (wd / "write.done").unlink()
    (wd / "write.started").write_text(str((tmp_path / "out").resolve()))
    r = runner.invoke(app, _args(tmp_path))
    assert r.exit_code != 0 and isinstance(r.exception, ValueError)
    assert "not in the pool" in str(r.exception)
