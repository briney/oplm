import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from oplm.data.build.uniref70 import (
    SCREEN_ARGS,
    Cluster,
    LowSurvivalError,
    cluster_uniref90,
    iter_clusters,
    iter_rows,
    screen_against,
    select_eval,
    shuffle_clusters,
    write_dataset,
)


def test_iter_clusters_parses_all_seqs_format(tmp_path):
    f = tmp_path / "a.fasta"
    f.write_text(">B\n>B x\nAAAA\n>A desc n=2\nAAAC\n>C\n>C y\nWWWW\n")
    out = list(iter_clusters(f))
    assert [c.rep_id for c in out] == ["B", "C"]
    assert out[0].member_ids == ["B", "A"]
    assert out[0].member_sequences == ["AAAA", "AAAC"]
    assert out[1].member_ids == ["C"] and out[1].member_sequences == ["WWWW"]


def test_iter_clusters_ids_and_wrapped_sequences(tmp_path):
    seq = "ACDEFGHIKL" * 15
    wrapped = "\n".join(seq[i : i + 60] for i in range(0, len(seq), 60))
    f = tmp_path / "a.fasta"
    hdr = "UniRef90_P1 Cluster: x n=3 Tax=y RepID=Z"
    f.write_text(f">UniRef90_P1\n>{hdr}\n{wrapped}\n>UniRef90_P2 Cluster: x n=3\n{wrapped}\n")
    (c,) = iter_clusters(f)
    assert c.member_ids == ["UniRef90_P1", "UniRef90_P2"]
    assert c.member_sequences == [seq, seq]


def test_rep_is_first_member(tmp_path):
    f = tmp_path / "a.fasta"
    f.write_text(">R\n>R\nAC\n>Q\nAD\n>S\nAE\n")
    (c,) = iter_clusters(f)
    assert c.rep_id == "R" and c.member_ids[0] == "R"


@pytest.mark.slow
def test_cluster_uniref90_end_to_end(tmp_path):
    if shutil.which("mmseqs") is None:
        pytest.skip("mmseqs not on PATH")
    a = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAA"
    c = "WWWWPPPPGGGGHHHHCCCCWWWWPPPPGGGGHHHHCCCCWWWWPPPPGGGGHHHHCCCCWWWW"
    d = "MDEFHRKLNQSTVWYACMDEFHRKLNQSTVWYACMDEFHRKLNQSTVWYACMDEFHRKLNQSTVW"
    seqs = {"a1": a, "a2": a[:-1] + "Q", "c1": c, "c2": c[:-1] + "A", "d1": d, "e1": d[::-1]}
    fa = tmp_path / "in.fa"
    fa.write_text("".join(f">{k}\n{v}\n" for k, v in seqs.items()))
    wd = tmp_path / "wd"
    wd.mkdir()
    out = cluster_uniref90(fa, wd, mmseqs="mmseqs", threads=1)
    clusters = list(iter_clusters(out))
    ids = [i for c in clusters for i in c.member_ids]
    assert sorted(ids) == sorted(seqs)
    assert len(clusters) == 4


def _clusters(n):
    return [Cluster(f"R{i:05d}", [f"R{i:05d}", f"M{i}"], ["AAA", "CCC"]) for i in range(n)]


def _order(tmp_path, name, seed, n=1000, n_buckets=8):
    parts = shuffle_clusters(_clusters(n), tmp_path / name, seed=seed, n_buckets=n_buckets)
    return [r["sequence_id"] for r in iter_rows(parts)]


def test_shuffle_is_a_permutation(tmp_path):
    order = _order(tmp_path, "a", 0)
    inp = [c.rep_id for c in _clusters(1000)]
    assert sorted(order) == inp and order != inp
    row = next(iter_rows(sorted((tmp_path / "a").glob("*.parquet"))))
    assert row["sequence"] == "AAA" and row["member_ids"][0] == row["sequence_id"]
    assert not list((tmp_path / "a").glob("*.tmp")) and not list((tmp_path / "a").glob("*.pass1"))


def test_shuffle_is_deterministic(tmp_path):
    assert _order(tmp_path, "a", 1) == _order(tmp_path, "b", 1)
    assert _order(tmp_path, "c", 2) != _order(tmp_path, "a", 1)


def test_shuffle_breaks_input_order(tmp_path):
    pos = {f"R{i:05d}": i for i in range(1000)}
    out = np.array([pos[r] for r in _order(tmp_path, "a", 0)])
    assert abs(np.corrcoef(np.arange(1000), out)[0, 1]) < 0.1


def test_shuffle_resumes_and_ignores_flush_size(tmp_path, monkeypatch):
    base = _order(tmp_path, "a", 3)
    import oplm.data.build.uniref70 as u

    monkeypatch.setattr(u, "_FLUSH_ROWS", 7)
    assert _order(tmp_path, "b", 3) == base
    d = tmp_path / "b"
    (d / "part-00002.parquet").unlink()  # lost bucket: pass 1 reruns, finished buckets kept
    assert [
        r["sequence_id"]
        for r in iter_rows(shuffle_clusters(_clusters(1000), d, seed=3, n_buckets=8))
    ] == base


def _rows(n):
    return [
        {
            "sequence_id": f"R{i}",
            "sequence": "A",
            "member_ids": [f"R{i}"],
            "member_sequences": ["A"],
        }
        for i in range(n)
    ]


def test_select_eval_skips_hits_and_keeps_order():
    rows = _rows(20)
    seen = []

    def screen(cands):
        seen.append([c["sequence_id"] for c in cands])
        return {"R1", "R4", "R5"}

    chosen, stats = select_eval(iter(rows), pool=10, n_eval=4, screen=screen, min_survival=0.05)
    assert seen == [[f"R{i}" for i in range(10)]]
    assert [r["sequence_id"] for r in chosen] == ["R0", "R2", "R3", "R6"]
    assert stats == {"pool": 10, "hits": 3, "survivors": 7, "survival_rate": 0.7}


def test_select_eval_low_survival_raises():
    hit_all_but_one = lambda cands: {c["sequence_id"] for c in cands[1:]}  # noqa: E731
    with pytest.raises(LowSurvivalError):
        select_eval(_rows(100), pool=100, n_eval=1, screen=hit_all_but_one, min_survival=0.05)
    assert issubclass(LowSurvivalError, RuntimeError)


def test_select_eval_too_few_survivors_raises():
    with pytest.raises(ValueError, match="grow"):
        select_eval(_rows(10), pool=10, n_eval=8, screen=lambda c: {"R0", "R1", "R2"})


def _multi_clusters(n):
    # cluster i has 1 + i % 4 members; ids unique across clusters
    out = []
    for i in range(n):
        ids = [f"R{i:05d}"] + [f"M{i}_{j}" for j in range(i % 4)]
        out.append(Cluster(ids[0], ids, ["ACD" * (j + 1) for j in range(len(ids))]))
    return out


def test_write_dataset_partitions(tmp_path):
    parts = shuffle_clusters(_multi_clusters(200), tmp_path / "parts", seed=0, n_buckets=4)
    shuffled = list(iter_rows(parts))
    eval_rows = [shuffled[i] for i in (0, 3, 50, 199)]
    out = tmp_path / "out"
    summary = write_dataset(parts, eval_rows, out, rows_per_shard=30, row_group_size=7)

    ev = pq.read_table(out / "eval.parquet").to_pylist()
    assert ev == eval_rows
    shards = sorted((out / "train").iterdir())
    assert [p.name for p in shards] == [f"shard-{i:05d}.parquet" for i in range(7)]
    sizes = [pq.ParquetFile(p).metadata.num_rows for p in shards]
    assert sizes == [30] * 6 + [16]
    assert max(pq.ParquetFile(shards[0]).metadata.row_group(g).num_rows for g in range(5)) <= 7
    train = list(iter_rows(shards))
    eval_ids = {r["sequence_id"] for r in eval_rows}
    assert [r["sequence_id"] for r in train] == [
        r["sequence_id"] for r in shuffled if r["sequence_id"] not in eval_ids
    ]
    reps = [r["sequence_id"] for r in train] + [r["sequence_id"] for r in ev]
    assert sorted(reps) == sorted(c.rep_id for c in _multi_clusters(200))
    eval_members = {m for r in ev for m in r["member_ids"]}
    assert not eval_members & {m for r in train for m in r["member_ids"]}
    assert not list(out.rglob("*.tmp"))

    t = summary["train"]
    assert t["rows"] == 196 and summary["eval"]["rows"] == 4 and summary["train_shards"] == 7
    assert t["member_rows"] == sum(len(r["member_ids"]) for r in train)
    assert t["rep_residues"] == sum(len(r["sequence"]) for r in train)
    assert t["member_residues"] == sum(len(s) for r in train for s in r["member_sequences"])
    sizes = sorted(len(r["member_ids"]) for r in train)
    mpc = t["members_per_cluster"]
    assert (mpc["min"], mpc["max"]) == (1, 4)
    assert mpc["median"] == (sizes[97] + sizes[98]) / 2
    assert mpc["mean"] == pytest.approx(sum(sizes) / len(sizes))
    assert mpc["log2_histogram"] == {
        "1": sizes.count(1),
        "2-3": sizes.count(2) + sizes.count(3),
        "4-7": sizes.count(4),
    }


def test_chunk_fasta_spans_shards_in_order(tmp_path):
    ref = tmp_path / "ref"
    ref.mkdir()
    ids = [f"S{i}" for i in range(7)]
    for k, (a, b) in enumerate([(0, 3), (3, 4), (4, 7)]):
        pq.write_table(
            pa.table({"sequence_id": ids[a:b], "sequence": ["MK"] * (b - a)}),
            ref / f"x-{k}.parquet",
        )
    from oplm.data.build.uniref70 import _ref_shards, _write_chunk_fasta

    shards = _ref_shards(ref)
    got = []
    for lo in range(0, 7, 2):
        fa = tmp_path / "c.fa"
        _write_chunk_fasta(shards, lo, lo + 2, fa)
        got += [line[1:] for line in fa.read_text().splitlines() if line.startswith(">")]
    assert got == ids


def test_screen_args_do_not_cap_hits():
    args = dict(zip(SCREEN_ARGS[::2], SCREEN_ARGS[1::2], strict=True))
    assert int(args["--max-seqs"]) >= 100_000 and float(args["-e"]) >= 1e6
    assert (args["--min-seq-id"], args["-c"], args["--cov-mode"]) == ("0.7", "0.8", "0")


_CANDS = [{"sequence_id": "A", "sequence": "MK", "member_ids": ["A"], "member_sequences": ["MK"]}]


def _no_mmseqs(ref, wd):  # runs up to the first mmseqs call (key handling happens before it)
    with pytest.raises(RuntimeError, match="not found"):
        screen_against(_CANDS, ref, wd, mmseqs="/nonexistent/mmseqs", threads=1, chunk_rows=2)


def test_screen_key_covers_reference(tmp_path):
    ref, wd = tmp_path / "ref", tmp_path / "wd"
    ref.mkdir()
    pq.write_table(pa.table({"sequence_id": ["u"], "sequence": ["MK"]}), ref / "a.parquet")
    _no_mmseqs(ref, wd)
    (wd / "hits-00000.txt").write_text("A\n")
    _no_mmseqs(ref, wd)
    assert (wd / "hits-00000.txt").exists()  # same inputs: finished chunks kept
    pq.write_table(pa.table({"sequence_id": ["v"], "sequence": ["MK"]}), ref / "b.parquet")
    _no_mmseqs(ref, wd)
    assert not (wd / "hits-00000.txt").exists()  # reference changed: hits discarded


def test_screen_empty_reference_raises(tmp_path):
    ref = tmp_path / "ref"
    ref.mkdir()
    pq.write_table(
        pa.table({"sequence_id": pa.array([], pa.string()), "sequence": pa.array([], pa.string())}),
        ref / "a.parquet",
    )
    with pytest.raises(ValueError, match="no rows"):
        screen_against(_CANDS, ref, tmp_path / "wd", mmseqs="mmseqs", threads=1, chunk_rows=2)


@pytest.mark.slow
def test_screen_against_end_to_end(tmp_path):
    if shutil.which("mmseqs") is None:
        pytest.skip("mmseqs not on PATH")
    a1 = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAA"
    a2 = "MDEFHRKLNQSTVWYACMDEFHRKLNQSTVWYACMDEFHRKLNQSTVWYACMDEFHRKLNQSTVW"
    b1 = "WWWWPPPPGGGGHHHHCCCCWWWWPPPPGGGGHHHHCCCCWWWWPPPPGGGGHHHHCCCCWWWW"
    near_a2 = a2[:20] + "LLL" + a2[23:]  # ~95% identical to a2
    cands = [
        {
            "sequence_id": "A",
            "sequence": a1,
            "member_ids": ["A", "A2"],
            "member_sequences": [a1, a2],
        },
        {"sequence_id": "B", "sequence": b1, "member_ids": ["B"], "member_sequences": [b1]},
    ]
    ref = tmp_path / "ref"
    ref.mkdir()
    unrelated = ["MSTNPKPQRKTKRNTNRRPQDVKFPGG" * 2, "MGSSHHHHHHSSGLVPRGSHMASMTGGQQMG" * 2]
    pq.write_table(
        pa.table({"sequence_id": ["u0", "u1"], "sequence": unrelated}), ref / "a.parquet"
    )
    pq.write_table(pa.table({"sequence_id": ["hit"], "sequence": [near_a2]}), ref / "b.parquet")
    wd = tmp_path / "wd"
    assert screen_against(cands, ref, wd, mmseqs="mmseqs", threads=1, chunk_rows=2) == {"A"}
    assert sorted(p.name for p in wd.glob("hits-*.txt")) == ["hits-00000.txt", "hits-00001.txt"]
    assert not list(wd.glob("chunk-*"))
    (wd / "hits-00000.txt").unlink()  # restart: only the missing chunk reruns
    assert screen_against(cands, ref, wd, mmseqs="mmseqs", threads=1, chunk_rows=2) == {"A"}
