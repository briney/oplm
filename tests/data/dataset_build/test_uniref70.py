import json
import shutil
from pathlib import Path

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


@pytest.mark.parametrize("text", [">R\n>Q\nAD\n>R\nAC\n", ">R\n>S\n>S\nAC\n"])
def test_rep_not_first_raises(tmp_path, text):
    f = tmp_path / "a.fasta"
    f.write_text(text)
    with pytest.raises(ValueError, match="cluster R"):
        list(iter_clusters(f))


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


def test_shuffle_refuses_stale_parts(tmp_path):
    d = tmp_path / "s"
    shuffle_clusters(_clusters(100), d, seed=0, n_buckets=8)
    assert json.loads((d / "params.json").read_text()) == {"seed": 0, "n_buckets": 8}
    with pytest.raises(ValueError, match="another shuffle"):
        shuffle_clusters(_clusters(100), d, seed=1, n_buckets=8)
    with pytest.raises(ValueError, match="another shuffle"):
        shuffle_clusters(_clusters(100), d, seed=0, n_buckets=4)
    (d / "params.json").unlink()  # parts without a record (e.g. an older build)
    with pytest.raises(ValueError, match="another shuffle"):
        shuffle_clusters(_clusters(100), d, seed=0, n_buckets=8)


def test_shuffle_records_counts(tmp_path):
    d = tmp_path / "s"
    shuffle_clusters(_multi_clusters(10), d, seed=0, n_buckets=2)
    counts = json.loads((d / "counts.json").read_text())
    assert counts == {"clusters": 10, "members": sum(1 + i % 4 for i in range(10))}


def test_guards_reject_nonpositive_sizes(tmp_path):
    with pytest.raises(ValueError, match="n_buckets"):
        shuffle_clusters(_clusters(3), tmp_path / "s", seed=0, n_buckets=0)
    with pytest.raises(ValueError, match="rows_per_shard"):
        write_dataset([], {}, tmp_path / "o", rows_per_shard=0)


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


def _mixed(n):
    # even i: singleton, odd i: gte2 (2 members)
    return [
        {
            "sequence_id": f"R{i}",
            "sequence": "A",
            "member_ids": [f"R{i}"] + ([f"M{i}"] if i % 2 else []),
            "member_sequences": ["A"] * (2 if i % 2 else 1),
        }
        for i in range(n)
    ]


def _ids(rows):
    return [r["sequence_id"] for r in rows]


def test_select_eval_per_stratum_first_n_then_next_n():
    seen = []
    walked = []

    def rows():
        for r in _mixed(100):
            walked.append(r["sequence_id"])
            yield r

    def screen(cands):
        seen.append(_ids(cands))
        return {"R2", "R3", "R9"}

    chosen, stats = select_eval(rows(), pool=4, n_eval=2, n_test=1, screen=screen)
    # singletons: R0 R2 R4 R6 -> survivors R0 R4 R6; gte2: R1 R3 R5 R7 -> survivors R1 R5 R7
    assert seen == [["R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7"]]
    assert walked == [f"R{i}" for i in range(8)]  # stopped as soon as both pools filled
    assert _ids(chosen["eval_singletons"]) == ["R0", "R4"]
    assert _ids(chosen["test_singletons"]) == ["R6"]
    assert _ids(chosen["eval_gte2"]) == ["R1", "R5"] and _ids(chosen["test_gte2"]) == ["R7"]
    assert stats["singletons"] == {"pool": 4, "hits": 1, "survivors": 3, "survival_rate": 0.75}
    assert stats["gte2"] == {"pool": 4, "hits": 1, "survivors": 3, "survival_rate": 0.75}
    assert (stats["pool"], stats["hits"], stats["survivors"]) == (8, 2, 6)
    assert stats["survival_rate"] == 6 / 8


def test_select_eval_n_test_zero_gives_empty_test():
    chosen, _ = select_eval(
        iter(_mixed(20)), pool=5, n_eval=2, n_test=0, screen=lambda c: {"R0", "R1"}
    )
    assert _ids(chosen["eval_singletons"]) == ["R2", "R4"]
    assert chosen["test_singletons"] == [] and chosen["test_gte2"] == []


def test_select_eval_low_survival_in_one_stratum_raises():
    odd = lambda cands: {c["sequence_id"] for c in cands if len(c["member_ids"]) > 1}  # noqa: E731
    with pytest.raises(LowSurvivalError, match="gte2"):
        select_eval(iter(_mixed(100)), pool=50, n_eval=1, n_test=0, screen=odd)
    assert issubclass(LowSurvivalError, RuntimeError)


def test_select_eval_low_survival_beats_shortfall():
    # singletons: 2 survivors < n_eval; gte2: survival 0 -> LowSurvivalError first
    def screen(cands):
        return {
            c["sequence_id"]
            for c in cands
            if len(c["member_ids"]) > 1 or c["sequence_id"] in {"R0", "R2", "R4"}
        }

    with pytest.raises(LowSurvivalError, match="gte2"):
        select_eval(iter(_mixed(20)), pool=5, n_eval=4, n_test=0, screen=screen)


def test_select_eval_too_few_survivors_names_stratum():
    hit = lambda cands: {c["sequence_id"] for c in cands if c["sequence_id"] in {"R1", "R3"}}  # noqa: E731
    with pytest.raises(ValueError, match=r"gte2.*grow"):
        select_eval(iter(_mixed(20)), pool=5, n_eval=2, n_test=2, screen=hit)


def _multi_clusters(n):
    # cluster i has 1 + i % 4 members; ids unique across clusters
    out = []
    for i in range(n):
        ids = [f"R{i:05d}"] + [f"M{i}_{j}" for j in range(i % 4)]
        out.append(Cluster(ids[0], ids, ["ACD" * (j + 1) for j in range(len(ids))]))
    return out


def _split(rows):
    return [r for r in rows if len(r["member_ids"]) == 1], [
        r for r in rows if len(r["member_ids"]) > 1
    ]


def test_write_dataset_partitions(tmp_path):
    parts = shuffle_clusters(_multi_clusters(200), tmp_path / "parts", seed=0, n_buckets=4)
    shuffled = list(iter_rows(parts))
    one, many = _split(shuffled)
    held = {
        "eval_singletons": [one[0], one[9]],
        "eval_gte2": [many[0], many[5], many[2]],
        "test_singletons": [one[3], one[1]],
        "test_gte2": [many[7], many[1]],
    }
    all_held = [r for rows in held.values() for r in rows]
    out = tmp_path / "out"
    summary = write_dataset(parts, held, out, rows_per_shard=30, row_group_size=7)

    for name, rows in held.items():
        assert pq.read_table(out / f"{name}.parquet").to_pylist() == rows
        assert summary[name]["rows"] == len(rows)
    assert sorted(p.name for p in out.glob("*.parquet")) == sorted(f"{n}.parquet" for n in held)
    assert summary["eval"]["rows"] == 5 and summary["test"]["rows"] == 4
    assert summary["test"]["member_rows"] == sum(
        len(r["member_ids"]) for r in held["test_singletons"] + held["test_gte2"]
    )
    assert summary["eval_gte2"]["rep_residues"] == sum(
        len(r["sequence"]) for r in held["eval_gte2"]
    )
    shards = sorted((out / "train").iterdir())
    sizes = [pq.ParquetFile(p).metadata.num_rows for p in shards]
    assert sizes == [30] * 6 + [11] and sum(sizes) == 191
    assert max(pq.ParquetFile(shards[0]).metadata.row_group(g).num_rows for g in range(5)) <= 7
    train = list(iter_rows(shards))
    held_ids = set(_ids(all_held))
    assert _ids(train) == [i for i in _ids(shuffled) if i not in held_ids]
    assert sorted(_ids(train) + _ids(all_held)) == sorted(c.rep_id for c in _multi_clusters(200))
    held_members = {m for r in all_held for m in r["member_ids"]}
    assert not held_members & {m for r in train for m in r["member_ids"]}
    assert not list(out.rglob("*.tmp"))
    t = summary["train"]
    assert t["rows"] == 191 and summary["train_shards"] == 7
    assert t["member_rows"] == sum(len(r["member_ids"]) for r in train)
    assert t["rep_residues"] == sum(len(r["sequence"]) for r in train)
    assert t["member_residues"] == sum(len(s) for r in train for s in r["member_sequences"])
    sz = sorted(len(r["member_ids"]) for r in train)
    mpc = t["members_per_cluster"]
    assert (mpc["min"], mpc["max"]) == (1, 4)
    assert mpc["median"] == (sz[95] + sz[96]) / 2
    assert mpc["mean"] == pytest.approx(sum(sz) / len(sz))
    assert mpc["log2_histogram"] == {
        "1": sz.count(1),
        "2-3": sz.count(2) + sz.count(3),
        "4-7": sz.count(4),
    }


def test_write_dataset_wrong_stratum_raises(tmp_path):
    parts = shuffle_clusters(_multi_clusters(20), tmp_path / "parts", seed=0, n_buckets=2)
    _, many = _split(list(iter_rows(parts)))
    with pytest.raises(ValueError, match="eval_singletons"):
        write_dataset(parts, {"eval_singletons": many[:1]}, tmp_path / "out", rows_per_shard=30)


def test_write_dataset_skips_empty_lists(tmp_path):
    parts = shuffle_clusters(_multi_clusters(20), tmp_path / "parts", seed=0, n_buckets=2)
    _, many = _split(list(iter_rows(parts)))
    out = tmp_path / "out"
    summary = write_dataset(
        parts, {"eval_singletons": [], "eval_gte2": many[:3]}, out, rows_per_shard=30
    )
    assert not (out / "eval_singletons.parquet").exists() and not list(out.glob("test*"))
    assert (out / "eval_gte2.parquet").exists() and "test" not in summary
    assert summary["eval_singletons"]["rows"] == 0 and summary["eval"]["rows"] == 3


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


def _cluster_parts(ref, groups):
    """Write uniref70 cluster rows; ``groups`` is one list of ``(rep, members)`` per part."""
    ref.mkdir()
    for k, rows in enumerate(groups):
        pq.write_table(
            pa.table(
                {
                    "sequence_id": [r for r, _ in rows],
                    "sequence": [m[0][1] for _, m in rows],
                    "member_ids": [[i for i, _ in m] for _, m in rows],
                    "member_sequences": [[s for _, s in m] for _, m in rows],
                }
            ),
            ref / f"part-{k:05d}.parquet",
        )
    (ref / "params.json").write_text("{}")  # the shuffle dir's sidecars are not shards
    (ref / "counts.json").write_text("{}")


def test_chunk_fasta_members_mode(tmp_path):
    from oplm.data.build.uniref70 import _ref_shards, _write_chunk_fasta

    ref = tmp_path / "ref"
    _cluster_parts(
        ref,
        [
            [("A", [("A", "MA"), ("a1", "MB")]), ("B", [("B", "MC")])],
            [("C", [("C", "MD"), ("c1", "ME"), ("c2", "MF")]), ("D", [("D", "MG")])],
        ],
    )
    shards = _ref_shards(ref)
    fa = tmp_path / "c.fa"
    _write_chunk_fasta(shards, 1, 3, fa, members=True)
    assert fa.read_text() == ">B\nMC\n>C\nMD\n>C\nME\n>C\nMF\n"


def _fake_mmseqs(monkeypatch, tsv_lines):
    import oplm.data.build.uniref70 as u

    seen = []

    def run(mmseqs, args):
        seen.append(args)
        if args[0] == "convertalis":
            Path(args[4]).write_text("".join(f"{q}\t{t}\n" for q, t in tsv_lines))

    monkeypatch.setattr(u, "run_mmseqs_cmd", run)
    return seen


_MEMBER_CANDS = [
    {
        "sequence_id": "A",
        "sequence": "MA",
        "member_ids": ["A", "A_member", "A_member2"],
        "member_sequences": ["MA", "MB", "MC"],
    }
]


@pytest.mark.parametrize(
    ("tsv", "want"),
    [
        ([("A", "A_member"), ("X", "A_member2")], {"A"}),
        ([("A", "A_member")], set()),
    ],
)
def test_screen_members_ignores_self_hits(tmp_path, monkeypatch, tsv, want):
    seen = _fake_mmseqs(monkeypatch, tsv)
    ref = tmp_path / "ref"
    _cluster_parts(
        ref, [[("A", [("A", "MA"), ("A_member", "MB"), ("A_member2", "MC")]), ("X", [("X", "MD")])]]
    )
    got = screen_against(
        _MEMBER_CANDS, ref, tmp_path / "wd", mmseqs="m", threads=1, chunk_rows=2, members=True
    )
    assert got == want
    assert [a[a.index("--format-output") + 1] for a in seen if a[0] == "convertalis"] == [
        "query,target"
    ]


def test_screen_key_covers_members_mode(tmp_path, monkeypatch):
    _fake_mmseqs(monkeypatch, [])
    ref, wd = tmp_path / "ref", tmp_path / "wd"
    _cluster_parts(ref, [[("A", [("A", "MA")])]])
    kw = {"mmseqs": "m", "threads": 1, "chunk_rows": 2}
    screen_against(_MEMBER_CANDS, ref, wd, **kw)
    (wd / "hits-00000.txt").write_text("A_member\n")  # pretend the deepclust70-mode chunk hit
    assert screen_against(_MEMBER_CANDS, ref, wd, **kw) == {"A"}  # same mode: reused
    assert screen_against(_MEMBER_CANDS, ref, wd, **kw, members=True) == set()  # rerun


@pytest.mark.slow
def test_screen_against_members_end_to_end(tmp_path):
    if shutil.which("mmseqs") is None:
        pytest.skip("mmseqs not on PATH")
    a1 = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAA"
    a2 = "MDEFHRKLNQSTVWYACMDEFHRKLNQSTVWYACMDEFHRKLNQSTVWYACMDEFHRKLNQSTVW"
    b1 = "WWWWPPPPGGGGHHHHCCCCWWWWPPPPGGGGHHHHCCCCWWWWPPPPGGGGHHHHCCCCWWWW"
    near_a2 = a2[:20] + "LLL" + a2[23:]  # ~95% identical to a2
    u0 = "MSTNPKPQRKTKRNTNRRPQDVKFPGG" * 2
    ref = tmp_path / "ref"
    _cluster_parts(
        ref,
        [
            [("A", [("A", a1), ("A2", a2)]), ("B", [("B", b1)])],
            [("U", [("U", u0)]), ("X", [("X", near_a2)])],
        ],
    )
    cands = [r for r in iter_rows(sorted(ref.glob("*.parquet"))) if r["sequence_id"] in ("A", "B")]
    wd = tmp_path / "wd"
    kw = {"mmseqs": "mmseqs", "threads": 1, "chunk_rows": 2, "members": True}
    assert screen_against(cands, ref, wd, **kw) == {"A"}
    assert sorted(p.name for p in wd.glob("hits-*.txt")) == ["hits-00000.txt", "hits-00001.txt"]
