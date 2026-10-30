import shutil

import numpy as np
import pytest

from oplm.data.build.uniref70 import (
    Cluster,
    cluster_uniref90,
    iter_clusters,
    iter_rows,
    shuffle_clusters,
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
