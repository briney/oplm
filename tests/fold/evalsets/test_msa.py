"""ColabFold MSA client and MSA preprocessing (no network: a fake fetch stands in)."""

from __future__ import annotations

import io
import json
import tarfile
import urllib.error

import pytest
import torch

from oplm.fold.evalsets.msa import (
    encode_msa,
    fetch_msas,
    filter_coverage,
    filter_gap_columns,
    neff,
    parse_a3m,
)

SEQ = "MKVLA"
UNIREF = f">101\n{SEQ}\n>u1\nMKVLA\n>u2\nMKaVLA\n"
ENV = f">101\n{SEQ}\n>e1\nMK-LA\n\x00"


def _tar() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, text in (("uniref.a3m", UNIREF), ("bfd.mgnify30.metaeuk30.smag30.a3m", ENV)):
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _fake(submits: list[dict], statuses: list[str]):
    submits, statuses = list(submits), list(statuses)

    def fetch(url: str, body: bytes | None) -> bytes:
        if url.endswith("/ticket/msa"):
            assert body is not None and b"q=%3E" in body and b"mode=env" in body
            return json.dumps(submits.pop(0)).encode()
        if "/ticket/" in url:
            # No "id" in poll replies, as on the lab's MSA server: the client keeps the ticket.
            return json.dumps({"status": statuses.pop(0)}).encode()
        if url.endswith("/result/download/abc"):
            return _tar()
        raise AssertionError(url)

    return fetch


def test_fetch_msa_polls_then_combines():
    sleeps: list[float] = []
    fake = _fake(
        [
            {"status": "RATELIMIT"},
            {"status": "RATELIMIT"},
            {"id": "abc", "status": "PENDING"},
        ],
        ["RUNNING", "COMPLETE"],
    )
    [a3m] = fetch_msas([SEQ], "http://x", fetch=fake, sleep=sleeps.append, poll_seconds=5.0)
    assert a3m.count(">") == 1 + 2 + 1  # env query not repeated
    assert "\x00" not in a3m
    assert sleeps == [5.0, 10.0, 5.0, 5.0]  # backs off on RATELIMIT, resets after
    assert parse_a3m(a3m)[-1] == "MK-LA"


def test_fetch_msa_error_raises():
    fake = _fake([{"id": "abc", "status": "ERROR"}], [])
    with pytest.raises(RuntimeError, match="ERROR"):
        fetch_msas([SEQ], "http://x", fetch=fake, sleep=lambda s: None)


def test_parse_a3m_removes_insertions():
    assert parse_a3m(">q\nAC-D\n>h\nAcC-D\n") == ["AC-D", "AC-D"]


def test_encode_maps_noncanonical_to_gap_and_caps_depth():
    out = encode_msa(["LAX-"])
    assert out.dtype == torch.int8
    assert out.tolist() == [[0, 1, 20, 20]]
    assert encode_msa(["LA"] * 5, max_seqs=3).shape == (3, 2)


def test_filter_gap_columns():
    msa = torch.tensor([[0, 20, 1], [1, 20, 2], [2, 3, 3], [3, 20, 4]], dtype=torch.int8)
    kept, cols = filter_gap_columns(msa)
    assert cols.tolist() == [0, 2]
    assert kept.shape == (4, 2)


def test_filter_coverage_drops_fragments():
    msa = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 20], [0, 20, 20, 3]], dtype=torch.int8)
    assert filter_coverage(msa).tolist() == msa[:2].tolist()  # 4/4 and 3/4 covered; 2/4 dropped


def test_neff_identical_and_distinct():
    same = torch.zeros(4, 6, dtype=torch.int8)
    n, w = neff(same)
    assert n == pytest.approx(1.0) and w.shape == (4,)
    distinct = torch.tensor([[i] * 6 for i in range(4)], dtype=torch.int8)
    assert neff(distinct)[0] == pytest.approx(4.0)
    assert neff(same[:1])[0] == 1.0


def test_fetch_msa_retries_transient_http_error():
    inner = _fake([{"id": "abc", "status": "COMPLETE"}], [])
    calls = {"n": 0}

    def flaky(url: str, body: bytes | None) -> bytes:
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(url, 503, "unavailable", {}, None)  # ty: ignore[invalid-argument-type]
        if calls["n"] == 2:
            raise TimeoutError
        return inner(url, body)

    sleeps: list[float] = []
    assert fetch_msas([SEQ], "http://x", fetch=flaky, sleep=sleeps.append)[0].startswith(">101")
    assert sleeps == [5.0, 10.0]


def test_fetch_msa_nontransient_http_error_raises():
    def bad(url: str, body: bytes | None) -> bytes:
        raise urllib.error.HTTPError(url, 404, "nope", {}, None)  # ty: ignore[invalid-argument-type]

    with pytest.raises(urllib.error.HTTPError):
        fetch_msas([SEQ], "http://x", fetch=bad, sleep=lambda s: None)


def test_fetch_msa_resubmits_when_reply_has_no_id():
    fake = _fake([{"status": "UNKNOWN"}, {"id": "abc", "status": "COMPLETE"}], [])
    assert fetch_msas([SEQ], "http://x", fetch=fake, sleep=lambda s: None)[0].count(">") == 4


def test_fetch_msas_splits_a_batch_by_query_header():
    seq2 = "GHIKL"
    # Queries out of order, as ColabFold's own client allows; env lacks query 102.
    uniref = f">102\n{seq2}\n>v1\nGHIKL\n\x00>101\n{SEQ}\n>u1\nMKVLA\n\x00"
    env = f">101\n{SEQ}\n>e1\nMK-LA\n"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, text in (("uniref.a3m", uniref), ("bfd.mgnify30.metaeuk30.smag30.a3m", env)):
            info = tarfile.TarInfo(name)
            info.size = len(text.encode())
            tf.addfile(info, io.BytesIO(text.encode()))
    bodies: list[bytes] = []

    def fetch(url: str, body: bytes | None) -> bytes:
        if url.endswith("/ticket/msa"):
            assert body is not None
            bodies.append(body)
            return json.dumps({"id": "abc", "status": "COMPLETE"}).encode()
        return buf.getvalue()

    first, second = fetch_msas([SEQ, seq2], "http://x", fetch=fetch, sleep=lambda s: None)
    assert parse_a3m(first) == [SEQ, "MKVLA", "MK-LA"]
    assert parse_a3m(second) == [seq2, "GHIKL"]
    assert b"%3E101%0AMKVLA%0A%3E102%0AGHIKL" in bodies[0]


def test_fetch_msas_missing_query_raises():
    fake = _fake([{"id": "abc", "status": "COMPLETE"}], [])  # serves query 101 only
    with pytest.raises(RuntimeError, match="no MSA for query 1"):
        fetch_msas([SEQ, "GHIKL"], "http://x", fetch=fake, sleep=lambda s: None)


def test_fetch_msas_redownloads_a_truncated_result():
    # randy reports COMPLETE once result.tar.gz exists, while it may still be being written.
    downloads = [_tar()[:40], _tar()]
    sleeps: list[float] = []

    def fetch(url: str, body: bytes | None) -> bytes:
        if url.endswith("/ticket/msa"):
            return json.dumps({"id": "abc", "status": "COMPLETE"}).encode()
        return downloads.pop(0)

    [a3m] = fetch_msas([SEQ], "http://x", fetch=fetch, sleep=sleeps.append)
    assert a3m.count(">") == 4 and not downloads and sleeps == [5.0]
