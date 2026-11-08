"""``oplm fold eval-coevolution``: dedup, resume, --no-fit, fixed bins, readable by the eval."""

from __future__ import annotations

import io
import json
import random
import shutil
import tarfile
import urllib.parse
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from oplm.eval.metrics.coevolution import NEFF_BINS, CoevolutionReference
from oplm.fold.evalsets.coevolution import CoevolutionBuildConfig, build_coevolution
from tests.fold.evalsets.conftest import CRAMBIN

FIXTURES = Path(__file__).parents[2] / "fixtures" / "eval" / "structures"


class FakeServer:
    """The ColabFold ticket protocol, answering each sequence with mutated copies of it."""

    def __init__(self, error_on: str | None = None) -> None:
        self.calls = 0
        self.error_on = error_on
        self.queries: dict[str, str] = {}

    def __call__(self, url: str, body: bytes | None) -> bytes:
        if url.endswith("/ticket/msa"):
            assert body is not None
            query = urllib.parse.parse_qs(body.decode())["q"][0].split("\n")[1]
            if query == self.error_on:
                return json.dumps({"status": "ERROR", "id": "x"}).encode()
            self.calls += 1
            ticket = f"t{self.calls}"
            self.queries[ticket] = query
            return json.dumps({"status": "COMPLETE", "id": ticket}).encode()
        query = self.queries[url.rsplit("/", 1)[1]]
        rng = random.Random(query)
        lines = [">101", query]
        for i in range(30):
            row = [rng.choice("ACDEFGHIKLMNPQRSTVWY") if rng.random() < 0.2 else c for c in query]
            lines += [f">hit{i}", "".join(row)]
        data = ("\n".join(lines) + "\n").encode()
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tf:
            for member in ("uniref.a3m", "bfd.mgnify30.metaeuk30.smag30.a3m"):
                info = tarfile.TarInfo(member)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
        return buffer.getvalue()


@pytest.fixture
def sets_root(tmp_path: Path) -> Path:
    root = tmp_path / "sets"
    for name, files in (("set_a", ("1CRN", "1L2Y")), ("set_b", ("1CRN",))):
        (root / name).mkdir(parents=True)
        for f in files:
            shutil.copy(FIXTURES / f"{f}.pdb", root / name / f"{f}.pdb")
    return root


def _cfg(tmp_path: Path, sets_root: Path, server: FakeServer, **kw) -> CoevolutionBuildConfig:
    return CoevolutionBuildConfig(
        sets_root=sets_root,
        out=tmp_path / "coevolution",
        sets=("set_a", "set_b"),
        fetch=server,
        **kw,
    )


def test_build_end_to_end(tmp_path: Path, sets_root: Path) -> None:
    server = FakeServer()
    cfg = _cfg(tmp_path, sets_root, server)
    summary = build_coevolution(cfg)
    assert server.calls == 2  # deduped by sequence
    potts, bin_ = CoevolutionReference.load(cfg.out).lookup(CRAMBIN)  # type: ignore[misc]
    assert potts.shape == (46, 46) and bin_ in NEFF_BINS
    assert summary["neff_bin_edges"]
    assert pq.read_table(cfg.out / "manifest.parquet").num_rows == 3
    assert (cfg.out / "README.md").is_file() and (cfg.out / "summary.json").is_file()
    assert "fetch" not in summary["config"]
    json.dumps(summary)


def test_build_resumes_without_refetch_or_refit(tmp_path: Path, sets_root: Path) -> None:
    cfg = _cfg(tmp_path, sets_root, FakeServer())
    build_coevolution(cfg)
    files = sorted((cfg.out / "couplings").iterdir())
    before = [f.stat().st_mtime_ns for f in files]
    again = FakeServer()
    build_coevolution(_cfg(tmp_path, sets_root, again))
    assert again.calls == 0
    assert [f.stat().st_mtime_ns for f in files] == before


def test_no_fit_stops_after_msas(tmp_path: Path, sets_root: Path) -> None:
    cfg = _cfg(tmp_path, sets_root, FakeServer(), fit=False)
    build_coevolution(cfg)
    assert len(list((cfg.out / "msas").iterdir())) == 2
    assert not (cfg.out / "couplings").exists()
    assert not (cfg.out / "manifest.parquet").exists()


def test_fixed_bin_edges_are_used(tmp_path: Path, sets_root: Path) -> None:
    cfg = _cfg(tmp_path, sets_root, FakeServer(), bin_edges=(0.0, 1e9))
    build_coevolution(cfg)
    assert set(pq.read_table(cfg.out / "manifest.parquet")["neff_bin"].to_pylist()) == {"mid"}


def test_server_error_names_the_sha256(tmp_path: Path, sets_root: Path) -> None:
    import hashlib

    cfg = _cfg(tmp_path, sets_root, FakeServer(error_on=CRAMBIN))
    with pytest.raises(RuntimeError, match=hashlib.sha256(CRAMBIN.encode()).hexdigest()):
        build_coevolution(cfg)


def test_resume_with_changed_settings_raises(tmp_path: Path, sets_root: Path) -> None:
    build_coevolution(_cfg(tmp_path, sets_root, FakeServer()))
    with pytest.raises(ValueError, match=r"sequence [0-9a-f]{64}") as exc:
        build_coevolution(_cfg(tmp_path, sets_root, FakeServer(), max_seqs=10))
    assert "max_seqs" in str(exc.value)


def test_missing_couplings_are_refit(tmp_path: Path, sets_root: Path) -> None:
    cfg = _cfg(tmp_path, sets_root, FakeServer())
    build_coevolution(cfg)
    victim = next((cfg.out / "couplings").iterdir())
    victim.unlink()
    build_coevolution(_cfg(tmp_path, sets_root, FakeServer()))
    assert victim.is_file()
