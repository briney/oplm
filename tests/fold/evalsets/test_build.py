"""End-to-end build of the eval sets from the fixture dataset and crambin CASP sources."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pyarrow.parquet as pq
import pytest
import torch
from typer.testing import CliRunner

from oplm.data import load_structures
from oplm.fold.cli import app
from oplm.fold.evalsets import build as build_module
from oplm.fold.evalsets.build import BuildConfig, _overlap, build, nested_subsets
from oplm.fold.evalsets.chain import EvalChain

if TYPE_CHECKING:
    from pathlib import Path


_SMALL = {
    "casp_length": (20, 1024),
    "cameo_length": (20, 512),
    "pdb40_length": (20, 1024),
    "cameo_subsets": (1,),
    "pdb40_subsets": (1, 2),
}


def test_overlap_counts_cameo_chains_not_clusters() -> None:
    """Two CAMEO chains in one PDB40 cluster count twice: the spec reports chains."""

    def chain(name: str, sequence: str, cluster: str | None = None) -> EvalChain:
        meta = {"cluster": cluster} if cluster else {}
        return EvalChain(name=name, sequence=sequence, atoms=[{} for _ in sequence], meta=meta)

    cameo = [chain("7aaa_A", "MKV"), chain("7bbb_A", "MKI"), chain("7ccc_A", "GGG")]
    pdb40 = [chain("7aaa_A", "MKV", cluster="s0"), chain("1xyz_A", "WWW", cluster="s9")]
    sequence_cluster = {"MKV": "s0", "MKI": "s0", "GGG": "s5", "WWW": "s9"}
    assert _overlap(cameo, pdb40, sequence_cluster) == {
        "cameo_chains_in_pdb40": 1,
        "cameo_chains_in_pdb40_clusters": 2,
    }


def test_build_end_to_end(
    fixture_dataset: Path, casp_sources: Path, fake_mmseqs: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    cfg = BuildConfig(
        fold_data=fixture_dataset,
        casp_sources=casp_sources,
        out=out,
        mmseqs=str(fake_mmseqs),
        **_SMALL,
    )
    summary = build(cfg)

    assert summary["exported"] == {
        "casp15_16": 2,
        "cameo": 1,
        "pdb40": 3,
        "cameo_1": 1,
        "pdb40_1": 1,
        "pdb40_2": 2,
    }
    names = {d: sorted(p.stem for p in (out / d).glob("*.pdb")) for d in summary["exported"]}
    assert names["casp15_16"] == ["casp15_T1104-D1", "casp16_T1201-D1"]
    assert names["pdb40"] == ["1ejg_A", "4hhb_A", "4hhb_B"]
    assert set(names["pdb40_1"]) < set(names["pdb40_2"]) <= set(names["pdb40"])
    for files in names.values():
        assert len({n.lower() for n in files}) == len(files)
    assert summary["sets"]["casp15_16"]["casp16"]["duplicate"] == 1
    assert summary["overlap"] == {"cameo_chains_in_pdb40": 0, "cameo_chains_in_pdb40_clusters": 1}

    (gapped,) = [s for s in load_structures(out / "casp15_16") if s.name == "casp16_T1201-D1"]
    assert len(gapped.sequence) == 46
    assert torch.isnan(gapped.coords[19:24]).all()

    manifest = pq.read_table(out / "manifest.parquet").to_pylist()
    assert len(manifest) == sum(summary["exported"].values())
    row = next(r for r in manifest if r["set"] == "casp15_16" and r["name"] == "casp16_T1201-D1")
    assert row["modeled"] == 41 and row["length"] == 46 and len(row["sha256"]) == 64
    assert row["source_file"] == "T0201-D1.pdb"

    sources = {p.name for p in (out / "sources").iterdir()}
    assert {"cameo.json", "pdb40.fasta", "pdb40_cluster.tsv"} <= sources
    assert {p.name for p in casp_sources.iterdir()} <= sources
    readme = (out / "README.md").read_text()
    assert "casp15_16" in readme and "fake-mmseqs" in readme


def test_build_refuses_a_non_empty_output(
    fixture_dataset: Path, casp_sources: Path, fake_mmseqs: Path, tmp_path: Path
) -> None:
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "stale.pdb").write_text("")
    cfg = BuildConfig(
        fold_data=fixture_dataset,
        casp_sources=casp_sources,
        out=tmp_path / "out",
        mmseqs=str(fake_mmseqs),
    )
    with pytest.raises(FileExistsError, match="not empty"):
        build(cfg)


def test_a_failed_preflight_leaves_no_output_so_the_retry_builds(
    fixture_dataset: Path, casp_sources: Path, fake_mmseqs: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    cfg = BuildConfig(
        fold_data=fixture_dataset,
        casp_sources=casp_sources,
        out=out,
        mmseqs=str(tmp_path / "no-such-mmseqs"),
        **_SMALL,
    )
    with pytest.raises(RuntimeError):
        build(cfg)
    assert not out.exists()

    missing = dataclasses.replace(cfg, fold_data=tmp_path / "no-dataset", mmseqs=str(fake_mmseqs))
    with pytest.raises(FileNotFoundError, match="index.parquet"):
        build(missing)
    assert not out.exists()

    summary = build(dataclasses.replace(cfg, mmseqs=str(fake_mmseqs)))
    assert summary["exported"]["pdb40"] == 3


def test_build_fails_when_a_written_file_does_not_load_back(
    fixture_dataset: Path,
    casp_sources: Path,
    fake_mmseqs: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validation is the build's guarantee: a written file that differs from the chain fails it."""
    real_write_pdb = build_module.write_pdb

    def corrupting_write_pdb(chain: EvalChain, path: Path) -> None:
        real_write_pdb(chain, path)
        if chain.name == "4hhb_A":
            lines = path.read_text().splitlines(keepends=True)
            first_ca = next(
                i for i, ln in enumerate(lines) if ln[:4] == "ATOM" and ln[12:16] == " CA "
            )
            del lines[first_ca]
            path.write_text("".join(lines))

    monkeypatch.setattr(build_module, "write_pdb", corrupting_write_pdb)
    cfg = BuildConfig(
        fold_data=fixture_dataset,
        casp_sources=casp_sources,
        out=tmp_path / "out",
        mmseqs=str(fake_mmseqs),
        **_SMALL,
    )
    with pytest.raises(RuntimeError, match="does not load back as built"):
        build(cfg)


def test_nested_subsets_are_nested_and_deterministic() -> None:
    names = [f"c{i}" for i in range(10)]
    subsets = nested_subsets(names, (2, 5, 50), seed=42)
    assert len(subsets[2]) == 2 and len(subsets[5]) == 5 and subsets[50] == sorted(names)
    assert set(subsets[2]) < set(subsets[5])
    assert nested_subsets(list(reversed(names)), (2, 5, 50), seed=42) == subsets


def test_cli_eval_sets(
    fixture_dataset: Path, casp_sources: Path, fake_mmseqs: Path, tmp_path: Path
) -> None:
    """Spec defaults: the 46-residue crambins fall below 50, so only the globins remain."""
    out = tmp_path / "out"
    result = CliRunner().invoke(
        app,
        [
            "eval-sets",
            "--fold-data",
            str(fixture_dataset),
            "--casp-sources",
            str(casp_sources),
            "--out",
            str(out),
            "--mmseqs",
            str(fake_mmseqs),
            "--threads",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert sorted(p.stem for p in (out / "pdb40").glob("*.pdb")) == ["4hhb_A", "4hhb_B"]
    assert "pdb40_2048" in result.output
