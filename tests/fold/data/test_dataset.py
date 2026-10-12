"""Preprocessing to shards, index, splits and manifest (addendum §4; spec §4.7, §7.2, §11)."""

from __future__ import annotations

import gzip
import json
import os
import shutil
from functools import partial
from typing import TYPE_CHECKING

import pyarrow.parquet as pq
import pytest
import torch
import yaml
from torch.utils.data import DataLoader

import oplm.data.sequence.dataset as seq_dataset
from oplm.data.sequence.dataset import InterleavedDataset
from oplm.fold.data.ccd import ChemCompTable
from oplm.fold.data.collate import collate_rows
from oplm.fold.data.complex import COMPLEX_SCHEMA, row_to_complex
from oplm.fold.data.dataset import (
    PreprocessConfig,
    StructureDataset,
    assign_splits,
    discover_entries,
    preprocess,
    read_target_list,
)
from tests.fold.data.conftest import FIXTURE_IDS, FOLD_FIXTURES, MMCIF_DIR

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from oplm.fold.data.complex import MolecularComplex


def index_by_id(dataset: Path) -> dict[str, list[dict[str, object]]]:
    rows = pq.read_table(dataset / "index.parquet").to_pylist()
    out: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        out.setdefault(str(row["id"]), []).append(row)
    return out


def test_preprocess_layout_counts_and_schema(
    fixture_dataset: Path, parsed: Callable[[str], MolecularComplex]
) -> None:
    manifest = json.loads((fixture_dataset / "manifest.json").read_text())
    assert manifest["schema_version"] == 1 and manifest["parse_version"] == 1
    assert manifest["accepted"] == 6 and manifest["rejected"] == {}
    expected_dropped: dict[str, int] = {}
    for eid in FIXTURE_IDS:  # 4hhb: 2 ligand entities + water; 2ds7: 1 + water; 1jb6: water
        for kind, n in parsed(eid).metadata.dropped.items():
            expected_dropped[kind] = expected_dropped.get(kind, 0) + n
    assert manifest["dropped"] == expected_dropped == {"ligand": 3, "water": 3}
    assert manifest["row_group_size"] == 2 and manifest["config"]["cutoff"] == "2021-09-30"
    assert manifest["licenses"] == {"wwPDB": "CC0 1.0", "CCD": "CC0 1.0"}
    assert manifest["ccd"]["components"] == 45 and len(manifest["ccd"]["sha256"]) == 64
    assert manifest["archive"] == "mmcif" and manifest["ccd"]["path"] == "ccd_subset.cif"
    assert manifest["config"]["archive"] == "mmcif" and manifest["config"]["out"] == "dataset"
    assert manifest["config"]["heldout"] == {"cameo": "cameo.txt", "casp14": "casp14.txt"}
    assert manifest["heldout_missing"] == {"cameo": [], "casp14": []}
    assert "basenames" in manifest["note"]
    shards = sorted((fixture_dataset / "shards").glob("shard-*.parquet"))
    assert len(shards) >= 2 and [s["file"] for s in manifest["shards"]] == [s.name for s in shards]
    assert sum(s["rows"] for s in manifest["shards"]) == 6
    for shard in shards:
        pf = pq.ParquetFile(shard)
        assert pf.schema_arrow.equals(COMPLEX_SCHEMA)
        assert all(
            pf.metadata.row_group(i).num_rows <= 2 for i in range(pf.metadata.num_row_groups)
        )
    checksums = (fixture_dataset / "checksums.txt").read_text().splitlines()
    assert len(checksums) == 6 and all(len(line.split()[0]) == 64 for line in checksums)
    assert all((MMCIF_DIR / line.split()[1]).exists() for line in checksums)  # archive-relative
    assert (fixture_dataset / "ccd_table.json").exists()
    card = (fixture_dataset / "README.md").read_text()
    assert card.startswith("---\nlicense: cc0-1.0") and "6 entries" in card
    front_matter = yaml.safe_load(card.split("---\n")[1])  # the Hub parses this block
    assert front_matter["license"] == "cc0-1.0"
    assert front_matter["pretty_name"].startswith("OPLM fold data")
    assert json.loads((fixture_dataset / "targets" / "cameo.json").read_text()) == [
        {"id": "1crn", "chain": "A"}
    ]


def test_index_rows_point_at_their_shard_rows(fixture_dataset: Path) -> None:
    by_id = index_by_id(fixture_dataset)
    assert sorted(by_id) == ["1crn", "1ejg", "1jb6", "1l2y", "2ds7", "4hhb"]
    assert len(by_id["4hhb"]) == 4 and [r["chain_index"] for r in by_id["4hhb"]] == [0, 1, 2, 3]
    row = by_id["2ds7"][0]
    assert row["length"] == 51 and row["resolved_count"] == 40 and row["n_chains"] == 1
    assert row["resolution"] == pytest.approx(2.5) and row["method"] == "X-RAY DIFFRACTION"
    table = pq.read_table(fixture_dataset / "shards" / str(row["shard"]))
    c = row_to_complex(table.slice(int(row["row"]), 1).to_pylist()[0])
    assert c.metadata.id == "2ds7" and c.chains[int(row["chain_index"])].chain_id == row["chain_id"]
    assert by_id["1l2y"][0]["resolution"] is None


def test_splits_with_the_real_cutoff(fixture_dataset: Path) -> None:
    split = {eid: {r["split"] for r in rows} for eid, rows in index_by_id(fixture_dataset).items()}
    assert split == {
        "1crn": {"cameo"},
        "1jb6": {"casp14"},
        "1ejg": {"train"},  # crambin too, but L at residue 25 (vs I in 1crn): no exact match
        "4hhb": {"train"},
        "2ds7": {"train"},
        "1l2y": {"train"},
    }


def test_splits_with_an_early_cutoff_pick_valid(fixture_dataset_early_cutoff: Path) -> None:
    split = {
        eid: {r["split"] for r in rows}
        for eid, rows in index_by_id(fixture_dataset_early_cutoff).items()
    }
    assert split["4hhb"] == {"train"} and split["2ds7"] == {"valid"}
    assert split["1ejg"] == {"post_cutoff"} and split["1l2y"] == {"post_cutoff"}
    manifest = json.loads((fixture_dataset_early_cutoff / "manifest.json").read_text())
    assert manifest["splits"] == {"cameo": 1, "casp14": 1, "train": 1, "valid": 1, "post_cutoff": 2}


def test_assign_splits_valid_is_seeded_and_capped() -> None:
    def entry(eid: str, date: str, res: float | None = 2.0, n: int = 1, length: int = 100) -> dict:
        return {
            "id": eid,
            "release_date": date,
            "resolution": res,
            "n_chains": n,
            "length": length,
            "sequence": eid * 10,
            "split": "",
        }

    rows = [entry(f"p{i:03d}", "2022-01-01") for i in range(10)] + [
        entry("old1", "2020-01-01"),
        entry("big1", "2022-01-01", n=2),
        entry("low1", "2022-01-01", res=3.0),
        entry("nmr1", "2022-01-01", res=None),
        entry("short", "2022-01-01", length=20),
        entry("none1", None),
        entry("none2", ""),  # the parser yields '' for a '?' release date
    ]
    assign_splits(rows, cutoff="2021-09-30", heldout={}, valid_size=4, valid_seed=0)
    chosen = sorted(r["id"] for r in rows if r["split"] == "valid")
    assert len(chosen) == 4 and all(c.startswith("p") for c in chosen)
    assert {r["id"]: r["split"] for r in rows if not r["id"].startswith("p")} == {
        "old1": "train",
        "big1": "post_cutoff",
        "low1": "post_cutoff",
        "nmr1": "post_cutoff",
        "short": "post_cutoff",
        "none1": "undated",
        "none2": "undated",
    }
    again = [dict(r, split="") for r in rows]
    assign_splits(again, cutoff="2021-09-30", heldout={}, valid_size=4, valid_seed=0)
    assert sorted(r["id"] for r in again if r["split"] == "valid") == chosen
    other = [dict(r, split="") for r in rows]
    assign_splits(other, cutoff="2021-09-30", heldout={}, valid_size=4, valid_seed=1)
    assert sorted(r["id"] for r in other if r["split"] == "valid") != chosen


def test_heldout_ids_are_case_insensitive(tmp_path: Path) -> None:
    (tmp_path / "t.txt").write_text("4HHB_B\n1L2Y\n\n# comment\n")
    assert read_target_list(tmp_path / "t.txt") == [("4hhb", "B"), ("1l2y", None)]
    rows = [
        {
            "id": "4hhb",
            "release_date": "1984-07-17",
            "resolution": 1.74,
            "n_chains": 4,
            "length": 141,
            "sequence": "VLSP",
            "split": "",
        },
        {
            "id": "abcd",
            "release_date": "1990-01-01",
            "resolution": 2.0,
            "n_chains": 1,
            "length": 100,
            "sequence": "VLSP",
            "split": "",
        },  # same sequence as a held-out chain
    ]
    assign_splits(
        rows, cutoff="2021-09-30", heldout={"cameo": {"4hhb"}}, valid_size=0, valid_seed=0
    )
    assert rows[0]["split"] == "cameo" and rows[1]["split"] == "excluded_heldout_sequence"


def test_duplicate_entry_ids_raise(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    (archive / "a").mkdir(parents=True)
    (archive / "b").mkdir()
    shutil.copy(MMCIF_DIR / "1crn.cif.gz", archive / "a" / "1crn.cif.gz")
    shutil.copy(MMCIF_DIR / "1crn.cif.gz", archive / "b" / "1CRN.cif.gz")
    with pytest.raises(ValueError, match="1crn"):
        discover_entries(archive)
    assert [p.name for p in discover_entries(MMCIF_DIR)] == [
        "1crn.cif.gz",
        "1ejg.cif.gz",
        "1jb6.cif.gz",
        "1l2y.cif.gz",
        "2ds7.cif.gz",
        "4hhb.cif.gz",
    ]
    (tmp_path / "ids.txt").write_text("4HHB\n1l2y\n")
    assert [p.name for p in discover_entries(MMCIF_DIR, ids=tmp_path / "ids.txt")] == [
        "1l2y.cif.gz",
        "4hhb.cif.gz",
    ]


def test_rejections_are_counted_not_fatal(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    archive.mkdir()
    for name in ("1crn.cif.gz", "1l2y.cif.gz"):
        shutil.copy(MMCIF_DIR / name, archive / name)
    with gzip.open(archive / "zzzz.cif.gz", "wt") as fh:
        fh.write("not a cif\n")
    manifest = preprocess(
        PreprocessConfig(
            archive=archive, ccd=FOLD_FIXTURES / "ccd_subset.cif", out=tmp_path / "out", workers=1
        )
    )
    assert manifest["accepted"] == 2 and manifest["rejected"] == {"parse_error": 1}
    assert len(manifest["rejections"]) == 1
    rejection = manifest["rejections"][0]
    assert rejection["id"] == "zzzz.cif.gz" and rejection["reason"] == "parse_error"
    assert "detail" in rejection


def _small_config(tmp_path: Path, **overrides: object) -> PreprocessConfig:
    """One fixture (1crn) in a fresh archive, preprocessed in-process into ``tmp_path / "out"``."""
    archive = tmp_path / "archive"
    archive.mkdir(exist_ok=True)
    shutil.copy(MMCIF_DIR / "1crn.cif.gz", archive / "1crn.cif.gz")
    kwargs: dict[str, object] = {
        "archive": archive,
        "ccd": FOLD_FIXTURES / "ccd_subset.cif",
        "out": tmp_path / "out",
        "workers": 1,
    }
    return PreprocessConfig(**(kwargs | overrides))  # type: ignore[arg-type]


def test_unexpected_worker_exception_is_an_internal_error_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("oplm.fold.data.dataset._chain_stats", boom)
    cfg = _small_config(tmp_path)
    shutil.copy(MMCIF_DIR / "1l2y.cif.gz", cfg.archive / "1l2y.cif.gz")
    manifest = preprocess(cfg)
    assert manifest["accepted"] == 0 and manifest["rejected"] == {"internal_error": 2}
    assert manifest["rejections"][0]["detail"] == "RuntimeError: boom"
    assert (cfg.out / "manifest.json").exists() and len(manifest["shards"]) == 0


def test_checksums_name_files_relative_to_the_archive_root(tmp_path: Path) -> None:
    cfg = _small_config(tmp_path)  # a divided archive, as wwPDB ships: <mid>/<id>.cif.gz
    (cfg.archive / "cr").mkdir()
    (cfg.archive / "1crn.cif.gz").rename(cfg.archive / "cr" / "1crn.cif.gz")
    preprocess(cfg)
    digest, name = (cfg.out / "checksums.txt").read_text().split()
    assert name == "cr/1crn.cif.gz" and len(digest) == 64


def test_malformed_cutoff_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="2021-9-30"):
        preprocess(_small_config(tmp_path, cutoff="2021-9-30"))


def test_cutoff_is_normalised_in_the_manifest(tmp_path: Path) -> None:
    manifest = preprocess(_small_config(tmp_path, cutoff="20210930"))
    assert manifest["config"]["cutoff"] == "2021-09-30"


def test_empty_archive_raises(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    cfg = _small_config(tmp_path, archive=tmp_path / "empty")
    with pytest.raises(ValueError, match="no mmCIF files"):
        preprocess(cfg)
    assert not cfg.out.exists()


def test_rerun_into_an_existing_output_raises(tmp_path: Path) -> None:
    cfg = _small_config(tmp_path)
    preprocess(cfg)
    with pytest.raises(FileExistsError, match="shards"):
        preprocess(cfg)


def test_preprocess_logs_progress(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO", logger="oplm.fold.data.dataset")
    preprocess(_small_config(tmp_path))
    assert any("1/1 parsed" in r.getMessage() for r in caplog.records)


def test_heldout_targets_missing_from_the_snapshot_are_recorded(tmp_path: Path) -> None:
    (tmp_path / "cameo.txt").write_text("1crn_A\n9xyz\n")
    cfg = _small_config(tmp_path, archive=MMCIF_DIR, heldout={"cameo": tmp_path / "cameo.txt"})
    manifest = preprocess(cfg)
    assert manifest["heldout_missing"] == {"cameo": ["9xyz"]}
    assert manifest["splits"]["cameo"] == 1


def test_workers_load_the_saved_ccd_json_not_the_cif(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[object] = []
    original = ChemCompTable.parse

    def counting(cls: type[ChemCompTable], path: Path) -> ChemCompTable:
        calls.append(path)
        return original(path)

    monkeypatch.setattr(ChemCompTable, "parse", classmethod(counting))
    manifest = preprocess(_small_config(tmp_path))
    assert manifest["accepted"] == 1 and len(calls) == 1  # _load_ccd only; workers read the JSON


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads files regardless of mode")
def test_unreadable_input_is_a_rejection_not_an_abort(tmp_path: Path) -> None:
    cfg = _small_config(tmp_path)
    shutil.copy(MMCIF_DIR / "1l2y.cif.gz", cfg.archive / "1l2y.cif.gz")
    victim = cfg.archive / "1l2y.cif.gz"
    victim.chmod(0o000)
    try:
        manifest = preprocess(cfg)
    finally:
        victim.chmod(0o644)
    assert manifest["accepted"] == 1 and manifest["rejected"] == {"internal_error": 1}
    assert manifest["rejections"][0]["id"] == "1l2y.cif.gz"
    assert len((cfg.out / "checksums.txt").read_text().splitlines()) == 1  # only the readable file


@pytest.fixture
def single_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(seq_dataset, "_resolve_distributed_context", lambda: (0, 1, 0, 1))


def stream(ds: StructureDataset | InterleavedDataset) -> list[tuple[str, list[int], int]]:
    """(id, crop residue indices, conformer hash) per row, so rotations are compared too."""
    return [
        (
            str(r["id"]),
            r["example"].features.residue_index[0].tolist(),
            hash(r["example"].features.ref_pos.numpy().tobytes()),
        )
        for r in ds
    ]


def test_eligibility_filters_and_len(fixture_dataset: Path, single_process: None) -> None:
    train = StructureDataset(fixture_dataset, split="train", crop_size=16)
    assert len(train) == 4 and train.total_length == 4  # 1ejg, 4hhb, 2ds7, 1l2y
    assert len(StructureDataset(fixture_dataset, split="train", max_chain_length=50)) == 2
    assert len(StructureDataset(fixture_dataset, split="train", min_resolved_residues=45)) == 2
    assert len(StructureDataset(fixture_dataset, split="casp14")) == 1
    assert len(StructureDataset(fixture_dataset, split="cameo")) == 1
    assert len(StructureDataset(fixture_dataset, split=None)) == 6
    assert len(StructureDataset(fixture_dataset, split="nope")) == 0


def test_examples_are_seeded_single_chain_crops(
    fixture_dataset: Path, single_process: None
) -> None:
    ds = StructureDataset(fixture_dataset, split="train", crop_size=16, seed=0)
    ds.set_epoch(0)
    rows = list(ds)
    assert len(rows) == 4
    assert sorted(str(r["id"]).split("_")[0] for r in rows) == ["1ejg", "1l2y", "2ds7", "4hhb"]
    by_id = index_by_id(fixture_dataset)
    for r in rows:
        ex = r["example"]
        f = ex.features
        entry, chain_id = ex.id.split("_")
        sequence = next(c["sequence"] for c in by_id[entry] if c["chain_id"] == chain_id)
        assert f.num_tokens <= 16 and f.num_chains == 1 and f.asym_id.eq(0).all()
        assert f.lm_input_ids.shape[1] == len(sequence) + 2  # the LM row is the full chain
        assert ex.targets.coords.shape[1] == f.num_atoms
        starts = f.atom_to_token[0, f.atom_mask[0]]
        for token in range(f.num_tokens):  # rotated, centered conformers (spec §4.3)
            atoms = f.ref_pos[0, f.atom_mask[0]][starts == token]
            assert torch.allclose(atoms.mean(0), torch.zeros(3), atol=1e-4)
    hhb = next(r for r in rows if str(r["id"]).startswith("4hhb"))
    assert hhb["example"].features.lm_input_ids.shape[1] in (143, 148)  # a full alpha/beta chain
    again = StructureDataset(fixture_dataset, split="train", crop_size=16, seed=0)
    again.set_epoch(0)
    assert stream(again) == stream(ds)
    ds.set_epoch(1)
    assert stream(ds) != stream(again)  # a new epoch gives new chains/crops
    other = StructureDataset(fixture_dataset, split="train", crop_size=16, seed=1)
    other.set_epoch(0)
    assert stream(other) != stream(again)


@pytest.mark.parametrize("skip", [0, 1, 2, 5])
def test_resumed_stream_is_baseline_suffix(
    fixture_dataset: Path, single_process: None, skip: int
) -> None:
    baseline = StructureDataset(fixture_dataset, split=None, crop_size=16, seed=3)
    baseline.set_epoch(2)
    expected = stream(baseline)
    assert len(expected) == 6
    resumed = StructureDataset(fixture_dataset, split=None, crop_size=16, seed=3)
    resumed.set_epoch(2)
    resumed.set_resume_skip(batches_in_epoch=skip, per_rank_batch=1, num_workers=1)
    assert stream(resumed) == expected[skip:]
    resumed.clear_resume_skip()
    assert stream(resumed) == expected


def test_resume_with_two_dataloader_workers_matches_batch_suffix(fixture_dataset: Path) -> None:
    collate = partial(collate_rows, crop_size=16)

    def loader(ds: StructureDataset) -> DataLoader:
        return DataLoader(
            ds, batch_size=1, collate_fn=collate, num_workers=2, persistent_workers=False
        )

    def batches(ds: StructureDataset) -> list[tuple[list[str], list[int]]]:
        return [(b.ids, b.features.residue_index[0].tolist()) for b in loader(ds)]

    baseline = StructureDataset(fixture_dataset, split=None, crop_size=16, seed=0)
    baseline.set_epoch(0)
    expected = batches(baseline)
    assert len(expected) == 6
    k = 2  # a full worker-cycle boundary (k % num_workers == 0)
    resumed = StructureDataset(fixture_dataset, split=None, crop_size=16, seed=0)
    resumed.set_epoch(0)
    resumed.set_resume_skip(batches_in_epoch=k, per_rank_batch=1, num_workers=2)
    assert batches(resumed) == expected[k:]


def test_interleaved_sources_resume(fixture_dataset: Path, single_process: None) -> None:
    def mix() -> InterleavedDataset:
        a = StructureDataset(fixture_dataset, split="train", crop_size=16, seed=0)
        b = StructureDataset(fixture_dataset, split="casp14", crop_size=16, seed=0)
        return InterleavedDataset([a, b], [0.7, 0.3], num_samples=8, seed=0)

    baseline = mix()
    baseline.set_epoch(0)
    expected = stream(baseline)
    assert len(expected) == 8
    resumed = mix()
    resumed.set_epoch(0)
    resumed.set_resume_skip(batches_in_epoch=3, per_rank_batch=1, num_workers=1)
    assert stream(resumed) == expected[3:]


def test_missing_shard_raises_and_empty_shards_are_skipped(
    fixture_dataset: Path, tmp_path: Path, single_process: None
) -> None:
    cameo = StructureDataset(fixture_dataset, split="cameo", crop_size=16)
    assert len(cameo) == 1 and [r["id"] for r in cameo] == ["1crn_A"]  # other shards have no rows
    broken = tmp_path / "broken"
    shutil.copytree(fixture_dataset, broken)
    victim = sorted((broken / "shards").glob("shard-*.parquet"))[-1]
    victim.unlink()
    with pytest.raises(FileNotFoundError, match=victim.name):
        StructureDataset(broken, split=None)
    with pytest.raises(FileNotFoundError, match=victim.name):  # even if it holds no eligible rows
        StructureDataset(broken, split="cameo")
