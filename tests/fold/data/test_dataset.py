"""Preprocessing to shards, index, splits and manifest (addendum §4; spec §4.7, §7.2, §11)."""

from __future__ import annotations

import gzip
import json
import shutil
from typing import TYPE_CHECKING

import pyarrow.parquet as pq
import pytest

from oplm.fold.data.complex import COMPLEX_SCHEMA, row_to_complex
from oplm.fold.data.dataset import (
    PreprocessConfig,
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
    assert (fixture_dataset / "ccd_table.json").exists()
    card = (fixture_dataset / "README.md").read_text()
    assert card.startswith("---\nlicense: cc0-1.0") and "6 entries" in card
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
    assert manifest["rejections"] == [{"id": "zzzz.cif.gz", "reason": "parse_error"}]
