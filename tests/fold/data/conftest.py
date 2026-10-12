"""Real structure fixtures for the fold data tests (tests/fixtures/fold/README.md)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from oplm.fold.data.ccd import ChemCompTable, ReferenceConformers
from oplm.fold.data.dataset import PreprocessConfig, preprocess

if TYPE_CHECKING:
    from collections.abc import Callable

    from oplm.fold.data.complex import MolecularComplex

FOLD_FIXTURES = Path(__file__).parents[2] / "fixtures" / "fold"
MMCIF_DIR = FOLD_FIXTURES / "mmcif"  # the archive root for preprocessing tests
FIXTURE_IDS = ("1crn", "1ejg", "4hhb", "2ds7", "1jb6", "1l2y")


@pytest.fixture(scope="session")
def ccd_table() -> ChemCompTable:
    return ChemCompTable.parse(FOLD_FIXTURES / "ccd_subset.cif")


@pytest.fixture(scope="session")
def conformers() -> ReferenceConformers:
    return ReferenceConformers.load()


@pytest.fixture(scope="session")
def parsed(
    ccd_table: ChemCompTable, conformers: ReferenceConformers
) -> Callable[[str], MolecularComplex]:
    """``parsed("1crn")`` -> the parsed fixture complex (cached per session)."""
    from oplm.fold.data.mmcif import Rejection, parse_mmcif

    cache: dict[str, MolecularComplex] = {}

    def _parsed(entry_id: str) -> MolecularComplex:
        if entry_id not in cache:
            result = parse_mmcif(
                MMCIF_DIR / f"{entry_id}.cif.gz", ccd=ccd_table, conformers=conformers
            )
            assert not isinstance(result, Rejection), result
            cache[entry_id] = result
        return cache[entry_id]

    return _parsed


def _preprocess_fixtures(out: Path, cutoff: str) -> Path:
    targets = out / "targets-in"
    targets.mkdir(parents=True)
    (targets / "cameo.txt").write_text("1CRN_A\n")
    (targets / "casp14.txt").write_text("# CASP14 subset\n1jb6_A\n")
    preprocess(
        PreprocessConfig(
            archive=MMCIF_DIR,
            ccd=FOLD_FIXTURES / "ccd_subset.cif",
            out=out / "dataset",
            workers=2,
            shard_bytes=50_000,  # row estimates: 4hhb 155k, the other five 61k total -> 2 shards
            row_group_size=2,
            cutoff=cutoff,
            valid_size=64,
            heldout={"cameo": targets / "cameo.txt", "casp14": targets / "casp14.txt"},
        )
    )
    return out / "dataset"


@pytest.fixture(scope="session")
def fixture_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The six fixtures preprocessed with the real cutoff: train = 1ejg, 4hhb, 2ds7, 1l2y."""
    return _preprocess_fixtures(tmp_path_factory.mktemp("fold_ds"), cutoff="2021-09-30")


@pytest.fixture(scope="session")
def fixture_dataset_early_cutoff(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Cutoff 1990-01-01: 4hhb trains, 2ds7 becomes valid, 1ejg and 1l2y are post_cutoff."""
    return _preprocess_fixtures(tmp_path_factory.mktemp("fold_ds_early"), cutoff="1990-01-01")
