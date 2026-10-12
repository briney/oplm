"""Real structure fixtures for the fold data tests (tests/fixtures/fold/README.md)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from oplm.fold.data.ccd import ChemCompTable, ReferenceConformers

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
