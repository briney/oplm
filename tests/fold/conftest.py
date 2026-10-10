"""Fixture-directory gate for the parity tests (spec §9: skip with an explicit reason)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    env = os.environ.get("OPLM_FOLD_FIXTURES")
    if not env:
        pytest.skip(
            "OPLM_FOLD_FIXTURES is unset: ESMFold2 golden fixtures unavailable (docs/FOLD.md §7)"
        )
    path = Path(env)
    if not (path / "manifest.json").exists():
        pytest.skip(f"no manifest.json under OPLM_FOLD_FIXTURES={path}")
    return path
