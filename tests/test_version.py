"""The public version must identify the installed distribution."""

from __future__ import annotations

from importlib.metadata import version

import oplm


def test_version_matches_distribution() -> None:
    assert oplm.__version__ == version("oplm")
