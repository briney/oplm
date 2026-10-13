"""Real structure fixtures for the fold data tests (tests/fixtures/fold/README.md).

The pytest fixtures built on these constants (``parsed``, ``fixture_dataset``, ...) live in
``tests/fold/conftest.py`` so tests directly under ``tests/fold/`` can use them too.
"""

from __future__ import annotations

from pathlib import Path

FOLD_FIXTURES = Path(__file__).parents[2] / "fixtures" / "fold"
MMCIF_DIR = FOLD_FIXTURES / "mmcif"  # the archive root for preprocessing tests
FIXTURE_IDS = ("1crn", "1ejg", "4hhb", "2ds7", "1jb6", "1l2y")
