"""Import-hygiene guards for the ``oplm.data`` package (Phase 0.3).

``oplm.data`` (and every submodule) must import cleanly without pulling in
``oplm.eval`` — the eval harness depends on ``oplm.data``, not the reverse.
Checks run in a fresh subprocess so they are not masked by other tests in the
session having already imported ``oplm.eval``.
"""

from __future__ import annotations

import subprocess
import sys

_SUBMODULES = [
    "oplm.data",
    "oplm.data.tokenizer",
    "oplm.data.config",
    "oplm.data.sequence",
    "oplm.data.sequence.dataset",
    "oplm.data.sequence.collate",
    "oplm.data.sequence.loaders",
    "oplm.data.structure",
    "oplm.data.structure.loader",
    "oplm.data.variant",
    "oplm.data.variant.loader",
    "oplm.data.downstream",
    "oplm.data.downstream.loader",
]


def _import_in_subprocess(module: str) -> subprocess.CompletedProcess[str]:
    """Import ``module`` in a fresh interpreter and report any ``oplm.eval`` leak."""
    code = (
        "import sys\n"
        f"import {module}\n"
        "assert 'oplm.eval' not in sys.modules, "
        f"'{module} transitively imported oplm.eval'\n"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )


def test_data_package_imports_without_eval() -> None:
    """``import oplm.data`` succeeds and does not import ``oplm.eval``."""
    result = _import_in_subprocess("oplm.data")
    assert result.returncode == 0, result.stderr


def test_submodules_import_without_eval() -> None:
    """Every ``oplm.data`` submodule imports cleanly and eval-free."""
    for module in _SUBMODULES:
        result = _import_in_subprocess(module)
        assert result.returncode == 0, f"{module}: {result.stderr}"
