"""Held-out monomer evaluation for the structure head (addendum §6, spec §8; docs/FOLD.md §10).

Importing this package registers the ``fold_monomer`` eval type. ``oplm.fold.training`` imports
it, so every fold run can name the type under ``data.eval``; MLM tooling never imports fold code.
"""

from __future__ import annotations

from oplm.fold.eval import tasks  # noqa: F401  (registers fold_monomer)
