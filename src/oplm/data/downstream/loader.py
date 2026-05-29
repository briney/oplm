"""Downstream-task loading.

Loads labeled-sequence benchmarks (TAPE / ProteinGLUE) into examples and defines
the label-collation contract (per-residue label tensors padded with ``-100``;
sequence-level regression/classification targets). Embedding extraction and the
supervised head belong to the eval harness.
"""

from __future__ import annotations
