"""Iterable sequence datasets.

:class:`ShardedProteinDataset` streams rows from one or more parquet shards with
reproducible, rank/worker-aware shuffling and striping. :class:`InterleavedDataset`
samples across several sources according to fixed mixing fractions.
"""

from __future__ import annotations
