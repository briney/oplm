"""OPLM data tooling.

Loaders, tokenization access, collation, and the dataset/dataloader builders for
pretraining and evaluation. The public surface (re-exported here in Phase 9) is
consumed by the trainer and, later, the eval harness. This package must never
import :mod:`oplm.eval`.
"""
