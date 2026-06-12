"""OPLM data tooling.

Loaders, tokenization access, collation, and the dataset/dataloader builders for
pretraining and evaluation. The public surface re-exported below is consumed by
the trainer and, later, the eval harness. This package must never import
:mod:`oplm.eval`.
"""

from __future__ import annotations

from oplm.data.config import parse_eval_configs, parse_train_configs
from oplm.data.sequence.collate import MLMCollator, tokenize_and_pad
from oplm.data.sequence.dataset import InterleavedDataset, ShardedProteinDataset
from oplm.data.sequence.loaders import (
    build_sequence_eval_dataloader,
    build_train_dataloader,
)
from oplm.data.structure.loader import StructureData, load_structures
from oplm.data.tokenizer import get_tokenizer
from oplm.data.variant.loader import (
    VariantAssay,
    load_clinical_variant_assays,
    load_variant_assays,
)

__all__ = [
    "InterleavedDataset",
    "MLMCollator",
    "ShardedProteinDataset",
    "StructureData",
    "VariantAssay",
    "build_sequence_eval_dataloader",
    "build_train_dataloader",
    "get_tokenizer",
    "load_clinical_variant_assays",
    "load_structures",
    "load_variant_assays",
    "parse_eval_configs",
    "parse_train_configs",
    "tokenize_and_pad",
]
