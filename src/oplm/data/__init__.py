"""OPLM data utilities."""

from __future__ import annotations

from oplm.data.tokenizer import ProteinTokenizer

__all__ = [
    "ProteinTokenizer",
    "ShardedProteinDataset",
    "InterleavedDataset",
    "MLMCollator",
    "build_train_dataloader",
]


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    """Lazy imports for modules that depend on pyarrow."""
    if name in ("ShardedProteinDataset", "InterleavedDataset"):
        from oplm.data.dataset import InterleavedDataset, ShardedProteinDataset

        globals()["ShardedProteinDataset"] = ShardedProteinDataset
        globals()["InterleavedDataset"] = InterleavedDataset
        return globals()[name]
    if name == "MLMCollator":
        from oplm.data.collate import MLMCollator

        globals()["MLMCollator"] = MLMCollator
        return MLMCollator
    if name == "build_train_dataloader":
        from oplm.data.loader import build_train_dataloader

        globals()["build_train_dataloader"] = build_train_dataloader
        return build_train_dataloader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
