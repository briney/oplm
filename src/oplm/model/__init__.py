"""Public surface for the OPLM model package — see docs/MODEL_ARCHITECTURE.md."""

from __future__ import annotations

from .embedding import OplmEmbedding, cls_pool, mean_pool
from .outputs import LogitsConfig, LogitsOutput

__all__ = [
    "LogitsConfig",
    "LogitsOutput",
    "OplmEmbedding",
    "cls_pool",
    "mean_pool",
]
