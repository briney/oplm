"""Public surface for the OPLM model package — see docs/MODEL_ARCHITECTURE.md."""

from __future__ import annotations

from .embedding import OplmEmbedding, cls_pool, mean_pool
from .ffn import SwiGLU, make_ffn, round_up_to
from .outputs import LogitsConfig, LogitsOutput

__all__ = [
    "LogitsConfig",
    "LogitsOutput",
    "OplmEmbedding",
    "SwiGLU",
    "cls_pool",
    "make_ffn",
    "mean_pool",
    "round_up_to",
]
