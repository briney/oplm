"""Public surface for the OPLM model package — see docs/MODEL_ARCHITECTURE.md."""

from __future__ import annotations

from .attention import OplmAttention
from .configuration_oplm import OplmConfig
from .conv import CanonConv, resolve_canon_kernel_sizes
from .embedding import OplmEmbedding, cls_pool, mean_pool
from .ffn import SwiGLU, make_ffn, round_up_to
from .modeling_oplm import (
    EsmcCompatMixin,
    OplmForMaskedLM,
    OplmForSequenceClassification,
    OplmForTokenClassification,
    OplmMLMHead,
    OplmModel,
    OplmPreTrainedModel,
)
from .outputs import LogitsConfig, LogitsOutput
from .tokenization_oplm import OplmTokenizerFast
from .transformer import OplmBlock, OplmStack

__all__ = [
    "CanonConv",
    "EsmcCompatMixin",
    "LogitsConfig",
    "LogitsOutput",
    "OplmAttention",
    "OplmBlock",
    "OplmConfig",
    "OplmEmbedding",
    "OplmForMaskedLM",
    "OplmForSequenceClassification",
    "OplmForTokenClassification",
    "OplmMLMHead",
    "OplmModel",
    "OplmPreTrainedModel",
    "OplmStack",
    "OplmTokenizerFast",
    "SwiGLU",
    "cls_pool",
    "make_ffn",
    "mean_pool",
    "resolve_canon_kernel_sizes",
    "round_up_to",
]
