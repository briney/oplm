"""Public surface for the OPLM model package.

OPLM is an encoder-only protein language model built on a configurable
pre-norm transformer backbone. A single :class:`OplmConfig` selects every
architectural variant — norm operator (LayerNorm / RMSNorm), norm placement
strategy (pre / sandwich / hybrid / post-SDPA), full vs. partial RoPE, optional
QK-norm, SwiGLU feed-forward, optional Canon depthwise convolutions, and
sqrt-depth residual scaling — so the same code path covers the whole design
space. Attention runs through PyTorch's ``scaled_dot_product_attention`` — a
fused FlashAttention / memory-efficient kernel on CUDA, the math backend on
CPU. The package exposes the backbone
(:class:`OplmModel`) and the task heads (:class:`OplmForMaskedLM`,
:class:`OplmForSequenceClassification`, :class:`OplmForTokenClassification`),
all registered with the HuggingFace Auto* classes and carrying an ESM-C-style
``tokenize`` / ``encode`` / ``logits`` convenience API via
:class:`EsmcCompatMixin`. Internal building blocks (norm, rope, embedding, ffn,
conv, attention, transformer, masking) live in their own modules and are
re-exported here for convenience.

See ``docs/MODEL_ARCHITECTURE.md`` for the full architecture specification.
"""

from __future__ import annotations

from .attention import OplmAttention
from .configuration_oplm import OplmConfig
from .conv import CanonConv, resolve_canon_kernel_sizes
from .embedding import OplmEmbedding, cls_pool, mean_pool
from .ffn import GEGLU, SquaredReLU, SwiGLU, make_ffn, round_up_to
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
    "GEGLU",
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
    "SquaredReLU",
    "SwiGLU",
    "cls_pool",
    "make_ffn",
    "mean_pool",
    "resolve_canon_kernel_sizes",
    "round_up_to",
]
