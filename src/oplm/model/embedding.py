"""Token embedding plus mean / CLS pooling helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from .norm import make_norm

if TYPE_CHECKING:
    from .configuration_oplm import OplmConfig

__all__ = ["OplmEmbedding", "mean_pool", "cls_pool"]


class OplmEmbedding(nn.Module):
    """Token embedding lookup with an optional post-embedding norm.

    The token IDs map straight into a learnable `(V, D)` embedding table; no
    positional embedding is added at this stage (RoPE is applied inside the
    attention sublayer). When `config.post_embed_norm` is `True`, a
    `make_norm(norm_type, D)` instance is applied to the lookup output before
    it enters the residual stream.
    """

    def __init__(self, config: OplmConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        if config.post_embed_norm:
            self.post_norm: nn.Module = make_norm(
                config.norm_type, config.hidden_size, eps=config.norm_eps
            )
        else:
            self.post_norm = nn.Identity()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # (B, T) int64 -> (B, T, D)
        x = self.embed_tokens(input_ids)
        return self.post_norm(x)


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mask-aware mean over the sequence dimension.

    Args:
        hidden: `(B, T, D)` tensor.
        attention_mask: `(B, T)` tensor with `1` at real tokens and `0` at
            pad positions. Cast to `hidden`'s dtype before the multiply.

    Returns:
        `(B, D)` tensor. Rows whose mask is entirely zero are returned as
        zeros (the denominator is clamped to at least `1`).
    """
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask).sum(dim=1)
    counts = attention_mask.sum(dim=1, keepdim=True).to(hidden.dtype).clamp(min=1)
    return summed / counts


def cls_pool(hidden: torch.Tensor) -> torch.Tensor:
    """Return the `<cls>` position (token index 0) of every row.

    Args:
        hidden: `(B, T, D)` tensor.

    Returns:
        `(B, D)` tensor.
    """
    return hidden[:, 0, :]
