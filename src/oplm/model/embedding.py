"""Token embedding plus mean / CLS pooling helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from .norm import make_norm

if TYPE_CHECKING:
    from .configuration_oplm import OplmConfig

__all__ = ["OplmEmbedding", "mean_pool", "cls_pool"]

# Floor on the `(1 - observed_mask_ratio)` denominator so an all-`<mask>` row
# (observed ratio 1.0) yields a finite scale instead of a division by zero. The
# zeroed `<mask>` rows it multiplies are zero anyway, so the exact floor value is
# immaterial as long as it keeps the product finite.
_MASK_DROPOUT_EPS = 1e-6


class OplmEmbedding(nn.Module):
    """Token embedding lookup with optional mask dropout and post-embedding norm.

    The token IDs map straight into a learnable `(V, D)` embedding table; no
    positional embedding is added at this stage (RoPE is applied inside the
    attention sublayer).

    When `config.mask_dropout` is `True`, every `<mask>` embedding row is zeroed
    and the surviving rows are rescaled so the per-sequence embedding magnitude
    matches the masking rate the model expects at training time (see
    `_apply_mask_dropout`). This runs deterministically — it is not a stochastic
    regularizer — so it applies in both train and eval. When `config.post_embed_norm`
    is `True`, a `make_norm(norm_type, D)` instance is applied (after mask dropout)
    before the result enters the residual stream.
    """

    def __init__(self, config: OplmConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.mask_dropout = bool(config.mask_dropout)
        self.mask_dropout_reference_ratio = float(config.mask_dropout_reference_ratio)
        self.mask_token_id = int(config.mask_token_id)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        if config.post_embed_norm:
            self.post_norm: nn.Module = make_norm(
                config.norm_type, config.hidden_size, eps=config.norm_eps
            )
        else:
            self.post_norm = nn.Identity()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed `input_ids`, optionally apply mask dropout, then the post-norm.

        Args:
            input_ids: `(B, T)` int64 token IDs.
            attention_mask: Optional `(B, T)` mask with `1` at real tokens and
                `0` at pads. Only consumed when `mask_dropout` is enabled (to
                count real tokens per row); all-ones is assumed when absent.

        Returns:
            `(B, T, D)` embeddings.
        """
        # (B, T) int64 -> (B, T, D)
        x = self.embed_tokens(input_ids)
        if self.mask_dropout:
            x = self._apply_mask_dropout(x, input_ids, attention_mask)
        return self.post_norm(x)

    def _apply_mask_dropout(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Zero `<mask>` embedding rows and rescale the survivors per sequence.

        For each row, `observed_mask_ratio = count(<mask>) / count(real tokens)`
        and every embedding is scaled by
        `(1 - reference_ratio) / (1 - observed_mask_ratio)` — inverted-dropout-style
        compensation that holds the expected embedding magnitude constant as the
        actual `<mask>` fraction varies around the training reference. Ratios are
        computed in fp32; the denominator and real-token count are clamped so
        all-pad and all-`<mask>` rows stay finite.

        Args:
            x: `(B, T, D)` raw embedding lookup.
            input_ids: `(B, T)` token IDs (used to locate `<mask>` positions).
            attention_mask: `(B, T)` real/pad mask, or `None` (treated as all real).

        Returns:
            `(B, T, D)` embeddings with `<mask>` rows zeroed and survivors rescaled.
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        is_mask = input_ids == self.mask_token_id  # (B, T) bool
        real = attention_mask > 0  # (B, T) bool
        real_count = real.sum(dim=1).clamp(min=1).float()  # (B,)
        mask_count = (is_mask & real).sum(dim=1).float()  # (B,)
        observed_ratio = mask_count / real_count  # (B,)
        denom = (1.0 - observed_ratio).clamp(min=_MASK_DROPOUT_EPS)  # (B,)
        scale = ((1.0 - self.mask_dropout_reference_ratio) / denom).to(x.dtype)  # (B,)

        keep = (~is_mask).unsqueeze(-1).to(x.dtype)  # (B, T, 1)
        return x * keep * scale.view(-1, 1, 1)


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
