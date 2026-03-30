"""Attention-mask normalization helpers."""

from __future__ import annotations

import torch
from torch import Tensor


def normalize_attention_mask(attention_mask: Tensor | None) -> Tensor | None:
    """Normalize public attention masks to the encoder's internal format.

    Supported inputs:
    - ``None``
    - Standard keep masks shaped ``(B, T)`` with bool or integer values
    - Pre-expanded masks shaped ``(B, 1, T, T)`` or ``(B, 1, 1, T)``

    Returns:
        ``None`` or a normalized mask that is either 4D bool keep-mask or a
        4D floating additive mask.
    """
    if attention_mask is None:
        return None

    if attention_mask.ndim == 2:
        return attention_mask.to(dtype=torch.bool).unsqueeze(1).unsqueeze(1)

    if attention_mask.ndim != 4:
        raise ValueError("attention_mask must have shape (B, T), (B, 1, 1, T), or (B, 1, T, T)")

    if attention_mask.dtype == torch.bool or not attention_mask.is_floating_point():
        return attention_mask.to(dtype=torch.bool)

    return attention_mask
