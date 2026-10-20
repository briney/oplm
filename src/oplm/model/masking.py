"""Pad-mask helpers and conv-input zeroing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch.nn.attention.flex_attention import BlockMask

__all__ = [
    "prepare_attention_mask",
    "segment_attention_mask",
    "segment_block_mask",
    "zero_pad_positions",
]


def segment_attention_mask(segment_ids: torch.Tensor) -> torch.Tensor:
    """Block-diagonal `(B, 1, T, T)` attend mask for a packed batch (several sequences per row).

    `segment_ids` is `(B, T)`: `1..n` for the sequences packed back to back in a row, `0` at
    pads. A query attends only to keys of its own segment; the pad run is a segment of its
    own, so pad queries attend to pad keys and no row is empty. RoPE positions need no reset:
    attention scores depend only on query-key offsets, which never cross a segment.
    """
    return (segment_ids[:, :, None] == segment_ids[:, None, :])[:, None]


def segment_block_mask(segment_ids: torch.Tensor) -> BlockMask:
    """FlexAttention block mask for a packed batch: each token attends only within its segment.

    The same pattern as :func:`segment_attention_mask`, as a block-sparse mask that lets
    FlexAttention skip every 128x128 block between different sequences, so attention cost follows
    the sequences' own lengths rather than the row width. Pads (segment 0) attend only to each
    other, so no query row is empty.
    """
    from torch.nn.attention.flex_attention import create_block_mask

    def same_segment(b, h, q_idx, kv_idx):  # noqa: ANN001, ANN202 — FlexAttention mask_mod
        return segment_ids[b, q_idx] == segment_ids[b, kv_idx]

    rows, width = segment_ids.shape
    return create_block_mask(
        same_segment, B=rows, H=None, Q_LEN=width, KV_LEN=width, device=segment_ids.device
    )


def prepare_attention_mask(
    attention_mask: torch.Tensor | None,
    batch_size: int,
    seq_len: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    """Return a `(B, T)` attention mask, materializing all-ones when absent.

    Args:
        attention_mask: Optional caller-supplied mask of shape `(B, T)` with
            `1` for real tokens and `0` for pads.
        batch_size: Expected `B`. Used to validate caller input and to size the
            default mask.
        seq_len: Expected `T`. Same role as `batch_size`.
        device: Target device for the default mask. Ignored when
            `attention_mask` is provided (it is returned on its own device).
        dtype: Target dtype for the default mask. Ignored when
            `attention_mask` is provided.

    Returns:
        A `(B, T)` tensor. When `attention_mask` is `None`, an all-ones tensor
        on `device` with the requested `dtype`; otherwise the caller's tensor
        returned as-is after shape validation.

    Raises:
        ValueError: If the caller-supplied mask does not have shape
            `(batch_size, seq_len)`.
    """
    if attention_mask is None:
        return torch.ones(batch_size, seq_len, device=device, dtype=dtype)
    if attention_mask.dim() != 2 or attention_mask.shape != (batch_size, seq_len):
        raise ValueError(
            f"attention_mask has shape {tuple(attention_mask.shape)}; "
            f"expected ({batch_size}, {seq_len})."
        )
    return attention_mask


def zero_pad_positions(x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Zero out pad rows in a `(B, T, D)` tensor without touching real tokens.

    Used by `CanonConv` before the depthwise 1-D convolution so that pad
    positions cannot leak into real-token channels via the kernel's receptive
    field.

    Args:
        x: `(B, T, D)` tensor.
        attention_mask: `(B, T)` tensor with `1` at real tokens, `0` at pads.

    Returns:
        `(B, T, D)` tensor with pad rows zeroed.
    """
    return x * attention_mask.unsqueeze(-1).to(x.dtype)
