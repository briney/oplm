"""Pad-mask helpers, flex_attention mask_mod factories, conv-input zeroing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch.nn.attention.flex_attention import BlockMask

__all__ = ["prepare_attention_mask", "make_flex_block_mask", "zero_pad_positions"]


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


def make_flex_block_mask(
    attention_mask: torch.Tensor,
    num_heads: int,
) -> BlockMask:
    """Build a `BlockMask` for `flex_attention` from a `(B, T)` padding mask.

    The encoder is bidirectional, so the mask only enforces KV padding: any
    query may attend to any key whose position is a real (non-pad) token.

    Args:
        attention_mask: `(B, T)` tensor with `1` at real-token positions and
            `0` at pad positions.
        num_heads: Number of attention heads. Passed through to
            `create_block_mask` as an explicit positive integer (the head
            dimension is not actually used by the closure, but the documented
            signature requires it).

    Returns:
        A `BlockMask` ready to be passed to
        `torch.nn.attention.flex_attention.flex_attention`.
    """
    from torch.nn.attention.flex_attention import create_block_mask

    batch_size, seq_len = attention_mask.shape

    def mask_mod(b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor):
        del h, q_idx  # bidirectional: no causal restriction, no head dependence
        return attention_mask[b, kv_idx] == 1

    return create_block_mask(
        mask_mod,
        B=batch_size,
        H=num_heads,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=attention_mask.device,
    )


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
