"""Pair-biased and sliding-window attention primitives: FlexAttention plus a dense oracle.

Semantics ported from Biohub's ESMFold2 ``AttentionPairBias`` (additive pair bias,
key padding mask) and ``SWA3DRoPEAttention`` (window measured in rank among valid
atoms, self always visible, invalid outputs zeroed) in
``esm/models/esmfold2/layers.py`` (Apache-2.0; see THIRD_PARTY_NOTICES.md).
Modifications: FlexAttention with block masks replaces dense-bias SDPA and the
FlashAttention dependency; the dense formulation is kept as the small-input oracle
and the CPU path (FlexAttention has no CPU backward); a batch row with no valid key
yields zeros rather than NaN. Q/K/V are never cast: the caller owns the precision
policy (design §5.3, §5.4).

Both functions take ``(B, H, N, d)`` tensors. Projections, gates, RoPE and
adaptive norms belong to the modules that call these (milestone 1).
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Literal

import torch
from torch.nn import functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor
    from torch.nn.attention.flex_attention import BlockMask

__all__ = [
    "AttentionBackend",
    "pair_biased_attention",
    "resolve_attention_backend",
    "sliding_window_attention",
]

AttentionBackend = Literal["auto", "dense", "flex"]


def resolve_attention_backend(x: Tensor, backend: AttentionBackend) -> Literal["dense", "flex"]:
    """``"auto"`` is flex on CUDA and dense elsewhere; explicit choices pass through."""
    if backend == "auto":
        return "flex" if x.is_cuda else "dense"
    return backend


@functools.cache
def _compiled_flex() -> Callable[..., Tensor]:
    # Uncompiled flex_attention materializes the full score matrix (it warns about it);
    # the fused kernel only exists through torch.compile. Compiled once per process.
    return torch.compile(  # ty: ignore[invalid-return-type]  # typed Tensor | tuple; no return_lse
        flex_attention, dynamic=False
    )


def _flex(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    score_mod: Callable[..., Tensor] | None,
    block_mask: BlockMask | None,
) -> Tensor:
    fn = _compiled_flex() if q.is_cuda else flex_attention  # CPU: eager, forward-only use
    return fn(  # ty: ignore[invalid-return-type]  # typed Tensor | tuple; return_lse never requested
        q, k, v, score_mod=score_mod, block_mask=block_mask
    )


def pair_biased_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    bias: Tensor,
    key_mask: Tensor | None = None,
    *,
    backend: AttentionBackend = "auto",
) -> Tensor:
    """``softmax_j(q_i.k_j / sqrt(d) + bias[b,h,i,j]) v_j`` over valid keys.

    Args:
        q: Queries ``(B, H, N, d)``.
        k: Keys ``(B, H, N, d)``.
        v: Values ``(B, H, N, d)``.
        bias: Additive pair bias ``(B, H, N, N)`` (any float dtype; read inside the kernel).
        key_mask: ``(B, N)`` bool, True for valid keys; ``None`` means all valid. A batch
            row with no valid key returns zeros.
        backend: ``"auto"``, ``"dense"`` or ``"flex"``.

    Returns:
        ``(B, H, N, d)`` in ``v``'s dtype.
    """
    if resolve_attention_backend(q, backend) == "dense":
        scale = q.shape[-1] ** -0.5
        logits = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale + bias.float()
        if key_mask is not None:
            logits = logits.masked_fill(~key_mask[:, None, None, :], torch.finfo(logits.dtype).min)
        attn = torch.softmax(logits, dim=-1)
        if key_mask is not None:  # all -inf rows softmax to NaN: define them as zero
            any_valid = key_mask.any(dim=-1)[:, None, None, None]
            attn = torch.where(any_valid, attn, torch.zeros_like(attn))
        return torch.matmul(attn.to(v.dtype), v)

    batch, _heads, n_q, _ = q.shape

    def score_mod(score: Tensor, b: Tensor, h: Tensor, qi: Tensor, ki: Tensor) -> Tensor:
        return score + bias[b, h, qi, ki]

    block_mask: BlockMask | None = None
    if key_mask is not None:

        def mask_mod(b: Tensor, h: Tensor, qi: Tensor, ki: Tensor) -> Tensor:
            return key_mask[b, ki]

        block_mask = create_block_mask(mask_mod, batch, None, n_q, k.shape[2], device=q.device)
    return _flex(q, k, v, score_mod, block_mask)


def sliding_window_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    valid: Tensor,
    half_window: int,
    *,
    backend: AttentionBackend = "auto",
) -> Tensor:
    """Local attention over the valid-atom rank with window ``[-half_window, half_window]``.

    ``allowed[b,i,j] = (valid[b,i] & valid[b,j] & |rank_i - rank_j| <= half_window) | (i == j)``
    with ``rank = cumsum(valid) - 1``; outputs at invalid positions are zero. This is
    ESMFold2's SDPA formulation and what its FlashAttention varlen path computes once
    padding is removed, so the window is measured in reference space, not padded index
    space. Self-attention is always allowed, so a padded query stays finite.

    Args:
        q: Queries ``(B, H, N, d)``.
        k: Keys ``(B, H, N, d)``.
        v: Values ``(B, H, N, d)``.
        valid: ``(B, N)`` bool atom validity.
        half_window: Window radius in valid-atom rank (upstream default 64 -> window 128).
        backend: ``"auto"``, ``"dense"`` or ``"flex"``.
    """
    batch, _heads, n, _ = q.shape
    rank = torch.cumsum(valid.to(torch.int64), dim=1) - 1
    if resolve_attention_backend(q, backend) == "dense":
        within = (rank[:, :, None] - rank[:, None, :]).abs() <= half_window
        allowed = within & valid[:, :, None] & valid[:, None, :]
        allowed = allowed | torch.eye(n, dtype=torch.bool, device=q.device)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed[:, None])
    else:

        def mask_mod(b: Tensor, h: Tensor, qi: Tensor, ki: Tensor) -> Tensor:
            in_window = (rank[b, qi] - rank[b, ki]).abs() <= half_window
            return (valid[b, qi] & valid[b, ki] & in_window) | (qi == ki)

        block_mask = create_block_mask(mask_mod, batch, None, n, n, device=q.device)
        out = _flex(q, k, v, None, block_mask)
    return out * valid[:, None, :, None].to(out.dtype)
