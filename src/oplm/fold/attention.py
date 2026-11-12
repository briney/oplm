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
from torch._dynamo.exc import FailOnRecompileLimitHit
from torch.nn import functional as F
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

__all__ = [
    "AttentionBackend",
    "ensure_flex_recompile_limit",
    "pair_bias_block_mask",
    "pair_biased_attention",
    "resolve_attention_backend",
    "sliding_window_attention",
    "sliding_window_block_mask",
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
    if not q.is_cuda:  # CPU: eager, forward-only use
        return flex_attention(  # ty: ignore[invalid-return-type]  # return_lse never requested
            q, k, v, score_mod=score_mod, block_mask=block_mask
        )
    if torch.compiler.is_compiling():  # traced into an outer compiled graph: no cache to manage
        return _compiled_flex()(q, k, v, score_mod=score_mod, block_mask=block_mask)
    # Each new shape is one more static compile (dynamic shapes cost 2-5x in kernel time). Past
    # dynamo's recompile limit flex would silently run eagerly and materialize full score
    # matrices (#107), so hitting the limit raises here instead and starts a fresh compile cache.
    with torch._dynamo.config.patch(fail_on_recompile_limit_hit=True):
        try:
            return _compiled_flex()(q, k, v, score_mod=score_mod, block_mask=block_mask)
        except FailOnRecompileLimitHit:
            torch._dynamo.reset()  # also drops other compiled code; it recompiles on next use
            return _compiled_flex()(q, k, v, score_mod=score_mod, block_mask=block_mask)


def pair_bias_block_mask(key_mask: Tensor, n_queries: int | None = None) -> BlockMask:
    """Block mask for `pair_biased_attention`: keys valid where `key_mask[b, j]`.

    Build it once per forward (one per diffusion-sampling call) and pass it to every
    block; `create_block_mask` is uncompiled and materialises a `(B, N, N)` mask.
    """
    batch, n_keys = key_mask.shape

    def mask_mod(b: Tensor, h: Tensor, qi: Tensor, ki: Tensor) -> Tensor:
        return key_mask[b, ki]

    return create_block_mask(
        mask_mod, batch, None, n_queries or n_keys, n_keys, device=key_mask.device
    )


def sliding_window_block_mask(valid: Tensor, half_window: int, block: int = 128) -> BlockMask:
    """Block mask for `sliding_window_attention` (rank-based window, self always visible).

    Built from per-block key ranges, never an ``(N, N)`` mask: ``create_block_mask`` would
    materialize ``B·N²`` elements, 14 GiB per row at ~43k atoms (docs/FOLD.md §12). ``rank`` is
    non-decreasing, so a query's allowed keys lie in the index band whose rank is within
    ``half_window`` of its own; each query block lists the key blocks that band touches (plus
    its own, for padded queries) as partial blocks, and ``mask_mod`` decides inside them.
    """
    batch, n = valid.shape
    rank = torch.cumsum(valid.to(torch.int64), dim=1) - 1

    def mask_mod(b: Tensor, h: Tensor, qi: Tensor, ki: Tensor) -> Tensor:
        in_window = (rank[b, qi] - rank[b, ki]).abs() <= half_window
        return (valid[b, qi] & valid[b, ki] & in_window) | (qi == ki)

    n_blocks = -(-n // block)
    lo = torch.searchsorted(rank, rank - half_window, side="left")  # first key in the band
    hi = torch.searchsorted(rank, rank + half_window, side="right") - 1  # last key in the band
    pad = n_blocks * block - n  # ragged last block: its rows repeat the last query's band
    lo = F.pad(lo, (0, pad), mode="replicate") if pad else lo
    hi = F.pad(hi, (0, pad), mode="replicate") if pad else hi
    q_block = torch.arange(n_blocks, device=valid.device)
    first = torch.minimum(lo.view(batch, n_blocks, block).amin(-1) // block, q_block)
    last = torch.maximum(hi.view(batch, n_blocks, block).amax(-1) // block, q_block)
    count = last - first + 1
    # Full-width rows as create_block_mask makes them: BlockMask reads the index width as the
    # key-block count (q-side transpose, to_dense). Entries past ``count`` are never read.
    indices = first[..., None] + torch.arange(n_blocks, device=valid.device)
    indices = indices.clamp(max=n_blocks - 1).to(torch.int32)
    return BlockMask.from_kv_blocks(
        count[:, None].to(torch.int32),
        indices[:, None],
        BLOCK_SIZE=block,
        mask_mod=mask_mod,
        seq_lengths=(n, n),
    )


def ensure_flex_recompile_limit(limit: int = 64) -> None:
    """Raise dynamo's recompile limit: how many flex shapes compile before a cache reset.

    `_compiled_flex` uses `dynamic=False`; each new `(B, N, dtype)` compiles once. When the
    limit is hit, `_flex` resets dynamo and compiles afresh instead of letting flex fall back
    to the eager kernel, which materialises the full score matrix (docs/FOLD.md §3, #107).
    Only ever raises the limit.
    """
    from torch import _dynamo

    _dynamo.config.recompile_limit = max(  # ty: ignore[invalid-assignment]  # inferred Literal[8]
        _dynamo.config.recompile_limit, limit
    )


def pair_biased_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    bias: Tensor,
    key_mask: Tensor | None = None,
    *,
    backend: AttentionBackend = "auto",
    block_mask: BlockMask | None = None,
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
        block_mask: A precomputed mask from ``pair_bias_block_mask`` /
            ``sliding_window_block_mask``; built inline when ``None``; ignored on the
            dense path.

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

    n_q = q.shape[2]

    def score_mod(score: Tensor, b: Tensor, h: Tensor, qi: Tensor, ki: Tensor) -> Tensor:
        return score + bias[b, h, qi, ki]

    if block_mask is None and key_mask is not None:
        block_mask = pair_bias_block_mask(key_mask, n_q)
    return _flex(q, k, v, score_mod, block_mask)


def sliding_window_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    valid: Tensor,
    half_window: int,
    *,
    backend: AttentionBackend = "auto",
    block_mask: BlockMask | None = None,
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
        block_mask: A precomputed mask from ``pair_bias_block_mask`` /
            ``sliding_window_block_mask``; built inline when ``None``; ignored on the
            dense path.
    """
    if resolve_attention_backend(q, backend) == "dense":
        n = q.shape[2]
        rank = torch.cumsum(valid.to(torch.int64), dim=1) - 1
        within = (rank[:, :, None] - rank[:, None, :]).abs() <= half_window
        allowed = within & valid[:, :, None] & valid[:, None, :]
        allowed = allowed | torch.eye(n, dtype=torch.bool, device=q.device)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed[:, None])
    else:
        if block_mask is None:
            block_mask = sliding_window_block_mask(valid, half_window)
        out = _flex(q, k, v, None, block_mask)
    return out * valid[:, None, :, None].to(out.dtype)
