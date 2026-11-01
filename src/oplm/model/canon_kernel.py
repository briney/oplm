"""Triton kernels for the Canon convolution (same function as `conv.canon_conv_reference`).

The forward computes, per `(b, t, d)`, `sum_i w[d, i] * x[b, t + i - left, d]` over taps whose
source is inside the row, not a pad, and (packed rows) in the output's segment; fp32 accumulation,
cast to `out_dtype`, then optionally `x + conv` with the raw `x`. Kernels run on CUDA tensors, or
on CPU tensors under `TRITON_INTERPRET=1`.
"""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - triton ships with torch on Linux
    TRITON_AVAILABLE = False

__all__ = ["BLOCK_T", "TRITON_AVAILABLE", "canon_forward"]

BLOCK_T = 64  # tokens per program
BLOCK_D = 64  # channels per program
# ponytail: fixed tiles and num_warps, autotune if the B200 benchmark (Task 6) says it pays.


if TRITON_AVAILABLE:
    _TL_DTYPES = {torch.float32: tl.float32, torch.bfloat16: tl.bfloat16, torch.float16: tl.float16}

    @triton.jit
    def _round_bf16(v):
        # fp32 -> fp32 holding v rounded to bf16 (nearest-even). Only used under TRITON_INTERPRET:
        # the interpreter truncates on `.to(tl.bfloat16)`; GPU casts already round to nearest-even.
        bits = v.to(tl.uint32, bitcast=True)
        bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
        return bits.to(tl.float32, bitcast=True)

    @triton.jit
    def _canon_fwd_kernel(
        x_ptr,
        w_ptr,
        mask_ptr,
        seg_ptr,
        out_ptr,
        T,
        D,
        stride_wd,
        stride_wk,
        stride_mb,
        stride_mt,
        stride_sb,
        stride_st,
        K: tl.constexpr,
        LEFT: tl.constexpr,
        HAS_SEG: tl.constexpr,
        RESIDUAL: tl.constexpr,
        CONV_DTYPE: tl.constexpr,
        ROUND_CONV: tl.constexpr,
        ROUND_OUT: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        # One (BLOCK_T, BLOCK_D) output tile of row `b`. The k shifted reads of x cover the tile
        # plus a (k - 1)-token halo; the overlap is served from L1/L2, so DRAM sees x about once.
        b = tl.program_id(0).to(tl.int64)
        t = tl.program_id(1) * BLOCK_T + tl.arange(0, BLOCK_T)
        d = tl.program_id(2) * BLOCK_D + tl.arange(0, BLOCK_D)
        t_ok = t < T
        d_ok = d < D
        row = b * T * D  # int64 row base; within-row offsets fit int32
        mask_row = mask_ptr + b * stride_mb
        seg_row = seg_ptr + b * stride_sb
        if HAS_SEG:
            seg_t = tl.load(seg_row + t * stride_st, mask=t_ok, other=-1)

        acc = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)
        for i in tl.static_range(K):
            src = t + (i - LEFT)
            ok = (src >= 0) & (src < T)
            ok = ok & (tl.load(mask_row + src * stride_mt, mask=ok, other=0) != 0)
            if HAS_SEG:
                ok = ok & (tl.load(seg_row + src * stride_st, mask=ok, other=-1) == seg_t)
            xv = tl.load(
                x_ptr + row + src[:, None] * D + d[None, :],
                mask=ok[:, None] & d_ok[None, :],
                other=0.0,
            )
            wv = tl.load(w_ptr + d * stride_wd + i * stride_wk, mask=d_ok, other=0.0)
            acc += xv.to(tl.float32) * wv.to(tl.float32)[None, :]

        if ROUND_CONV:
            acc = _round_bf16(acc)
        conv = acc.to(CONV_DTYPE)
        io_ok = t_ok[:, None] & d_ok[None, :]
        offs = row + t[:, None] * D + d[None, :]
        if RESIDUAL:
            # `x + conv` as torch computes it: conv rounded to out_dtype first, add in fp32.
            xr = tl.load(x_ptr + offs, mask=io_ok, other=0.0)
            res = xr.to(tl.float32) + conv.to(tl.float32)
            if ROUND_OUT:
                res = _round_bf16(res)
            tl.store(out_ptr + offs, res.to(out_ptr.dtype.element_ty), mask=io_ok)
        else:
            tl.store(out_ptr + offs, conv, mask=io_ok)


def canon_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    attention_mask: torch.Tensor,
    segment_ids: torch.Tensor | None,
    out_dtype: torch.dtype,
    residual: bool,
) -> torch.Tensor:
    """Triton Canon conv; same contract as `conv.canon_conv_reference`.

    `x` is `(B, T, D)`, `weight` `(D, k)`, `attention_mask` `(B, T)` bool or integer,
    `segment_ids` `(B, T)` integer or None. Returns the conv in `out_dtype`, or `x + conv` in
    `torch.promote_types(x.dtype, out_dtype)` when `residual`.
    """
    x = x.contiguous()
    B, T, D = x.shape
    k = weight.shape[1]
    out = torch.empty_like(
        x, dtype=torch.promote_types(x.dtype, out_dtype) if residual else out_dtype
    )
    interpret = os.environ.get("TRITON_INTERPRET") == "1"
    if out.numel() == 0:
        return out
    seg = segment_ids if segment_ids is not None else attention_mask  # unread when None
    grid = (B, triton.cdiv(T, BLOCK_T), triton.cdiv(D, BLOCK_D))
    _canon_fwd_kernel[grid](
        x,
        weight,
        attention_mask,
        seg,
        out,
        T,
        D,
        weight.stride(0),
        weight.stride(1),
        attention_mask.stride(0),
        attention_mask.stride(1),
        seg.stride(0),
        seg.stride(1),
        K=k,
        LEFT=k // 2,
        HAS_SEG=segment_ids is not None,
        RESIDUAL=residual,
        CONV_DTYPE=_TL_DTYPES[out_dtype],
        ROUND_CONV=interpret and out_dtype == torch.bfloat16,
        ROUND_OUT=interpret and out.dtype == torch.bfloat16,
        BLOCK_T=BLOCK_T,
        BLOCK_D=BLOCK_D,
    )
    return out
