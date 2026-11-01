"""Triton kernels for the Canon convolution (same function as `conv.canon_conv_reference`).

The forward computes, per `(b, t, d)`, `sum_i w[d, i] * x[b, t + i - left, d]` over taps whose
source is inside the row, not a pad, and (packed rows) in the output's segment; fp32 accumulation,
cast to `out_dtype`, then optionally `x + conv` with the raw `x`. The backward is one pass giving
`dx` (the transposed conv, plus `dy` for the residual) and per-program fp32 `dw` partials, summed
after. Kernels run on CUDA tensors, or on CPU tensors under `TRITON_INTERPRET=1`.
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

__all__ = ["BLOCK_T", "BWD_TILES", "TRITON_AVAILABLE", "canon_backward", "canon_forward"]

BLOCK_T = 64  # tokens per program
BLOCK_D = 64  # channels per program
BWD_TILES = 4  # backward: token tiles per program; divides the dw partials buffer by this
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

    @triton.jit
    def _canon_bwd_kernel(
        dy_ptr,
        x_ptr,
        w_ptr,
        mask_ptr,
        seg_ptr,
        dx_ptr,
        dw_part_ptr,
        T,
        D,
        stride_wd,
        stride_wk,
        stride_mb,
        stride_mt,
        stride_sb,
        stride_st,
        K: tl.constexpr,
        K_POW2: tl.constexpr,
        LEFT: tl.constexpr,
        HAS_SEG: tl.constexpr,
        RESIDUAL: tl.constexpr,
        CONV_DTYPE: tl.constexpr,
        ROUND_G: tl.constexpr,
        ROUND_DX: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_D: tl.constexpr,
        TILES: tl.constexpr,
    ):
        # TILES consecutive (BLOCK_T, BLOCK_D) tiles of row `b`, indexed by the input position `s`.
        # Tap i of output t = s - i + LEFT reads s, so per tap, with the forward's keep test on
        # the pair (t inside the row, s not a pad, same segment):
        #   dx[s] += w[i] * dy[t]   and   dw[i] += x[s] * dy[t]   (summed over the tiles' s).
        # x and dx are touched once; dy's k shifted reads are cached like the forward's x.
        b = tl.program_id(0).to(tl.int64)
        pid_t = tl.program_id(1)
        d = tl.program_id(2) * BLOCK_D + tl.arange(0, BLOCK_D)
        d_ok = d < D
        row = b * T * D  # int64 row base; within-row offsets fit int32
        mask_row = mask_ptr + b * stride_mb
        seg_row = seg_ptr + b * stride_sb
        taps = tl.arange(0, K_POW2)
        dw_acc = tl.zeros((BLOCK_D, K_POW2), dtype=tl.float32)  # column i: tap i's partial

        for j in range(TILES):
            s = (pid_t * TILES + j) * BLOCK_T + tl.arange(0, BLOCK_T)
            s_ok = s < T
            # A pad source gets no conv gradient (x is zeroed before the conv) and feeds no dw.
            real = s_ok & (tl.load(mask_row + s * stride_mt, mask=s_ok, other=0) != 0)
            if HAS_SEG:
                seg_s = tl.load(seg_row + s * stride_st, mask=s_ok, other=-1)
            offs = row + s[:, None] * D + d[None, :]
            xv = tl.load(x_ptr + offs, mask=real[:, None] & d_ok[None, :], other=0.0)
            xv = xv.to(tl.float32)

            acc = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)
            for i in tl.static_range(K):
                t = s + (LEFT - i)
                ok = real & (t >= 0) & (t < T)
                if HAS_SEG:
                    ok = ok & (tl.load(seg_row + t * stride_st, mask=ok, other=-1) == seg_s)
                g = tl.load(
                    dy_ptr + row + t[:, None] * D + d[None, :],
                    mask=ok[:, None] & d_ok[None, :],
                    other=0.0,
                ).to(tl.float32)
                # The conv's gradient is dy in the conv's dtype (autograd casts it through the
                # forward's `.to(out_dtype)`); the residual path below keeps the raw dy.
                if ROUND_G:
                    g = _round_bf16(g)
                g = g.to(CONV_DTYPE).to(tl.float32)
                wv = tl.load(w_ptr + d * stride_wd + i * stride_wk, mask=d_ok, other=0.0)
                acc += g * wv.to(tl.float32)[None, :]
                dw_acc += tl.where(taps[None, :] == i, tl.sum(xv * g, axis=0)[:, None], 0.0)

            io_ok = s_ok[:, None] & d_ok[None, :]
            if ROUND_DX:
                acc = _round_bf16(acc)
            dx = acc.to(dx_ptr.dtype.element_ty)
            if RESIDUAL:
                # As autograd sums the two paths into x.grad: each rounded to x's dtype, then added.
                r = tl.load(dy_ptr + offs, mask=io_ok, other=0.0).to(tl.float32)
                if ROUND_DX:
                    r = _round_bf16(r)
                res = dx.to(tl.float32) + r.to(dx_ptr.dtype.element_ty).to(tl.float32)
                if ROUND_DX:
                    res = _round_bf16(res)
                dx = res.to(dx_ptr.dtype.element_ty)
            tl.store(dx_ptr + offs, dx, mask=io_ok)

        part = (b * tl.num_programs(1) + pid_t) * D * K
        tl.store(
            dw_part_ptr + part + d[:, None] * K + taps[None, :],
            dw_acc,
            mask=d_ok[:, None] & (taps[None, :] < K),
        )


def _check_row_size(T: int, D: int) -> None:
    if T * D >= 2**31:
        raise ValueError(f"Canon kernel needs T * D < 2**31 (int32 in-row offsets); got {T} * {D}")


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
    B, T, D = x.shape
    _check_row_size(T, D)
    x = x.contiguous()
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


def canon_backward(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    attention_mask: torch.Tensor,
    segment_ids: torch.Tensor | None,
    out_dtype: torch.dtype,
    residual: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gradients of `canon_forward` for its output gradient `dy`: `(dx, dw)`.

    `out_dtype` is the forward's conv dtype: the conv path sees `dy` rounded to it, as autograd's
    cast backward does; the residual path sees the raw `dy`. `dx` is `(B, T, D)` in `x.dtype`;
    `dw` is `(D, k)` in `weight.dtype`, reduced deterministically (per-program fp32 partials,
    then a sum).
    """
    B, T, D = x.shape
    _check_row_size(T, D)
    x = x.contiguous()
    dy = dy.contiguous()
    k = weight.shape[1]
    dx = torch.empty_like(x)
    interpret = os.environ.get("TRITON_INTERPRET") == "1"
    if x.numel() == 0:
        return dx, torch.zeros_like(weight)
    seg = segment_ids if segment_ids is not None else attention_mask  # unread when None
    grid = (B, triton.cdiv(T, BLOCK_T * BWD_TILES), triton.cdiv(D, BLOCK_D))
    partials = torch.empty(B * grid[1], D, k, device=x.device, dtype=torch.float32)
    _canon_bwd_kernel[grid](
        dy,
        x,
        weight,
        attention_mask,
        seg,
        dx,
        partials,
        T,
        D,
        weight.stride(0),
        weight.stride(1),
        attention_mask.stride(0),
        attention_mask.stride(1),
        seg.stride(0),
        seg.stride(1),
        K=k,
        K_POW2=triton.next_power_of_2(k),
        LEFT=k // 2,
        HAS_SEG=segment_ids is not None,
        RESIDUAL=residual,
        CONV_DTYPE=_TL_DTYPES[out_dtype],
        ROUND_G=interpret and out_dtype == torch.bfloat16,
        ROUND_DX=interpret and x.dtype == torch.bfloat16,
        BLOCK_T=BLOCK_T,
        BLOCK_D=BLOCK_D,
        TILES=BWD_TILES,
    )
    return dx, partials.sum(0).to(weight.dtype)
