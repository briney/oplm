"""Triton kernels for the Canon convolution (same function as `conv.canon_conv_reference`).

The forward computes, per `(b, t, d)`, `sum_i w[d, i] * x[b, t + i - left, d]` over taps whose
source is inside the row, not a pad, and (packed rows) in the output's segment; fp32 accumulation,
cast to `out_dtype`, then optionally `x + conv` with the raw `x`. The backward is one pass giving
`dx` (the transposed conv, plus `dy` for the residual) and per-program fp32 `dw` partials, summed
after. Kernels run on CUDA tensors, or on CPU tensors under `TRITON_INTERPRET=1`. Launch shapes
come from `FWD_CONFIGS` / `BWD_CONFIGS` (picked by `canon_kernel_bench.py --sweep`).

Both kernels walk a strip of tokens in order, keeping the k rows the current position needs in
registers, so each row is read from DRAM once per program. Tile kernels that issue the k shifted
reads together miss L1 together (each read goes to L2), and their wide 2D/3D tiles spilled.

`torch.ops.oplm.canon_conv` wraps them as one custom op with its own autograd, opaque to
`torch.compile` (no graph break, no tracing into Triton): `canon_conv_kernel` is its entry point.
`canon_self_check` compares it with the reference on a model's real Canon inputs (run start).
"""

from __future__ import annotations

import contextlib
import inspect
import os
from typing import TYPE_CHECKING, NamedTuple

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import nn

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - triton ships with torch on Linux
    TRITON_AVAILABLE = False

__all__ = [
    "BWD_CONFIGS",
    "FWD_CONFIGS",
    "TRITON_AVAILABLE",
    "BwdConfig",
    "CanonKernelMismatch",
    "FwdConfig",
    "canon_backward",
    "canon_conv_kernel",
    "canon_forward",
    "canon_self_check",
]


class FwdConfig(NamedTuple):
    """Forward launch: each program walks `strip` tokens of `block_d` channels, `unroll` tokens
    per loop step (`unroll` divides `strip`)."""

    block_d: int = 128
    strip: int = 128
    unroll: int = 8
    num_warps: int = 1
    num_stages: int = 1


class BwdConfig(NamedTuple):
    """Backward launch: as `FwdConfig`; `strip` is also the tokens per dw partial."""

    block_d: int = 128
    strip: int = 256
    unroll: int = 4
    num_warps: int = 1
    num_stages: int = 1


# Per `(D, x dtype, conv dtype)`; other shapes take the defaults, which compile without spills on
# sm_100 at k=7 (tests/model/test_canon_kernel.py checks) and await the B200 sweep. A fixed table,
# not `triton.autotune`: autotune picks by timing at the first call (in the self-check, on every
# rank), so ranks and reruns could pick different strips, and the strip sets dw's summation
# order: training would stop being reproducible run to run.
FWD_CONFIGS: dict[tuple[int, torch.dtype, torch.dtype], FwdConfig] = {}
BWD_CONFIGS: dict[tuple[int, torch.dtype, torch.dtype], BwdConfig] = {}


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
    def _load_row(
        ptr, row, pos, T, D, d, d_ok, mask_row, stride_mt, seg_row, stride_st, HAS_SEG: tl.constexpr
    ):
        # Row `pos` of a row-major (T, D) slab at `row`: its raw values at `d` in fp32 (0 outside
        # the row), whether it is inside the row, whether it is also not a pad, and its segment id
        # (-1 outside the row; 0 when unpacked).
        in_row = (pos >= 0) & (pos < T)
        v = tl.load(ptr + row + pos * D + d, mask=in_row & d_ok, other=0.0).to(tl.float32)
        real = in_row & (tl.load(mask_row + pos * stride_mt, mask=in_row, other=0) != 0)
        if HAS_SEG:
            seg = tl.load(seg_row + pos * stride_st, mask=in_row, other=-1).to(tl.int32)
        else:
            seg = in_row.to(tl.int32) - 1
        return v, in_row, real, seg

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
        BLOCK_D: tl.constexpr,
        STRIP: tl.constexpr,
        UNROLL: tl.constexpr,
    ):
        # Channels `d` of row `b`, outputs t0 <= t < t0 + STRIP in order. `xs` holds the k source
        # rows t - LEFT .. t + RIGHT of output t (oldest first), with each row's real flag and
        # segment id: one new row is read per output, the rest shift along.
        b = tl.program_id(0).to(tl.int64)
        t0 = tl.program_id(1) * STRIP
        d = tl.program_id(2) * BLOCK_D + tl.arange(0, BLOCK_D)
        d_ok = d < D
        row = b * T * D  # int64 row base; within-row offsets fit int32
        mask_row = mask_ptr + b * stride_mb
        seg_row = seg_ptr + b * stride_sb
        RIGHT: tl.constexpr = K - 1 - LEFT

        w = ()
        for i in tl.static_range(K):
            wv = tl.load(w_ptr + d * stride_wd + i * stride_wk, mask=d_ok, other=0.0)
            w = w + (wv.to(tl.float32),)
        xs, reals, segs = (), (), ()
        for i in tl.static_range(K - 1):
            v, _, real, seg = _load_row(
                x_ptr, row, t0 - LEFT + i, T, D, d, d_ok, mask_row, stride_mt, seg_row, stride_st,
                HAS_SEG,
            )  # fmt: skip
            xs, reals, segs = xs + (v,), reals + (real,), segs + (seg,)

        for step in range(STRIP // UNROLL):  # positions past T are masked
            for u in tl.static_range(UNROLL):
                t = t0 + step * UNROLL + u
                v, _, real, seg = _load_row(
                    x_ptr, row, t + RIGHT, T, D, d, d_ok, mask_row, stride_mt, seg_row, stride_st,
                    HAS_SEG,
                )  # fmt: skip
                xs, reals, segs = xs + (v,), reals + (real,), segs + (seg,)
                acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
                for i in tl.static_range(K):
                    keep = reals[i]
                    if HAS_SEG:
                        keep = keep & (segs[i] == segs[LEFT])
                    acc += tl.where(keep, xs[i], 0.0) * w[i]

                if ROUND_CONV:
                    acc = _round_bf16(acc)
                conv = acc.to(CONV_DTYPE)
                io_ok = d_ok & (t < T)
                offs = row + t * D + d
                if RESIDUAL:
                    # `x + conv` as torch computes it: conv rounded to out_dtype first, add in fp32.
                    res = xs[LEFT] + conv.to(tl.float32)
                    if ROUND_OUT:
                        res = _round_bf16(res)
                    tl.store(out_ptr + offs, res.to(out_ptr.dtype.element_ty), mask=io_ok)
                else:
                    tl.store(out_ptr + offs, conv, mask=io_ok)
                xs, reals, segs = xs[1:], reals[1:], segs[1:]

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
        LEFT: tl.constexpr,
        HAS_SEG: tl.constexpr,
        RESIDUAL: tl.constexpr,
        CONV_DTYPE: tl.constexpr,
        ROUND_G: tl.constexpr,
        ROUND_DX: tl.constexpr,
        BLOCK_D: tl.constexpr,
        STRIP: tl.constexpr,
        UNROLL: tl.constexpr,
    ):
        # Channels `d` of row `b`, input positions s0 <= s < s0 + STRIP in order. Tap i of output
        # t = s - i + LEFT reads s, so per tap, with the forward's keep test on the pair (t inside
        # the row, s real, same segment):
        #   dx[s] = sum_i w[i] * dy[t]   and   dw[i] += x[s] * dy[t].
        # `gs` holds dy's rows t = s - RIGHT .. s + LEFT (oldest first: tap i is gs[K - 1 - i])
        # with each row's in-row flag and segment id; dw's k sums stay in registers.
        b = tl.program_id(0).to(tl.int64)
        pid_t = tl.program_id(1)
        s0 = pid_t * STRIP
        d = tl.program_id(2) * BLOCK_D + tl.arange(0, BLOCK_D)
        d_ok = d < D
        row = b * T * D  # int64 row base; within-row offsets fit int32
        mask_row = mask_ptr + b * stride_mb
        seg_row = seg_ptr + b * stride_sb
        RIGHT: tl.constexpr = K - 1 - LEFT

        w, dw = (), ()
        for i in tl.static_range(K):
            wv = tl.load(w_ptr + d * stride_wd + i * stride_wk, mask=d_ok, other=0.0)
            w, dw = w + (wv.to(tl.float32),), dw + (tl.zeros((BLOCK_D,), dtype=tl.float32),)
        gs, in_rows, segs = (), (), ()
        for j in tl.static_range(K - 1):
            g, in_row, _, seg = _load_row(
                dy_ptr, row, s0 - RIGHT + j, T, D, d, d_ok, mask_row, stride_mt, seg_row,
                stride_st, HAS_SEG,
            )  # fmt: skip
            gs, in_rows, segs = gs + (g,), in_rows + (in_row,), segs + (seg,)

        for step in range(STRIP // UNROLL):  # positions past T are masked
            for u in tl.static_range(UNROLL):
                s = s0 + step * UNROLL + u
                g, in_row, _, seg = _load_row(
                    dy_ptr, row, s + LEFT, T, D, d, d_ok, mask_row, stride_mt, seg_row,
                    stride_st, HAS_SEG,
                )  # fmt: skip
                gs, in_rows, segs = gs + (g,), in_rows + (in_row,), segs + (seg,)
                xv, _, real_s, seg_s = _load_row(
                    x_ptr, row, s, T, D, d, d_ok, mask_row, stride_mt, seg_row, stride_st,
                    HAS_SEG,
                )  # fmt: skip
                # A pad source gets no conv gradient (x is zeroed before the conv) and feeds no dw.
                xv = tl.where(real_s, xv, 0.0)
                acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
                new_dw = ()
                for i in tl.static_range(K):
                    ok = real_s & in_rows[K - 1 - i]
                    if HAS_SEG:
                        ok = ok & (segs[K - 1 - i] == seg_s)
                    # The conv's gradient is dy in the conv's dtype (autograd casts it through the
                    # forward's `.to(out_dtype)`); the residual path below keeps the raw dy.
                    gi = gs[K - 1 - i]
                    if ROUND_G:
                        gi = _round_bf16(gi)
                    gi = tl.where(ok, gi.to(CONV_DTYPE).to(tl.float32), 0.0)
                    acc += gi * w[i]
                    new_dw = new_dw + (dw[i] + gi * xv,)
                dw = new_dw

                io_ok = d_ok & (s < T)
                if ROUND_DX:
                    acc = _round_bf16(acc)
                dx = acc.to(dx_ptr.dtype.element_ty)
                if RESIDUAL:
                    # As autograd sums the two paths into x.grad: each rounded to x's dtype, then
                    # added. gs[RIGHT] is dy[s], raw.
                    r = gs[RIGHT]
                    if ROUND_DX:
                        r = _round_bf16(r)
                    res = dx.to(tl.float32) + r.to(dx_ptr.dtype.element_ty).to(tl.float32)
                    if ROUND_DX:
                        res = _round_bf16(res)
                    dx = res.to(dx_ptr.dtype.element_ty)
                tl.store(dx_ptr + row + s * D + d, dx, mask=io_ok)
                gs, in_rows, segs = gs[1:], in_rows[1:], segs[1:]

        part = (b * tl.num_programs(1) + pid_t) * D * K
        for i in tl.static_range(K):
            tl.store(dw_part_ptr + part + d * K + i, dw[i], mask=d_ok)


def _on_device(x: torch.Tensor) -> contextlib.AbstractContextManager:
    """Launch on `x`'s GPU (Triton uses the current device); a no-op for interpreter CPU tensors."""
    return torch.cuda.device(x.device) if x.is_cuda else contextlib.nullcontext()


def _check_row_size(T: int, D: int) -> None:
    if T * D >= 2**31:
        raise ValueError(f"Canon kernel needs T * D < 2**31 (int32 in-row offsets); got {T} * {D}")


def _check_strip(cfg: FwdConfig | BwdConfig) -> None:
    # A loop step past the strip's end would redo the next strip's tokens (and their dw twice).
    if cfg.strip % cfg.unroll:
        raise ValueError(f"Canon kernel config needs unroll to divide strip; got {cfg}")


def canon_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    attention_mask: torch.Tensor,
    segment_ids: torch.Tensor | None,
    out_dtype: torch.dtype,
    residual: bool,
    *,
    config: FwdConfig | None = None,
) -> torch.Tensor:
    """Triton Canon conv; same contract as `conv.canon_conv_reference`.

    `x` is `(B, T, D)`, `weight` `(D, k)`, `attention_mask` `(B, T)` bool or integer,
    `segment_ids` `(B, T)` integer or None. Returns the conv in `out_dtype`, or `x + conv` in
    `torch.promote_types(x.dtype, out_dtype)` when `residual`. `config` overrides the
    `FWD_CONFIGS` launch (benchmark sweeps).
    """
    B, T, D = x.shape
    _check_row_size(T, D)
    x = x.contiguous()
    k = weight.shape[1]
    cfg = config or FWD_CONFIGS.get((D, x.dtype, out_dtype), FwdConfig())
    _check_strip(cfg)
    out = torch.empty_like(
        x, dtype=torch.promote_types(x.dtype, out_dtype) if residual else out_dtype
    )
    interpret = os.environ.get("TRITON_INTERPRET") == "1"
    if out.numel() == 0:
        return out
    seg = segment_ids if segment_ids is not None else attention_mask  # unread when None
    grid = (B, triton.cdiv(T, cfg.strip), triton.cdiv(D, cfg.block_d))
    # ty ignores: Triton boundary, plain values for tl.constexpr parameters.
    with _on_device(x):
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
            K=k,  # ty: ignore[invalid-argument-type]
            LEFT=k // 2,  # ty: ignore[invalid-argument-type]
            HAS_SEG=segment_ids is not None,  # ty: ignore[invalid-argument-type]
            RESIDUAL=residual,  # ty: ignore[invalid-argument-type]
            CONV_DTYPE=_TL_DTYPES[out_dtype],  # ty: ignore[invalid-argument-type]
            ROUND_CONV=interpret and out_dtype == torch.bfloat16,  # ty: ignore[invalid-argument-type]
            ROUND_OUT=interpret and out.dtype == torch.bfloat16,  # ty: ignore[invalid-argument-type]
            BLOCK_D=cfg.block_d,  # ty: ignore[invalid-argument-type]
            STRIP=cfg.strip,  # ty: ignore[invalid-argument-type]
            UNROLL=cfg.unroll,  # ty: ignore[invalid-argument-type]
            num_warps=cfg.num_warps,  # ty: ignore[unknown-argument]
            num_stages=cfg.num_stages,  # ty: ignore[unknown-argument]
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
    *,
    config: BwdConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gradients of `canon_forward` for its output gradient `dy`: `(dx, dw)`.

    `out_dtype` is the forward's conv dtype: the conv path sees `dy` rounded to it, as autograd's
    cast backward does; the residual path sees the raw `dy`. `dx` is `(B, T, D)` in `x.dtype`;
    `dw` is `(D, k)` in `weight.dtype`, reduced deterministically (per-program fp32 partials,
    then a sum). `config` overrides the `BWD_CONFIGS` launch (benchmark sweeps).
    """
    B, T, D = x.shape
    _check_row_size(T, D)
    x = x.contiguous()
    dy = dy.contiguous()
    k = weight.shape[1]
    cfg = config or BWD_CONFIGS.get((D, x.dtype, out_dtype), BwdConfig())
    _check_strip(cfg)
    dx = torch.empty_like(x)
    interpret = os.environ.get("TRITON_INTERPRET") == "1"
    if x.numel() == 0:
        return dx, torch.zeros_like(weight)
    seg = segment_ids if segment_ids is not None else attention_mask  # unread when None
    grid = (B, triton.cdiv(T, cfg.strip), triton.cdiv(D, cfg.block_d))
    partials = torch.empty(B * grid[1], D, k, device=x.device, dtype=torch.float32)
    # ty ignores: Triton boundary, plain values for tl.constexpr parameters.
    with _on_device(x):
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
            K=k,  # ty: ignore[invalid-argument-type]
            LEFT=k // 2,  # ty: ignore[invalid-argument-type]
            HAS_SEG=segment_ids is not None,  # ty: ignore[invalid-argument-type]
            RESIDUAL=residual,  # ty: ignore[invalid-argument-type]
            CONV_DTYPE=_TL_DTYPES[out_dtype],  # ty: ignore[invalid-argument-type]
            ROUND_G=interpret and out_dtype == torch.bfloat16,  # ty: ignore[invalid-argument-type]
            ROUND_DX=interpret and x.dtype == torch.bfloat16,  # ty: ignore[invalid-argument-type]
            BLOCK_D=cfg.block_d,  # ty: ignore[invalid-argument-type]
            STRIP=cfg.strip,  # ty: ignore[invalid-argument-type]
            UNROLL=cfg.unroll,  # ty: ignore[invalid-argument-type]
            num_warps=cfg.num_warps,  # ty: ignore[unknown-argument]
            num_stages=cfg.num_stages,  # ty: ignore[unknown-argument]
        )
    return dx, partials.sum(0).to(weight.dtype)


# `custom_op`, not `triton_op`: Inductor traces into a `triton_op` body and re-emits its kernels,
# and the op must stay opaque. Registration needs no Triton (the kernels launch only when the op
# runs), so `import oplm` works on CPU-only installs; `CanonConv` dispatches here only on CUDA.
# The backward is an op of its own so AOTAutograd traces it through its fake impl.
@torch.library.custom_op("oplm::canon_conv", mutates_args=())
def _canon_conv_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    attention_mask: torch.Tensor,
    segment_ids: torch.Tensor | None,
    out_dtype: torch.dtype,
    residual: bool,
) -> torch.Tensor:
    return canon_forward(x, weight, attention_mask, segment_ids, out_dtype, residual)


@_canon_conv_op.register_fake
def _(x, weight, attention_mask, segment_ids, out_dtype, residual):  # noqa: ANN001, ANN202
    return x.new_empty(
        x.shape, dtype=torch.promote_types(x.dtype, out_dtype) if residual else out_dtype
    )


@torch.library.custom_op("oplm::canon_conv_backward", mutates_args=())
def _canon_conv_backward_op(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    attention_mask: torch.Tensor,
    segment_ids: torch.Tensor | None,
    out_dtype: torch.dtype,
    residual: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return canon_backward(dy, x, weight, attention_mask, segment_ids, out_dtype, residual)


@_canon_conv_backward_op.register_fake
def _(dy, x, weight, attention_mask, segment_ids, out_dtype, residual):  # noqa: ANN001, ANN202
    return x.new_empty(x.shape), weight.new_empty(weight.shape)


def _setup_context(ctx, inputs, output):  # noqa: ANN001, ANN202
    x, weight, attention_mask, segment_ids, out_dtype, residual = inputs
    ctx.save_for_backward(x, weight, attention_mask, segment_ids)
    ctx.out_dtype = out_dtype
    ctx.residual = residual


def _backward(ctx, dy):  # noqa: ANN001, ANN202
    x, weight, attention_mask, segment_ids = ctx.saved_tensors
    dx, dw = _canon_conv_backward_op(
        dy, x, weight, attention_mask, segment_ids, ctx.out_dtype, ctx.residual
    )
    return dx, dw, None, None, None, None  # no gradient for the masks or the flags


_canon_conv_op.register_autograd(_backward, setup_context=_setup_context)


def canon_conv_kernel(
    x: torch.Tensor,
    weight: torch.Tensor,
    attention_mask: torch.Tensor,
    segment_ids: torch.Tensor | None,
    out_dtype: torch.dtype,
    residual: bool,
) -> torch.Tensor:
    """`torch.ops.oplm.canon_conv`: Triton Canon conv with autograd; same signature and function
    as `conv.canon_conv_reference`."""
    return torch.ops.oplm.canon_conv(x, weight, attention_mask, segment_ids, out_dtype, residual)


class CanonKernelMismatch(RuntimeError):
    """The Canon kernel disagreed with `canon_conv_reference` on a model's real inputs."""


def canon_self_check(
    model: nn.Module,
    run_forward: Callable[[], object],
    kernel: Callable[..., torch.Tensor] = canon_conv_kernel,
) -> int:
    """Check `kernel` against the reference on every Canon call of one `run_forward()`.

    Each `CanonConv` call's `x`, `attention_mask` and `segment_ids` go through `kernel` and
    `canon_conv_reference` in fp32 with the residual flag `CanonConv.forward` gives the op (the
    conv before any activation); one random `dy` is backpropagated through both, and the output,
    `dx` and `dW` must each agree within `rtol = 1e-4`, `atol = 1e-4 * max(1, max|ref|)` (finite
    entries; matching NaNs pass), or `CanonKernelMismatch` is raised. The forward runs under
    `no_grad` on detached inputs (no parameter `.grad`), the RNG state (CPU and the model's CUDA
    device) is restored after, and the hooks are always removed. Returns the number of calls
    checked.
    """
    from .conv import CanonConv, canon_conv_reference  # conv.py imports this module

    convs = [(n, m) for n, m in model.named_modules() if isinstance(m, CanonConv)]
    if not convs:
        return 0
    calls = 0

    def make_hook(name: str) -> Callable[..., None]:
        def hook(module: CanonConv, args: tuple, kwargs: dict) -> None:
            nonlocal calls
            call = inspect.signature(module.forward).bind(*args, **kwargs)
            call.apply_defaults()
            a = call.arguments
            residual = bool(a["residual"]) and module._activation is None  # as forward does
            x = a["x"].detach().float().requires_grad_()
            w = module.conv.weight.detach().squeeze(1).float().requires_grad_()
            results = []
            with torch.enable_grad(), torch.autocast(x.device.type, enabled=False):
                dy = torch.randn_like(x)
                for fn in (kernel, canon_conv_reference):
                    out = fn(x, w, a["attention_mask"], a["segment_ids"], torch.float32, residual)
                    results.append((out.detach(), *torch.autograd.grad(out, (x, w), dy)))
            # Scale-relative atol: dW sums over every token, so its fp32 error grows with |x|.
            checks = [
                (n, got, ref, 1e-4 * ref.abs().nan_to_num(0.0, 0.0).max().clamp_min(1).item())
                for n, got, ref in zip(("out", "dx", "dW"), *results, strict=True)
            ]
            if not all(
                torch.allclose(got, ref, rtol=1e-4, atol=atol, equal_nan=True)
                for _, got, ref, atol in checks
            ):
                errors = ", ".join(
                    f"{n} {(g - r).abs().max().item():.3g} (atol {atol:.3g})"
                    for n, g, r, atol in checks
                )
                raise CanonKernelMismatch(
                    f"Canon kernel != reference at {name} (input {tuple(x.shape)}, "
                    f"k={module.kernel_size}, residual={residual}); max abs error vs tol "
                    f"(rtol 1e-4): {errors}"
                )
            calls += 1

        return hook

    device = convs[0][1].conv.weight.device
    handles = [m.register_forward_pre_hook(make_hook(n), with_kwargs=True) for n, m in convs]
    try:
        rng_devices = [device] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=rng_devices), torch.no_grad():
            run_forward()
    finally:
        for handle in handles:
            handle.remove()
    return calls
