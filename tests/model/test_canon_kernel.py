"""Triton Canon kernel vs `canon_conv_reference`.

Runs on CUDA, or on CPU under `TRITON_INTERPRET=1` (Triton's interpreter), so the kernels are
checked without a GPU: `TRITON_INTERPRET=1 pytest tests/model/test_canon_kernel.py`.
"""

from __future__ import annotations

import copy
import os

import pytest
import torch

from oplm.model import conv as conv_mod
from oplm.model.conv import CanonConv, canon_conv_reference

# Kernel tests; the dispatch tests at the end set their own marks.
kernel = pytest.mark.skipif(
    not (torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET") == "1"),
    reason="needs CUDA or TRITON_INTERPRET=1",
)
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from oplm.model.canon_kernel import (  # noqa: E402
    BLOCK_T,
    BWD_TILES,
    canon_backward,
    canon_conv_kernel,
    canon_forward,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FP32_TOL = {"rtol": 1e-5, "atol": 1e-5}
BF16_TOL = {"rtol": 1.6e-2, "atol": 1e-2}


def _segments_from_lengths(lengths: list[int], T: int) -> torch.Tensor:
    """Row of segment ids `1..n` for the given lengths, then `0` pads up to `T`."""
    seg = torch.zeros(T, dtype=torch.long)
    pos = 0
    for i, n in enumerate(lengths, start=1):
        seg[pos : pos + n] = i
        pos += n
    return seg


def _random_lengths(T: int, gen: torch.Generator) -> list[int]:
    """Random segment lengths filling a random prefix of `T` (the rest is trailing pad)."""
    used = int(torch.randint(1, T + 1, (1,), generator=gen))
    lengths, pos = [], 0
    while pos < used:
        n = min(int(torch.randint(1, max(2, T // 3) + 1, (1,), generator=gen)), used - pos)
        lengths.append(n)
        pos += n
    return lengths


def _case(
    B: int,
    T: int,
    D: int,
    k: int,
    packed: bool,
    seed: int,
    dtype: torch.dtype = torch.float32,
    all_pad_row: bool = True,
    interior_pads: bool = False,
):
    """`(x, weight, attention_mask, segment_ids)`; `segment_ids` is None unless `packed`.

    Rows get random segment lengths and trailing pads; with `all_pad_row` and `B > 1` the last row
    is all pad; with `interior_pads` about a fifth of positions anywhere in a row are pads too.
    Pad positions of `x` hold large garbage so a leak shows up.
    """
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(B, T, D, generator=gen)
    weight = torch.randn(D, k, generator=gen)
    seg = torch.stack([_segments_from_lengths(_random_lengths(T, gen), T) for _ in range(B)])
    if interior_pads:
        seg[torch.rand(B, T, generator=gen) < 0.2] = 0
    if all_pad_row and B > 1:
        seg[-1] = 0
    mask = seg.ne(0)
    x = torch.where(mask.unsqueeze(-1), x, torch.full_like(x, 1e3))
    return (
        x.to(DEVICE, dtype),
        weight.to(DEVICE),
        mask.to(DEVICE, torch.long),
        seg.to(DEVICE) if packed else None,
    )


@kernel
@pytest.mark.parametrize("residual", [True, False])
@pytest.mark.parametrize("packed", [True, False])
@pytest.mark.parametrize("T", [1, 2, 37, 128])
@pytest.mark.parametrize("D", [64, 100])
@pytest.mark.parametrize("k", [2, 3, 4, 7])
def test_forward_matches_reference_fp32(k, D, T, packed, residual):
    x, w, mask, seg = _case(2, T, D, k, packed, seed=k * 1000 + T)
    out = canon_forward(x, w, mask, seg, torch.float32, residual)
    ref = canon_conv_reference(x, w, mask, seg, torch.float32, residual)
    assert out.dtype == ref.dtype
    torch.testing.assert_close(out, ref, **FP32_TOL)


@kernel
@pytest.mark.parametrize("residual", [True, False])
@pytest.mark.parametrize("x_dtype", [torch.bfloat16, torch.float32])
def test_forward_bf16_and_mixed_dtypes(x_dtype, residual):
    x, w, mask, seg = _case(2, 37, 100, 7, True, seed=7, dtype=x_dtype)
    out = canon_forward(x, w, mask, seg, torch.bfloat16, residual)
    ref = canon_conv_reference(x, w, mask, seg, torch.bfloat16, residual)
    # bf16 conv; the residual takes the promoted dtype (fp32 when `x` is fp32).
    assert out.dtype == ref.dtype == (x_dtype if residual else torch.bfloat16)
    torch.testing.assert_close(out, ref, **BF16_TOL)
    if residual:  # raw `x` plus the bf16 conv, rounded as torch rounds `x + conv`
        assert torch.equal(out, x + canon_forward(x, w, mask, seg, torch.bfloat16, False))


@kernel
def test_forward_bool_mask():
    x, w, mask, seg = _case(2, 37, 64, 3, False, seed=3)
    out = canon_forward(x, w, mask.bool(), seg, torch.float32, False)
    ref = canon_conv_reference(x, w, mask, seg, torch.float32, False)
    torch.testing.assert_close(out, ref, **FP32_TOL)


def _edge_layouts(T: int, edges: list[int], n: int = 20):
    """`n` single-row `(x, weight, attention_mask, segment_ids)` cases of length `T`.

    Each has one segment boundary on one of `edges` (cycled), a few random ones elsewhere, and
    random trailing pad; `k` cycles through 2, 3, 4, 7.
    """
    gen = torch.Generator().manual_seed(0)
    for layout in range(n):
        cuts = {edges[layout % len(edges)]}
        cuts |= {int(c) for c in torch.randint(1, T, (3,), generator=gen)}
        end = int(torch.randint(max(cuts) + 1, T + 1, (1,), generator=gen))
        bounds = [0, *sorted(cuts), end]
        seg = _segments_from_lengths(torch.tensor(bounds).diff().tolist(), T)
        k = (2, 3, 4, 7)[layout % 4]
        x = torch.randn(1, T, 64, generator=gen).to(DEVICE)
        w = torch.randn(64, k, generator=gen).to(DEVICE)
        mask = seg.ne(0).long().unsqueeze(0).to(DEVICE)
        yield x, w, mask, seg.unsqueeze(0).to(DEVICE)


@kernel
def test_forward_segment_boundaries_on_tile_edges():
    for x, w, mask, seg in _edge_layouts(2 * BLOCK_T + 3, [BLOCK_T - 1, BLOCK_T, BLOCK_T + 1]):
        for residual in (True, False):
            out = canon_forward(x, w, mask, seg, torch.float32, residual)
            ref = canon_conv_reference(x, w, mask, seg, torch.float32, residual)
            torch.testing.assert_close(out, ref, **FP32_TOL)


@kernel
@pytest.mark.parametrize("packed", [True, False])
def test_forward_degenerate_rows(packed):
    # All-pad row: conv is zero, residual returns the raw `x` (garbage pads included).
    x, w, _, _ = _case(1, 5, 64, 3, packed, seed=1)
    mask = torch.zeros(1, 5, dtype=torch.long, device=DEVICE)
    seg = torch.zeros_like(mask) if packed else None
    assert torch.equal(canon_forward(x, w, mask, seg, torch.float32, False), torch.zeros_like(x))
    assert torch.equal(canon_forward(x, w, mask, seg, torch.float32, True), x)

    # T < k, and rows of single-token segments.
    for T, k, lengths in ((2, 7, [2]), (1, 4, [1]), (6, 3, [1] * 6), (5, 7, [1, 1, 1])):
        seg_row = _segments_from_lengths(lengths, T).unsqueeze(0).to(DEVICE)
        mask = seg_row.ne(0).long()
        seg = seg_row if packed else None
        x = torch.randn(1, T, 64, device=DEVICE)
        w = torch.randn(64, k, device=DEVICE)
        for residual in (True, False):
            out = canon_forward(x, w, mask, seg, torch.float32, residual)
            assert torch.isfinite(out).all()
            ref = canon_conv_reference(x, w, mask, seg, torch.float32, residual)
            torch.testing.assert_close(out, ref, **FP32_TOL)


@kernel
def test_forward_residual_keeps_raw_pads():
    # Packed: a pad's segment id (0) matches only other (zeroed) pads, so its conv is exactly 0.
    for dtype in (torch.float32, torch.bfloat16):
        x, w, mask, seg = _case(3, 37, 64, 7, True, seed=5, dtype=dtype)
        out = canon_forward(x, w, mask, seg, dtype, True)
        pad = mask.eq(0)
        assert pad.any()
        assert torch.equal(out[pad], x[pad])


@kernel
@pytest.mark.parametrize("residual", [True, False])
@pytest.mark.parametrize("k", [2, 7])
def test_forward_interior_pads(k, residual):
    # Unpacked rows with pads in the middle: zeroed before the conv, so neighbors read 0 there.
    x, w, mask, seg = _case(2, 37, 64, k, False, seed=11, interior_pads=True)
    assert (mask[0, :-1].eq(0) & mask[0, 1:].ne(0)).any()  # a pad followed by a real token
    out = canon_forward(x, w, mask, seg, torch.float32, residual)
    ref = canon_conv_reference(x, w, mask, seg, torch.float32, residual)
    torch.testing.assert_close(out, ref, **FP32_TOL)


@kernel
def test_row_size_guard():
    # In-row offsets are int32; a row of 2**31 elements must be refused, not overflow.
    T, D = 2**16, 2**15
    x = torch.zeros(1, 1, 1, device=DEVICE).expand(1, T, D)  # no 8 GB allocation
    w = torch.zeros(D, 3, device=DEVICE)
    mask = torch.ones(1, T, dtype=torch.long, device=DEVICE)
    with pytest.raises(ValueError, match="2\\*\\*31"):
        canon_forward(x, w, mask, None, torch.float32, False)
    with pytest.raises(ValueError, match="2\\*\\*31"):
        canon_backward(x, x, w, mask, None, torch.float32, False)


def _grads(x, w, mask, seg, out_dtype, residual, seed=0):
    """`(dx, dw, ref_dx, ref_dw, dy)`: `canon_backward` vs autograd through the reference.

    `dy` is random (pads included) in the forward output's dtype.
    """
    y_dtype = torch.promote_types(x.dtype, out_dtype) if residual else out_dtype
    gen = torch.Generator().manual_seed(seed)
    dy = torch.randn(x.shape, generator=gen).to(DEVICE, y_dtype)
    xr = x.detach().requires_grad_()
    wr = w.detach().requires_grad_()
    canon_conv_reference(xr, wr, mask, seg, out_dtype, residual).backward(dy)
    dx, dw = canon_backward(dy, x, w, mask, seg, out_dtype, residual)
    assert dx.dtype == x.dtype and dx.shape == x.shape
    assert dw.dtype == w.dtype and dw.shape == w.shape
    return dx, dw, xr.grad, wr.grad, dy


def _assert_pad_grads(dx, dy, mask, residual):
    """At pads `dx` is exactly `dy` (residual) or `0`: the conv never reads a pad."""
    pad = mask.eq(0)
    expect = dy[pad].to(dx.dtype) if residual else torch.zeros_like(dx[pad])
    assert torch.equal(dx[pad], expect)


@kernel
@pytest.mark.parametrize("residual", [True, False])
@pytest.mark.parametrize("packed", [True, False])
@pytest.mark.parametrize("T", [1, 2, 37, 128])
@pytest.mark.parametrize("D", [64, 100])
@pytest.mark.parametrize("k", [2, 3, 4, 7])
def test_backward_matches_reference_fp32(k, D, T, packed, residual):
    # Unpacked cases also get interior pads.
    x, w, mask, seg = _case(2, T, D, k, packed, seed=k * 1000 + T, interior_pads=not packed)
    dx, dw, ref_dx, ref_dw, dy = _grads(x, w, mask, seg, torch.float32, residual)
    torch.testing.assert_close(dx, ref_dx, **FP32_TOL)
    torch.testing.assert_close(dw, ref_dw, **FP32_TOL)
    _assert_pad_grads(dx, dy, mask, residual)


@kernel
@pytest.mark.parametrize("residual", [True, False])
@pytest.mark.parametrize("packed", [True, False])
def test_backward_bf16(packed, residual):
    # bf16 x and conv (autocast), fp32 weight: dx is bf16, dw fp32. dw's products are exact in
    # fp32 (bf16 x bf16), so only the summation order differs from the reference: fp32 tolerance.
    x, w, mask, seg = _case(2, 37, 100, 7, packed, seed=7, dtype=torch.bfloat16, interior_pads=True)
    dx, dw, ref_dx, ref_dw, dy = _grads(x, w, mask, seg, torch.bfloat16, residual)
    torch.testing.assert_close(dx, ref_dx, **BF16_TOL)
    torch.testing.assert_close(dw, ref_dw, **FP32_TOL)
    _assert_pad_grads(dx, dy, mask, residual)


@kernel
@pytest.mark.parametrize("residual", [True, False])
@pytest.mark.parametrize("packed", [True, False])
def test_backward_mixed_dtypes(packed, residual):
    # fp32 x under bf16 autocast (LayerNorm output into Canon): the conv-path gradient sees dy
    # rounded to bf16, as autograd does through the conv's cast; the residual path sees raw dy.
    x, w, mask, seg = _case(2, 37, 100, 7, packed, seed=8, interior_pads=not packed)
    dx, dw, ref_dx, ref_dw, dy = _grads(x, w, mask, seg, torch.bfloat16, residual)
    assert dy.dtype == (torch.float32 if residual else torch.bfloat16)
    torch.testing.assert_close(dx, ref_dx, **FP32_TOL)
    torch.testing.assert_close(dw, ref_dw, **FP32_TOL)
    _assert_pad_grads(dx, dy, mask, residual)


@kernel
def test_backward_bool_mask():
    x, w, mask, seg = _case(2, 37, 64, 3, False, seed=3, interior_pads=True)
    dx, dw, ref_dx, ref_dw, _ = _grads(x, w, mask.bool(), seg, torch.float32, False)
    torch.testing.assert_close(dx, ref_dx, **FP32_TOL)
    torch.testing.assert_close(dw, ref_dw, **FP32_TOL)


@kernel
def test_backward_tile_edges():
    # Boundaries on token-tile edges and on the edges of a program's span of BWD_TILES tiles.
    span = BLOCK_T * BWD_TILES
    edges = [BLOCK_T - 1, BLOCK_T, BLOCK_T + 1, span - 1, span, span + 1]
    for x, w, mask, seg in _edge_layouts(span + BLOCK_T + 3, edges):
        for residual in (True, False):
            dx, dw, ref_dx, ref_dw, dy = _grads(x, w, mask, seg, torch.float32, residual)
            torch.testing.assert_close(dx, ref_dx, **FP32_TOL)
            torch.testing.assert_close(dw, ref_dw, **FP32_TOL)
            _assert_pad_grads(dx, dy, mask, residual)


@kernel
@pytest.mark.parametrize("packed", [True, False])
def test_backward_degenerate_rows(packed):
    # All-pad row: no gradient into the conv, so dx is 0 (or dy) and dw is exactly 0.
    x, w, _, _ = _case(1, 5, 64, 3, packed, seed=1)
    mask = torch.zeros(1, 5, dtype=torch.long, device=DEVICE)
    seg = torch.zeros_like(mask) if packed else None
    gen = torch.Generator().manual_seed(0)
    for residual in (True, False):
        dy = torch.randn(x.shape, generator=gen).to(DEVICE)
        dx, dw = canon_backward(dy, x, w, mask, seg, torch.float32, residual)
        assert torch.equal(dx, dy if residual else torch.zeros_like(x))
        assert torch.equal(dw, torch.zeros_like(w))

    # An all-pad row adds nothing to dw: same dw with and without it in the batch.
    x, w, mask, seg = _case(3, 37, 64, 7, packed, seed=2)
    assert mask[-1].eq(0).all()
    dy = torch.randn(x.shape, generator=gen).to(DEVICE)
    dw_all = canon_backward(dy, x, w, mask, seg, torch.float32, False)[1]
    seg_two = None if seg is None else seg[:2]
    dw_two = canon_backward(dy[:2], x[:2], w, mask[:2], seg_two, torch.float32, False)[1]
    assert torch.equal(dw_all, dw_two)

    # T < k, and rows of single-token segments.
    for T, k, lengths in ((2, 7, [2]), (1, 4, [1]), (6, 3, [1] * 6), (5, 7, [1, 1, 1])):
        seg_row = _segments_from_lengths(lengths, T).unsqueeze(0).to(DEVICE)
        mask = seg_row.ne(0).long()
        seg = seg_row if packed else None
        x = torch.randn(1, T, 64, device=DEVICE)
        w = torch.randn(64, k, device=DEVICE)
        for residual in (True, False):
            dx, dw, ref_dx, ref_dw, dy = _grads(x, w, mask, seg, torch.float32, residual)
            assert torch.isfinite(dx).all() and torch.isfinite(dw).all()
            torch.testing.assert_close(dx, ref_dx, **FP32_TOL)
            torch.testing.assert_close(dw, ref_dw, **FP32_TOL)
            _assert_pad_grads(dx, dy, mask, residual)


@kernel
def test_backward_deterministic():
    x, w, mask, seg = _case(4, 300, 100, 7, True, seed=3)
    dy = torch.randn(x.shape, generator=torch.Generator().manual_seed(4)).to(DEVICE)
    dx1, dw1 = canon_backward(dy, x, w, mask, seg, torch.float32, True)
    dx2, dw2 = canon_backward(dy, x, w, mask, seg, torch.float32, True)
    assert torch.equal(dw1, dw2)
    assert torch.equal(dx1, dx2)


# --- the oplm::canon_conv op, its autograd, and CanonConv dispatch ---


@kernel
@pytest.mark.parametrize(
    ("packed", "dtype", "residual"),
    [(True, torch.bfloat16, True), (False, torch.float32, False)],
    ids=["packed-bf16", "unpacked-fp32"],
)
def test_op_opcheck(packed, dtype, residual):
    x, w, mask, seg = _case(2, 37, 100, 7, packed, seed=21, dtype=dtype, interior_pads=not packed)
    args = (x.requires_grad_(), w.requires_grad_(), mask, seg, dtype, residual)
    torch.library.opcheck(torch.ops.oplm.canon_conv, args)


def _op_and_reference_grads(fn, x, w, mask, seg, out_dtype, residual, dy):
    """Forward + backward of `fn` with the weight as a `(D, k)` view of a `(D, 1, k)` leaf (as
    `CanonConv` passes `conv.weight.squeeze(1)`): `(out, dx, dw)`, `dw` shaped `(D, 1, k)`."""
    xr = x.detach().requires_grad_()
    param = w.detach().unsqueeze(1).clone().requires_grad_()
    out = fn(xr, param.squeeze(1), mask, seg, out_dtype, residual)
    out.backward(dy.to(out.dtype))
    return out, xr.grad, param.grad


@kernel
@pytest.mark.parametrize("residual", [True, False])
@pytest.mark.parametrize(
    ("x_dtype", "out_dtype"),
    [
        (torch.float32, torch.float32),
        (torch.float32, torch.bfloat16),  # fp32 LayerNorm output under bf16 autocast
        (torch.bfloat16, torch.bfloat16),
    ],
)
def test_op_autograd_matches_reference(x_dtype, out_dtype, residual):
    x, w, mask, seg = _case(2, 37, 100, 7, True, seed=22, dtype=x_dtype)
    dy = torch.randn(x.shape, generator=torch.Generator().manual_seed(0)).to(DEVICE)
    got = _op_and_reference_grads(canon_conv_kernel, x, w, mask, seg, out_dtype, residual, dy)
    ref = _op_and_reference_grads(canon_conv_reference, x, w, mask, seg, out_dtype, residual, dy)
    out_tol = FP32_TOL if out_dtype == torch.float32 else BF16_TOL
    dx_tol = FP32_TOL if x_dtype == torch.float32 else BF16_TOL
    assert got[0].dtype == ref[0].dtype and got[1].dtype == x_dtype
    assert got[2].shape == (100, 1, 7) and got[2].dtype == torch.float32
    torch.testing.assert_close(got[0], ref[0], **out_tol)
    torch.testing.assert_close(got[1], ref[1], **dx_tol)
    torch.testing.assert_close(got[2], ref[2], **FP32_TOL)


@kernel
def test_op_compiles_without_graph_breaks():
    # The op is opaque to Dynamo/AOTAutograd: fullgraph tracing (fake impls for the forward and
    # the backward op) succeeds and runs the same kernels as eager.
    x, w, mask, seg = _case(2, 37, 100, 7, True, seed=23)
    dy = torch.randn(x.shape, generator=torch.Generator().manual_seed(0)).to(DEVICE)
    compiled = torch.compile(canon_conv_kernel, backend="aot_eager", fullgraph=True)
    got = _op_and_reference_grads(compiled, x, w, mask, seg, torch.float32, True, dy)
    ref = _op_and_reference_grads(canon_conv_kernel, x, w, mask, seg, torch.float32, True, dy)
    for g, r in zip(got, ref, strict=True):
        assert torch.equal(g, r)


@pytest.mark.parametrize("residual", [True, False])
@pytest.mark.parametrize("activation", ["none", "silu"])
def test_cpu_uses_reference(monkeypatch, activation, residual):
    def no_kernel(*args):  # noqa: ANN002, ANN202
        raise AssertionError("CPU tensors must not take the kernel path")

    monkeypatch.setattr(conv_mod, "canon_conv_kernel", no_kernel)
    torch.manual_seed(0)
    module = CanonConv(100, 7, activation)
    x, _, mask, seg = (t.cpu() for t in _case(2, 37, 100, 7, True, seed=24))
    weight = module.conv.weight.squeeze(1)
    with torch.no_grad():
        out = module(x, mask, seg, residual=residual)
        conv = canon_conv_reference(x, weight, mask, seg, torch.float32, False)
        if activation == "silu":
            conv = torch.nn.functional.silu(conv)
    assert torch.equal(out, x + conv if residual else conv)


@needs_cuda
@pytest.mark.parametrize("autocast", [False, True], ids=["fp32", "bf16-autocast"])
@pytest.mark.parametrize("residual", [True, False])
@pytest.mark.parametrize("activation", ["none", "silu"])
def test_module_kernel_matches_reference(monkeypatch, activation, residual, autocast):
    calls = []
    real = conv_mod.canon_conv_kernel

    def spy(*args):  # noqa: ANN002, ANN202
        calls.append(args[-1])  # the residual flag the module passed
        return real(*args)

    monkeypatch.setattr(conv_mod, "canon_conv_kernel", spy)
    torch.manual_seed(0)
    module = CanonConv(100, 7, activation).cuda()
    x, _, mask, seg = _case(2, 37, 100, 7, True, seed=25)  # fp32, like a LayerNorm output
    dy = torch.randn(x.shape, generator=torch.Generator().manual_seed(0)).to(DEVICE)
    results = []
    for enabled in (False, True):
        monkeypatch.setattr(conv_mod, "_KERNEL_ENABLED", enabled)
        module.zero_grad(set_to_none=True)
        xr = x.clone().requires_grad_()
        with torch.autocast("cuda", torch.bfloat16, enabled=autocast):
            out = module(xr, mask, seg, residual=residual)
        out.backward(dy.to(out.dtype))
        results.append((out, xr.grad, module.conv.weight.grad))
    # One kernel call, on the enabled pass; the residual is folded in only without activation.
    assert calls == [residual and activation == "none"]
    tol = BF16_TOL if autocast else FP32_TOL
    for got, ref in zip(results[1], results[0], strict=True):
        assert got.dtype == ref.dtype
        torch.testing.assert_close(got, ref, **tol)


def test_dispatch_predicate(monkeypatch):
    # Kernel only for CUDA tensors with Triton and kernel-tested dtypes; fp64 (`model.double()`)
    # and fp16 take the reference instead of a KeyError in the launch.
    from types import SimpleNamespace

    monkeypatch.setattr(conv_mod, "TRITON_AVAILABLE", True)
    f32, bf16 = torch.float32, torch.bfloat16

    def cuda(dtype):  # noqa: ANN001, ANN202 — a stand-in: CPU-only machines have no CUDA tensor
        return SimpleNamespace(is_cuda=True, dtype=dtype)

    assert conv_mod._use_kernel(cuda(f32), f32)
    assert conv_mod._use_kernel(cuda(f32), bf16)
    assert conv_mod._use_kernel(cuda(bf16), bf16)
    assert not conv_mod._use_kernel(cuda(torch.float64), torch.float64)
    assert not conv_mod._use_kernel(cuda(torch.float64), bf16)
    assert not conv_mod._use_kernel(cuda(f32), torch.float16)
    assert not conv_mod._use_kernel(torch.zeros(1), f32)  # CPU
    monkeypatch.setattr(conv_mod, "_KERNEL_ENABLED", False)
    assert not conv_mod._use_kernel(cuda(f32), f32)
    monkeypatch.setattr(conv_mod, "_KERNEL_ENABLED", True)
    monkeypatch.setattr(conv_mod, "TRITON_AVAILABLE", False)
    assert not conv_mod._use_kernel(cuda(f32), f32)


@needs_cuda
@pytest.mark.parametrize("autocast", [False, True], ids=["fp32", "bf16-autocast"])
def test_compiled_block_no_graph_breaks(monkeypatch, autocast):
    from torch._inductor.compile_fx import compile_fx

    from oplm import OplmConfig
    from oplm.model.masking import segment_block_mask
    from oplm.model.transformer import OplmBlock

    torch.manual_seed(0)
    cfg = OplmConfig(
        hidden_size=64,
        num_attention_heads=4,
        intermediate_size=128,
        num_hidden_layers=1,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        canon_enabled=True,
        canon_positions=["A", "B", "C", "D"],
        canon_kernel_sizes=[7],
        packed_attention="flex",
    )
    ref_block = OplmBlock(cfg, layer_idx=0).cuda().train()
    kernel_block = copy.deepcopy(ref_block)
    _, _, mask, seg = _case(2, 256, 64, 7, True, seed=26, all_pad_row=False)
    x = torch.randn(2, 256, 64, device=DEVICE)
    dy = torch.randn_like(x)

    def run(module, enabled):  # noqa: ANN001, ANN202
        """Inductor-compiled (fullgraph) forward + backward, and whether its graph has the op."""
        monkeypatch.setattr(conv_mod, "_KERNEL_ENABLED", enabled)
        torch._dynamo.reset()  # fresh compile per side: no cache entry shared across the flag
        graphs = []

        def inductor(gm, example_inputs):  # noqa: ANN001, ANN202 — what backend="inductor" runs
            graphs.append(gm)
            return compile_fx(gm, example_inputs)

        # As OplmStack does: the flex block mask is built inside the compiled forward.
        def fn(xr):  # noqa: ANN001, ANN202
            return module(xr, mask, segment_ids=seg, block_mask=segment_block_mask(seg))[0]

        xr = x.clone().requires_grad_()
        with torch.autocast("cuda", torch.bfloat16, enabled=autocast):
            y = torch.compile(fn, backend=inductor, fullgraph=True)(xr)
        y.backward(dy.to(y.dtype))
        has_op = any("oplm.canon_conv" in str(n.target) for g in graphs for n in g.graph.nodes)
        return has_op, [y, xr.grad] + [p.grad for p in module.parameters()]

    ref_has_op, ref = run(ref_block, False)
    got_has_op, got = run(kernel_block, True)
    assert got_has_op and not ref_has_op  # only the Canon conv differs between the two sides
    assert all(g is not None for g in got)
    for g, r in zip(got, ref, strict=True):
        assert g.dtype == r.dtype and g.shape == r.shape
    if not autocast:
        # Measured on CPU (Inductor both sides, kernel under the interpreter, SDPA): elementwise
        # FP32_TOL holds, max error ~1e-6 of each tensor's scale. 10x headroom for the GPU.
        for g, r in zip(got, ref, strict=True):
            torch.testing.assert_close(g, r, rtol=1e-4, atol=1e-4)
    else:
        # bf16 autocast: the gradients are themselves bf16 matmul outputs, so a 1-ulp change in
        # a conv output flips roundings downstream; the same CPU measurement gave max error up
        # to 1.2e-2 of each tensor's scale (elementwise BF16_TOL fails on near-zero elements).
        # This side checks the production dtype path end to end, for gross errors only; the fp32
        # case above is the precision check.
        for g, r in zip(got, ref, strict=True):
            assert (g - r).abs().max() <= 3e-2 * r.abs().max()
