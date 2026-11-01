"""Triton Canon kernel vs `canon_conv_reference`.

Runs on CUDA, or on CPU under `TRITON_INTERPRET=1` (Triton's interpreter), so the kernels are
checked without a GPU: `TRITON_INTERPRET=1 pytest tests/model/test_canon_kernel.py`.
"""

from __future__ import annotations

import os

import pytest
import torch

from oplm.model.conv import canon_conv_reference

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET") == "1"),
    reason="needs CUDA or TRITON_INTERPRET=1",
)

from oplm.model.canon_kernel import (  # noqa: E402
    BLOCK_T,
    BWD_TILES,
    canon_backward,
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


def test_forward_segment_boundaries_on_tile_edges():
    for x, w, mask, seg in _edge_layouts(2 * BLOCK_T + 3, [BLOCK_T - 1, BLOCK_T, BLOCK_T + 1]):
        for residual in (True, False):
            out = canon_forward(x, w, mask, seg, torch.float32, residual)
            ref = canon_conv_reference(x, w, mask, seg, torch.float32, residual)
            torch.testing.assert_close(out, ref, **FP32_TOL)


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


def test_forward_residual_keeps_raw_pads():
    # Packed: a pad's segment id (0) matches only other (zeroed) pads, so its conv is exactly 0.
    for dtype in (torch.float32, torch.bfloat16):
        x, w, mask, seg = _case(3, 37, 64, 7, True, seed=5, dtype=dtype)
        out = canon_forward(x, w, mask, seg, dtype, True)
        pad = mask.eq(0)
        assert pad.any()
        assert torch.equal(out[pad], x[pad])


@pytest.mark.parametrize("residual", [True, False])
@pytest.mark.parametrize("k", [2, 7])
def test_forward_interior_pads(k, residual):
    # Unpacked rows with pads in the middle: zeroed before the conv, so neighbors read 0 there.
    x, w, mask, seg = _case(2, 37, 64, k, False, seed=11, interior_pads=True)
    assert (mask[0, :-1].eq(0) & mask[0, 1:].ne(0)).any()  # a pad followed by a real token
    out = canon_forward(x, w, mask, seg, torch.float32, residual)
    ref = canon_conv_reference(x, w, mask, seg, torch.float32, residual)
    torch.testing.assert_close(out, ref, **FP32_TOL)


def test_row_size_guard():
    # In-row offsets are int32; a row of 2**31 elements must be refused, not overflow.
    T, D = 2**16, 2**15
    x = torch.zeros(1, 1, 1, device=DEVICE).expand(1, T, D)  # no 8 GB allocation
    w = torch.zeros(D, 3, device=DEVICE)
    mask = torch.ones(1, T, dtype=torch.long, device=DEVICE)
    with pytest.raises(ValueError, match="2\\*\\*31"):
        canon_forward(x, w, mask, None, torch.float32, False)
    with pytest.raises(ValueError, match="2\\*\\*31"):
        canon_backward(x, x, w, mask, None, False)


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
    dx, dw = canon_backward(dy, x, w, mask, seg, residual)
    assert dx.dtype == x.dtype and dx.shape == x.shape
    assert dw.dtype == w.dtype and dw.shape == w.shape
    return dx, dw, xr.grad, wr.grad, dy


def _assert_pad_grads(dx, dy, mask, residual):
    """At pads `dx` is exactly `dy` (residual) or `0`: the conv never reads a pad."""
    pad = mask.eq(0)
    expect = dy[pad].to(dx.dtype) if residual else torch.zeros_like(dx[pad])
    assert torch.equal(dx[pad], expect)


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


@pytest.mark.parametrize("packed", [True, False])
def test_backward_degenerate_rows(packed):
    # All-pad row: no gradient into the conv, so dx is 0 (or dy) and dw is exactly 0.
    x, w, _, _ = _case(1, 5, 64, 3, packed, seed=1)
    mask = torch.zeros(1, 5, dtype=torch.long, device=DEVICE)
    seg = torch.zeros_like(mask) if packed else None
    for residual in (True, False):
        dy = torch.randn_like(x)
        dx, dw = canon_backward(dy, x, w, mask, seg, residual)
        assert torch.equal(dx, dy if residual else torch.zeros_like(x))
        assert torch.equal(dw, torch.zeros_like(w))

    # An all-pad row adds nothing to dw: same dw with and without it in the batch.
    x, w, mask, seg = _case(3, 37, 64, 7, packed, seed=2)
    assert mask[-1].eq(0).all()
    dy = torch.randn_like(x)
    dw_all = canon_backward(dy, x, w, mask, seg, False)[1]
    dw_two = canon_backward(dy[:2], x[:2], w, mask[:2], None if seg is None else seg[:2], False)[1]
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


def test_backward_deterministic():
    x, w, mask, seg = _case(4, 300, 100, 7, True, seed=3)
    dy = torch.randn_like(x)
    dx1, dw1 = canon_backward(dy, x, w, mask, seg, True)
    dx2, dw2 = canon_backward(dy, x, w, mask, seg, True)
    assert torch.equal(dw1, dw2)
    assert torch.equal(dx1, dx2)
