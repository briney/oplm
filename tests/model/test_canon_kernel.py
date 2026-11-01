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

from oplm.model.canon_kernel import BLOCK_T, canon_forward  # noqa: E402

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
):
    """`(x, weight, attention_mask, segment_ids)`; `segment_ids` is None unless `packed`.

    Rows get random segment lengths and trailing pads; with `all_pad_row` and `B > 1` the last row
    is all pad. Pad positions of `x` hold large garbage so a leak shows up.
    """
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(B, T, D, generator=gen)
    weight = torch.randn(D, k, generator=gen)
    seg = torch.stack([_segments_from_lengths(_random_lengths(T, gen), T) for _ in range(B)])
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


def test_forward_segment_boundaries_on_tile_edges():
    T = 2 * BLOCK_T + 3
    gen = torch.Generator().manual_seed(0)
    edges = [BLOCK_T - 1, BLOCK_T, BLOCK_T + 1]
    for layout in range(20):
        # One boundary on a tile edge, a few random ones elsewhere, random trailing pad.
        cuts = {edges[layout % 3]}
        cuts |= {int(c) for c in torch.randint(1, T, (3,), generator=gen)}
        end = int(torch.randint(max(cuts) + 1, T + 1, (1,), generator=gen))
        bounds = [0, *sorted(cuts), end]
        seg = _segments_from_lengths(torch.tensor(bounds).diff().tolist(), T)
        k = (2, 3, 4, 7)[layout % 4]
        x = torch.randn(1, T, 64, generator=gen).to(DEVICE)
        w = torch.randn(64, k, generator=gen).to(DEVICE)
        mask = seg.ne(0).long().unsqueeze(0).to(DEVICE)
        seg = seg.unsqueeze(0).to(DEVICE)
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
