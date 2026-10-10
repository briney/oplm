"""Attention primitives: dense oracle vs the transcribed ESMFold2 formulations, masked-row
finiteness, gradients, flex/dense forward parity on CPU, and GPU gradient parity."""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from oplm.fold.attention import (
    ensure_flex_recompile_limit,
    pair_bias_block_mask,
    pair_biased_attention,
    resolve_attention_backend,
    sliding_window_attention,
    sliding_window_block_mask,
)


def _qkv(
    b: int = 2, h: int = 2, n: int = 48, d: int = 16, seed: int = 0
) -> tuple[torch.Tensor, ...]:
    g = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(b, h, n, d, generator=g) for _ in range(3))


def _upstream_pair_bias(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    key_mask: torch.Tensor | None,
) -> torch.Tensor:
    """ESMFold2 ``AttentionPairBias`` reference branch in (B, N, H, d) layout, transcribed."""
    qh, kh, vh = (t.transpose(1, 2) for t in (q, k, v))  # (B, N, H, d)
    logits = torch.einsum("... i h d, ... j h d -> ... i j h", qh, kh) * q.shape[-1] ** -0.5
    logits = logits + bias.permute(0, 2, 3, 1)  # (B, i, j, H)
    if key_mask is not None:
        min_val = torch.finfo(logits.dtype).min
        logits = logits + torch.where(key_mask.bool()[:, None, :, None], 0.0, min_val)
    attn = torch.softmax(logits, dim=-2)
    return torch.einsum("... i j h, ... j h d -> ... i h d", attn, vh).transpose(1, 2)


def _upstream_sliding_window(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, valid: torch.Tensor, half_window: int
) -> torch.Tensor:
    """ESMFold2 ``SWA3DRoPEAttention`` non-flash branch, transcribed."""
    n = q.shape[2]
    rank = torch.cumsum(valid, dim=1) - 1
    within = (rank.unsqueeze(2) - rank.unsqueeze(1)).abs() <= half_window
    allowed = within & valid.unsqueeze(1) & valid.unsqueeze(2)
    allowed |= torch.eye(n, dtype=torch.bool)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed.unsqueeze(1))
    return out * valid[:, None, :, None]


def test_backend_resolution() -> None:
    cpu = torch.zeros(1)
    assert resolve_attention_backend(cpu, "auto") == "dense"
    assert resolve_attention_backend(cpu, "flex") == "flex"
    assert resolve_attention_backend(cpu, "dense") == "dense"


def test_pair_biased_dense_matches_upstream_formulation() -> None:
    q, k, v = _qkv()
    bias = torch.randn(2, 2, 48, 48)
    key_mask = torch.ones(2, 48, dtype=torch.bool)
    key_mask[0, 40:] = False
    out = pair_biased_attention(q, k, v, bias, key_mask, backend="dense")
    torch.testing.assert_close(
        out, _upstream_pair_bias(q, k, v, bias, key_mask), rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        pair_biased_attention(q, k, v, bias, backend="dense"),
        _upstream_pair_bias(q, k, v, bias, None),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("backend", ["dense", "flex"])
def test_pair_biased_zero_valid_keys_row_is_zero_and_finite(backend: str) -> None:
    q, k, v = _qkv()
    bias = torch.randn(2, 2, 48, 48)
    key_mask = torch.ones(2, 48, dtype=torch.bool)
    key_mask[1] = False  # batch row 1: no valid key at all
    out = pair_biased_attention(q, k, v, bias, key_mask, backend=backend)
    assert torch.isfinite(out).all()
    assert out[1].abs().max() == 0
    assert out[0].abs().max() > 0


def test_pair_biased_dense_gradients_reach_bias_q_k_v() -> None:
    q, k, v = (t.requires_grad_(True) for t in _qkv())
    bias = torch.randn(2, 2, 48, 48, requires_grad=True)
    key_mask = torch.ones(2, 48, dtype=torch.bool)
    key_mask[0, 30:] = False
    pair_biased_attention(q, k, v, bias, key_mask, backend="dense").square().sum().backward()
    for t in (q, k, v, bias):
        assert t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs().sum() > 0
    assert bias.grad[0, :, :, 30:].abs().max() == 0  # masked keys receive no bias gradient


def test_flex_forward_matches_dense_on_cpu() -> None:
    q, k, v = _qkv(n=64)
    bias = torch.randn(2, 2, 64, 64)
    key_mask = torch.ones(2, 64, dtype=torch.bool)
    key_mask[0, 50:] = False
    with torch.no_grad():  # FlexAttention has no CPU backward; forward parity only here
        torch.testing.assert_close(
            pair_biased_attention(q, k, v, bias, key_mask, backend="flex"),
            pair_biased_attention(q, k, v, bias, key_mask, backend="dense"),
            rtol=1e-4,
            atol=1e-4,
        )
        valid = torch.ones(2, 64, dtype=torch.bool)
        valid[0, 40:] = False
        valid[1, 10:20] = False
        torch.testing.assert_close(
            sliding_window_attention(q, k, v, valid, 4, backend="flex"),
            sliding_window_attention(q, k, v, valid, 4, backend="dense"),
            rtol=1e-4,
            atol=1e-4,
        )


def test_sliding_window_dense_matches_upstream_and_zeroes_invalid() -> None:
    q, k, v = _qkv()
    valid = torch.ones(2, 48, dtype=torch.bool)
    valid[0, 36:] = False
    valid[1, 5:9] = False
    out = sliding_window_attention(q, k, v, valid, 3, backend="dense")
    torch.testing.assert_close(
        out, _upstream_sliding_window(q, k, v, valid, 3), rtol=1e-5, atol=1e-5
    )
    assert out[0, :, 36:].abs().max() == 0
    assert out[1, :, 5:9].abs().max() == 0


def test_sliding_window_counts_in_valid_atom_rank_not_padded_index() -> None:
    """Invalid atoms between two valid ones do not consume window budget.

    The window is measured in reference (valid-atom rank) space, not padded index space.
    """
    q, k, v = _qkv(n=16)
    valid = torch.zeros(2, 16, dtype=torch.bool)
    valid[:, 0] = True
    valid[:, 10] = True  # ranks 0 and 1 -> within half_window=1 despite index gap 10
    base = sliding_window_attention(q, k, v, valid, 1, backend="dense")
    v2 = v.clone()
    v2[:, :, 10] += 1.0
    moved = sliding_window_attention(q, k, v2, valid, 1, backend="dense")
    assert not torch.allclose(base[:, :, 0], moved[:, :, 0])  # atom 0 attends to atom 10


# --- GPU gradient parity (slow; skipped without CUDA) ----------------------------------

_requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
_ERR_MULT = 4.0
_ERR_FLOOR = 1e-2


@pytest.mark.slow
@_requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("n", [256, 512])
def test_flex_matches_dense_forward_and_backward_on_gpu(dtype: torch.dtype, n: int) -> None:
    torch.manual_seed(0)
    q32, k32, v32 = (torch.randn(2, 4, n, 32, device="cuda") for _ in range(3))
    bias32 = torch.randn(2, 4, n, n, device="cuda")
    key_mask = torch.ones(2, n, dtype=torch.bool, device="cuda")
    key_mask[0, n - 37 :] = False
    valid = key_mask.clone()
    valid[1, 100:140] = False

    def run(backend: str, dt: torch.dtype) -> list[torch.Tensor]:
        tensors = [t.detach().to(dt).requires_grad_(True) for t in (q32, k32, v32, bias32)]
        q, k, v, bias = tensors
        out = pair_biased_attention(q, k, v, bias, key_mask, backend=backend)
        out.float().square().sum().backward()
        grads = [t.grad.float() for t in tensors]
        tensors_sw = [t.detach().to(dt).requires_grad_(True) for t in (q32, k32, v32)]
        out_sw = sliding_window_attention(*tensors_sw, valid, 64, backend=backend)
        out_sw.float().square().sum().backward()
        return [out.float(), *grads, out_sw.float(), *[t.grad.float() for t in tensors_sw]]

    expected = run("dense", torch.float32)
    baseline = run("dense", dtype)
    flex = run("flex", dtype)
    for name, e, b, f in zip(
        ["out", "dq", "dk", "dv", "dbias", "out_sw", "dq_sw", "dk_sw", "dv_sw"],
        expected,
        baseline,
        flex,
        strict=True,
    ):
        base_err = (b - e).abs().max().item()
        err = (f - e).abs().max().item()
        assert err <= _ERR_MULT * base_err + _ERR_FLOOR, (name, err, base_err)


# --- precomputed block masks (milestone 1) ----------------------------------------------


def test_precomputed_block_masks_match_inline_construction_on_cpu_forward() -> None:
    q, k, v = _qkv(n=64)
    bias = torch.randn(2, 2, 64, 64)
    key_mask = torch.ones(2, 64, dtype=torch.bool)
    key_mask[0, 50:] = False
    valid = torch.ones(2, 64, dtype=torch.bool)
    valid[1, 10:20] = False
    with torch.no_grad():
        inline = pair_biased_attention(q, k, v, bias, key_mask, backend="flex")
        pre = pair_biased_attention(
            q, k, v, bias, key_mask, backend="flex", block_mask=pair_bias_block_mask(key_mask)
        )
        torch.testing.assert_close(pre, inline)
        inline_sw = sliding_window_attention(q, k, v, valid, 4, backend="flex")
        pre_sw = sliding_window_attention(
            q, k, v, valid, 4, backend="flex", block_mask=sliding_window_block_mask(valid, 4)
        )
        torch.testing.assert_close(pre_sw, inline_sw)


def test_block_mask_is_ignored_on_the_dense_path() -> None:
    q, k, v = _qkv()
    bias = torch.randn(2, 2, 48, 48)
    key_mask = torch.ones(2, 48, dtype=torch.bool)
    dense = pair_biased_attention(q, k, v, bias, key_mask, backend="dense")
    with_mask = pair_biased_attention(
        q, k, v, bias, key_mask, backend="dense", block_mask=pair_bias_block_mask(key_mask)
    )
    torch.testing.assert_close(dense, with_mask)


def test_ensure_flex_recompile_limit_only_raises() -> None:
    from torch import _dynamo

    before = _dynamo.config.recompile_limit
    ensure_flex_recompile_limit(before + 8)
    assert _dynamo.config.recompile_limit == before + 8
    ensure_flex_recompile_limit(1)
    assert _dynamo.config.recompile_limit == before + 8
    _dynamo.config.recompile_limit = before
