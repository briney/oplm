"""Triangle multiplication: staged reference vs the transcribed ESMFold2 equation, chunking,
masks, gradients, stage locality, and state-dict parity (docs/FOLD.md "Triangle multiplication")."""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from oplm.fold.trimul import (
    TriangleMultiplication,
    trimul_contract,
    trimul_post,
    trimul_pre,
    trimul_reference,
)

DIRECTIONS = ("outgoing", "incoming")
_EINSUM = {"outgoing": "bikd,bjkd->bijd", "incoming": "bkid,bkjd->bijd"}


def _upstream_forward(
    z: torch.Tensor, mask: torch.Tensor | None, m: TriangleMultiplication
) -> torch.Tensor:
    """ESMFold2 ``TriangleMultiplicativeBlock.forward``, unchunked, transcribed verbatim."""
    if mask is None:
        mask = z.new_ones(z.shape[:-1])
    zn = F.layer_norm(z, (m.width,), m.norm_start.weight, m.norm_start.bias, 1e-5)
    bundled = F.linear(zn, m.proj_bundle.weight)
    signal, gate_logits = bundled.split(2 * m.width, dim=-1)
    routed = signal * torch.sigmoid(gate_logits)
    routed = routed * mask.unsqueeze(-1)
    left, right = routed.float().chunk(2, dim=-1)
    contracted = torch.einsum(_EINSUM[m.direction], left, right)
    mixed = F.linear(
        F.layer_norm(contracted, (m.width,), m.norm_mix.weight, m.norm_mix.bias, 1e-5),
        m.proj_emit.weight,
    )
    return mixed * torch.sigmoid(F.linear(zn, m.proj_gate.weight))


def _inputs(n: int = 7, width: int = 32, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(2, n, n, width, generator=g)
    mask = (torch.rand(2, n, n, generator=g) > 0.3).float()
    return z, mask


def _module(direction: str, width: int = 32, seed: int = 0) -> TriangleMultiplication:
    torch.manual_seed(seed)
    m = TriangleMultiplication(width, direction)
    with torch.no_grad():  # non-trivial norms so a wrong eps/affine placement shows up
        for norm in (m.norm_start, m.norm_mix):
            norm.weight.uniform_(0.5, 1.5)
            norm.bias.uniform_(-0.5, 0.5)
    return m


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_matches_upstream_equation(direction: str) -> None:
    m = _module(direction)
    z, mask = _inputs()
    torch.testing.assert_close(m(z, mask), _upstream_forward(z, mask, m), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(m(z), _upstream_forward(z, None, m), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_chunked_equals_unchunked(direction: str) -> None:
    m = _module(direction)
    z, mask = _inputs(n=5)  # 5 rows, chunk 2 -> an uneven tail chunk
    m.chunk_size = 2
    chunked = m(z, mask)
    m.chunk_size = None
    torch.testing.assert_close(chunked, m(z, mask), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_masked_row_contributes_nothing(direction: str) -> None:
    m = _module(direction)
    z, _ = _inputs(n=6)
    mask = torch.ones(2, 6, 6)
    mask[:, 2, :] = 0.0  # row 2 fully masked
    left, right, _zn = trimul_pre(z, mask, *m.kernel_weights()[:4], eps=m.eps)
    contracted = trimul_contract(left, right, m.direction, chunk_size=2)
    if direction == "outgoing":  # out[i, j] = sum_k left[i, k] right[j, k]
        assert contracted[:, 2].abs().max() == 0 and contracted[:, :, 2].abs().max() == 0
    else:  # out[i, j] = sum_k left[k, i] right[k, j]: row 2 drops out of the k-sum
        unmasked = trimul_contract(
            *trimul_pre(z, None, *m.kernel_weights()[:4], eps=m.eps)[:2], m.direction, chunk_size=2
        )
        assert not torch.allclose(contracted, unmasked)
    assert torch.isfinite(m(z, mask)).all()


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_gradients_reach_input_and_every_parameter(direction: str) -> None:
    m = _module(direction)
    z, mask = _inputs()
    z.requires_grad_(True)
    m(z, mask).square().sum().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name


def test_pre_and_post_stages_are_pointwise_in_rows_and_columns() -> None:
    m = _module("outgoing")
    z, mask = _inputs(n=8)
    w = m.kernel_weights()
    full = trimul_pre(z, mask, *w[:4], eps=m.eps)
    block = trimul_pre(z[:, 2:5, 1:7], mask[:, 2:5, 1:7], *w[:4], eps=m.eps)
    for full_t, block_t in zip(full, block, strict=True):
        torch.testing.assert_close(block_t, full_t[:, 2:5, 1:7])
    contracted = trimul_contract(full[0], full[1], "outgoing", chunk_size=None)
    out_full = trimul_post(contracted, full[2], *w[4:], eps=m.eps)
    out_block = trimul_post(contracted[:, 2:5, 1:7], full[2][:, 2:5, 1:7], *w[4:], eps=m.eps)
    torch.testing.assert_close(out_block, out_full[:, 2:5, 1:7])


def test_functional_reference_equals_module() -> None:
    m = _module("incoming")
    z, mask = _inputs()
    torch.testing.assert_close(
        trimul_reference(z, "incoming", mask, *m.kernel_weights(), eps=m.eps, chunk_size=3),
        m(z, mask),
    )


def test_incoming_and_outgoing_differ() -> None:
    z, mask = _inputs()
    out = _module("outgoing")(z, mask)
    inc = _module("incoming")(z, mask)
    assert not torch.allclose(out, inc)


def test_state_dict_matches_upstream_names_and_shapes() -> None:
    m = TriangleMultiplication(64, "outgoing")
    shapes = {k: tuple(v.shape) for k, v in m.state_dict().items()}
    assert shapes == {
        "norm_start.weight": (64,),
        "norm_start.bias": (64,),
        "norm_mix.weight": (64,),
        "norm_mix.bias": (64,),
        "proj_bundle.weight": (256, 64),
        "proj_emit.weight": (64, 64),
        "proj_gate.weight": (64, 64),
    }


def test_rejects_unknown_direction() -> None:
    with pytest.raises(ValueError, match="direction"):
        TriangleMultiplication(32, "sideways")  # type: ignore[arg-type]
