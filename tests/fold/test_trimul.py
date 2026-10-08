"""Triangle multiplication: staged reference vs the transcribed ESMFold2 equation, chunking,
masks, gradients, stage locality, and state-dict parity (docs/FOLD.md "Triangle multiplication")."""

from __future__ import annotations

import copy
from typing import Any

import pytest
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from oplm.fold import trimul as trimul_module
from oplm.fold.trimul import (
    TriangleMultiplication,
    cueq_available,
    resolve_trimul_path,
    trimul_contract,
    trimul_mixed,
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


def test_contraction_stays_fp32_under_autocast() -> None:
    """Autocast must not demote the fp32 contraction (spec §5.4); the module output keeps the pair dtype."""
    m = _module("outgoing")
    z, mask = _inputs(n=5)
    left, right, _zn = trimul_pre(z, mask, *m.kernel_weights()[:4], eps=m.eps)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert trimul_contract(left, right, "outgoing", chunk_size=None).dtype == torch.float32
        assert trimul_contract(left, right, "outgoing", chunk_size=2).dtype == torch.float32
        out = m(z, mask)
    assert torch.isfinite(out).all()
    m_bf16 = copy.deepcopy(m).bfloat16()
    out_bf16 = m_bf16(z.bfloat16(), mask)
    assert out_bf16.dtype == torch.bfloat16 and torch.isfinite(out_bf16).all()


# --- dispatch and the mixed fused-forward/reference-backward path ---------------------


def _grads(m: TriangleMultiplication, z: torch.Tensor) -> dict[str, torch.Tensor]:
    grads = {name: p.grad.clone() for name, p in m.named_parameters() if p.grad is not None}
    assert z.grad is not None
    grads["z"] = z.grad.clone()
    m.zero_grad()
    z.grad = None
    return grads


def test_resolve_path_is_reference_on_cpu_regardless_of_backend() -> None:
    z = torch.randn(1, 4, 4, 32)
    for backend in ("auto", "fused", "fused_forward_reference_backward", "reference"):
        assert resolve_trimul_path(z, needs_grad=True, backend=backend) == "reference"
        assert resolve_trimul_path(z, needs_grad=False, backend=backend) == "reference"
    assert isinstance(cueq_available(), bool)


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_mixed_path_matches_reference_gradients(direction: str) -> None:
    m = _module(direction)
    z, mask = _inputs(n=6)
    z.requires_grad_(True)
    calls = {"fused": 0}

    def fake_fused(
        z_: torch.Tensor,
        direction_: str,
        mask_: torch.Tensor | None,
        *weights: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        calls["fused"] += 1
        assert not torch.is_grad_enabled()  # the fused forward never builds a graph
        return trimul_reference(z_, direction_, mask_, *weights, eps=eps, chunk_size=None)

    out = trimul_mixed(
        z, direction, mask, *m.kernel_weights(), eps=m.eps, chunk_size=2, fused_fn=fake_fused
    )
    out.square().sum().backward()
    mixed = _grads(m, z)
    assert calls["fused"] == 1

    m(z, mask).square().sum().backward()
    reference = _grads(m, z)
    assert mixed.keys() == reference.keys()
    for name in reference:
        torch.testing.assert_close(mixed[name], reference[name], rtol=1e-5, atol=1e-6)


def test_mixed_path_under_checkpoint_recomputes_once(monkeypatch: pytest.MonkeyPatch) -> None:
    m = _module("outgoing")
    z, mask = _inputs(n=6)
    z.requires_grad_(True)
    calls = {"fused": 0, "reference": 0}
    real_reference = trimul_module.trimul_reference

    def counting_reference(*args: Any, **kwargs: Any) -> torch.Tensor:
        calls["reference"] += 1
        return real_reference(*args, **kwargs)

    def fake_fused(
        z_: torch.Tensor,
        direction_: str,
        mask_: torch.Tensor | None,
        *weights: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        calls["fused"] += 1
        return real_reference(z_, direction_, mask_, *weights, eps=eps, chunk_size=None)

    monkeypatch.setattr(trimul_module, "trimul_reference", counting_reference)

    def block(z_: torch.Tensor) -> torch.Tensor:
        return trimul_mixed(
            z_, "outgoing", mask, *m.kernel_weights(), eps=m.eps, chunk_size=2, fused_fn=fake_fused
        )

    checkpoint(block, z, use_reentrant=False).square().sum().backward()
    assert calls == {"fused": 2, "reference": 1}  # forward + recompute; one backward pass
    mixed = _grads(m, z)

    monkeypatch.setattr(trimul_module, "trimul_reference", real_reference)
    m(z, mask).square().sum().backward()
    reference = _grads(m, z)
    for name in reference:
        torch.testing.assert_close(mixed[name], reference[name], rtol=1e-5, atol=1e-6)


def test_module_backend_knob_validates() -> None:
    with pytest.raises(ValueError, match="backend"):
        TriangleMultiplication(32, "outgoing", backend="triton")  # type: ignore[arg-type]
    assert TriangleMultiplication(32, "outgoing", backend="reference").backend == "reference"


# --- GPU parity (slow; skipped without CUDA + cuEquivariance) ---------------------------

_requires_cueq = pytest.mark.skipif(
    not (torch.cuda.is_available() and cueq_available()),
    reason="needs CUDA and the `fold` extra (cuequivariance-torch)",
)
# Tolerance policy (spec §9): the fused bf16 error against the fp32 reference may be at
# most 4x the bf16 *reference* error against the same fp32 oracle, plus a small floor.
_ERR_MULT = 4.0
_ERR_FLOOR = 1e-2


@pytest.mark.slow
@_requires_cueq
@pytest.mark.parametrize("width", [128, 256])
@pytest.mark.parametrize("length", [128, 512, 1024])
@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("backend", ["fused", "fused_forward_reference_backward"])
def test_fused_paths_match_fp32_reference_within_bf16_tolerance(
    width: int, length: int, direction: str, backend: str
) -> None:
    torch.manual_seed(0)
    ref32 = TriangleMultiplication(width, direction, backend="reference").cuda()
    z32 = torch.randn(1, length, length, width, device="cuda", requires_grad=True)
    mask = (torch.rand(1, length, length, device="cuda") > 0.2).float()
    ref32(z32, mask).float().square().mean().backward()
    expected_out = ref32(z32, mask).detach()
    expected = _grads(ref32, z32)

    def run(backend_: str) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """bf16 forward/backward on a copy of ref32 under ``backend_``; returns fp32 views."""
        m = copy.deepcopy(ref32).bfloat16()
        m.backend = backend_
        z = z32.detach().bfloat16().requires_grad_(True)
        out = m(z, mask)
        out.float().square().mean().backward()
        return out.detach().float(), {k: v.float() for k, v in _grads(m, z).items()}

    ref_bf16_out, ref_bf16 = run("reference")  # the bf16 *reference* sets the tolerance
    assert resolve_trimul_path(z32.bfloat16(), needs_grad=True, backend=backend) != "reference"
    fused_out, fused = run(backend)

    baseline = (ref_bf16_out - expected_out).abs().max().item()
    assert (fused_out - expected_out).abs().max().item() <= _ERR_MULT * baseline + _ERR_FLOOR
    for name in expected:
        baseline = (ref_bf16[name] - expected[name]).abs().max().item()
        err = (fused[name] - expected[name]).abs().max().item()
        assert err <= _ERR_MULT * baseline + _ERR_FLOOR, (name, err, baseline)
