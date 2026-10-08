"""Triangle multiplicative update: staged reference and the ``TriangleMultiplication`` module.

Ported from Biohub's ESMFold2 ``TriangleMultiplicativeBlock``
(https://github.com/Biohub/esm, ``esm/models/esmfold2/layers.py``, Apache-2.0;
see THIRD_PARTY_NOTICES.md). Modifications: the forward is split into three
explicit stages -- local pre-projection, triangular contraction, local
post-projection -- so the contraction is the only cross-row/column operation
(design §5.2, §5.6); the contraction accumulates in fp32 chunked over output
rows; LayerNorms compute in fp32 and cast back (the ``OplmLayerNorm`` contract);
kernel dispatch is explicit instead of a try/except fallback.

Parameter names and shapes match upstream so released ESMFold2 weights load
without remapping: ``norm_start``/``norm_mix`` (LayerNorm, eps 1e-5),
``proj_bundle`` (D -> 4D, no bias; rows [0:2D] signal, [2D:4D] gate logits),
``proj_emit`` (D -> D, no bias), ``proj_gate`` (D -> D, no bias). The functional
API takes the eight tensors in cuEquivariance's order so the fused kernel is a
straight pass-through: ``p_in_weight = proj_bundle.weight[:2D]``,
``g_in_weight = proj_bundle.weight[2D:]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

__all__ = [
    "TRIMUL_EPS",
    "Backend",
    "Direction",
    "TriMulPath",
    "TriangleMultiplication",
    "cueq_available",
    "fused_trimul",
    "resolve_trimul_path",
    "trimul_contract",
    "trimul_mixed",
    "trimul_post",
    "trimul_pre",
    "trimul_reference",
]

Direction = Literal["outgoing", "incoming"]
TRIMUL_EPS = 1e-5
_EINSUM: dict[str, str] = {"outgoing": "bikd,bjkd->bijd", "incoming": "bkid,bkjd->bijd"}

Backend = Literal["auto", "fused", "fused_forward_reference_backward", "reference"]
TriMulPath = Literal["fused_autograd", "fused_forward_reference_backward", "reference"]
_VALID_BACKENDS: tuple[str, ...] = (
    "auto",
    "fused",
    "fused_forward_reference_backward",
    "reference",
)
_FUSED_DTYPES = (torch.float32, torch.bfloat16, torch.float16)

try:
    from cuequivariance_torch import (  # ty: ignore[unresolved-import]  # optional `fold` extra
        triangle_multiplicative_update as _cueq_trimul,
    )
except ImportError:  # the `fold` extra is optional; CPU/test environments never have it
    _cueq_trimul = None


def cueq_available() -> bool:
    """Whether ``cuequivariance_torch.triangle_multiplicative_update`` is importable."""
    return _cueq_trimul is not None


def resolve_trimul_path(z: Tensor, *, needs_grad: bool, backend: Backend) -> TriMulPath:
    """Pick the execution path for one call (design §5.2).

    ``"reference"`` whenever cuEquivariance is absent, ``z`` is not on CUDA, the width is
    not a multiple of 32, the dtype is unsupported, or the backend says so. Otherwise a
    call with no gradient requirement uses the fused forward directly; a gradient call
    uses the library's autograd unless ``backend`` selects the mixed path. ``"auto"``
    trusts the library's autograd until ``oplm fold bench-kernels`` says otherwise --
    the recorded choice then goes into the stage config.
    """
    if backend == "reference" or not cueq_available() or not z.is_cuda:
        return "reference"
    if z.shape[-1] % 32 or z.dtype not in _FUSED_DTYPES:
        return "reference"
    if not needs_grad or backend == "fused":
        return "fused_autograd"
    if backend == "fused_forward_reference_backward":
        return "fused_forward_reference_backward"
    return "fused_autograd"


def fused_trimul(
    z: Tensor, direction: Direction, mask: Tensor | None, *weights: Tensor, eps: float = TRIMUL_EPS
) -> Tensor:
    """cuEquivariance ``triangle_multiplicative_update`` with the port's weight mapping."""
    if _cueq_trimul is None:
        raise RuntimeError("cuequivariance_torch is not installed; pip install 'oplm[fold]'")
    (
        norm_in_weight,
        norm_in_bias,
        p_in_weight,
        g_in_weight,
        norm_out_weight,
        norm_out_bias,
        p_out_weight,
        g_out_weight,
    ) = weights
    return _cueq_trimul(
        z,
        direction=direction,
        mask=None if mask is None else mask.to(z.dtype),
        norm_in_weight=norm_in_weight,
        norm_in_bias=norm_in_bias,
        p_in_weight=p_in_weight,
        g_in_weight=g_in_weight,
        norm_out_weight=norm_out_weight,
        norm_out_bias=norm_out_bias,
        p_out_weight=p_out_weight,
        g_out_weight=g_out_weight,
        eps=eps,
    )


class _FusedForwardReferenceBackward(torch.autograd.Function):
    """Fused inference kernel forward; the staged reference recomputed in backward.

    One recomputation boundary: the forward saves only its inputs (pair block, mask,
    weights -- what an activation checkpoint saves anyway); the backward rebuilds the
    staged reference graph under ``enable_grad`` and backpropagates through it, so
    reference activations exist only inside the backward call. Composes with
    ``torch.utils.checkpoint``: the outer checkpoint re-runs the cheap fused forward,
    then this backward runs the reference exactly once.
    """

    @staticmethod
    def forward(
        ctx: Any,
        z: Tensor,
        mask: Tensor | None,
        direction: Direction,
        eps: float,
        chunk_size: int | None,
        fused_fn: Callable[..., Tensor],
        *weights: Tensor,
    ) -> Tensor:
        ctx.direction, ctx.eps, ctx.chunk_size = direction, eps, chunk_size
        ctx.mask = mask  # not differentiated; may be None
        ctx.save_for_backward(z, *weights)
        with torch.no_grad():
            return fused_fn(z, direction, mask, *weights, eps=eps)

    @staticmethod
    def backward(  # ty: ignore[invalid-method-override]  # Function.backward is *grad_outputs: Any
        ctx: Any, grad_out: Tensor
    ) -> tuple[Tensor | None, ...]:
        z, *weights = ctx.saved_tensors
        needs = ctx.needs_input_grad
        with torch.enable_grad():
            z_live = z.detach().requires_grad_(needs[0])
            weights_live = [
                w.detach().requires_grad_(need) for w, need in zip(weights, needs[6:], strict=True)
            ]
            out = trimul_reference(
                z_live,
                ctx.direction,
                ctx.mask,
                *weights_live,
                eps=ctx.eps,
                chunk_size=ctx.chunk_size,
            )
            inputs = [t for t in (z_live, *weights_live) if t.requires_grad]
            grads = iter(torch.autograd.grad(out, inputs, grad_out, allow_unused=True))
        dz = next(grads) if z_live.requires_grad else None
        dweights = [next(grads) if w.requires_grad else None for w in weights_live]
        return (dz, None, None, None, None, None, *dweights)


def trimul_mixed(
    z: Tensor,
    direction: Direction,
    mask: Tensor | None,
    *weights: Tensor,
    eps: float = TRIMUL_EPS,
    chunk_size: int | None = 64,
    fused_fn: Callable[..., Tensor] = fused_trimul,
) -> Tensor:
    """Fused forward, reference backward. ``fused_fn`` is injectable so CPU tests can drive it."""
    return _FusedForwardReferenceBackward.apply(
        z, mask, direction, eps, chunk_size, fused_fn, *weights
    )


def _layer_norm_fp32(x: Tensor, weight: Tensor, bias: Tensor, eps: float) -> Tensor:
    """LayerNorm with fp32 internals, cast back to ``x``'s dtype."""
    return F.layer_norm(x.float(), (x.shape[-1],), weight.float(), bias.float(), eps).to(x.dtype)


def trimul_pre(
    z: Tensor,
    mask: Tensor | None,
    norm_in_weight: Tensor,
    norm_in_bias: Tensor,
    p_in_weight: Tensor,
    g_in_weight: Tensor,
    *,
    eps: float = TRIMUL_EPS,
) -> tuple[Tensor, Tensor, Tensor]:
    """Stage 1 (pointwise): normalize, project, gate, mask.

    Args:
        z: Pair block ``(B, I, J, D)``.
        mask: Pair validity ``(B, I, J)`` (bool or float); ``None`` means all valid.
        norm_in_weight: ``norm_start`` affine weight ``(D,)``.
        norm_in_bias: ``norm_start`` affine bias ``(D,)``.
        p_in_weight: Signal projection ``(2D, D)`` (``proj_bundle.weight[:2D]``).
        g_in_weight: Gate-logit projection ``(2D, D)`` (``proj_bundle.weight[2D:]``).
        eps: LayerNorm epsilon.

    Returns:
        ``(left, right, zn)``: the two contraction operands ``(B, I, J, D)`` in ``z``'s
        dtype and the normalized input the output gate is computed from.
    """
    zn = _layer_norm_fp32(z, norm_in_weight, norm_in_bias, eps)
    routed = F.linear(zn, p_in_weight) * torch.sigmoid(F.linear(zn, g_in_weight))
    if mask is not None:
        routed = routed * mask.to(routed.dtype).unsqueeze(-1)
    left, right = routed.chunk(2, dim=-1)
    return left, right, zn


def trimul_contract(
    left: Tensor, right: Tensor, direction: Direction, *, chunk_size: int | None = 64
) -> Tensor:
    """Stage 2, the only cross-row/column op: the triangular contraction in fp32.

    ``outgoing``: ``out[b,i,j] = sum_k left[b,i,k] * right[b,j,k]``;
    ``incoming``: ``out[b,i,j] = sum_k left[b,k,i] * right[b,k,j]``.
    Output rows are chunked so one fp32 chunk of ``left`` at a time is live next to
    the fp32 copy of ``right``. The result is fp32.
    """
    # ponytail: keeps one full fp32 copy of `right` (4 GiB at L=2048, D=256); the fused
    # cuEquivariance path is the production kernel, this is the oracle/fallback.
    equation = _EINSUM[direction]
    right32 = right.float()
    n = left.shape[1]
    # Spec §5.4: accumulate in fp32. einsum lowers to bmm, which autocast would re-cast to
    # the low-precision dtype, so autocast is disabled locally around the contraction.
    with torch.autocast(device_type=left.device.type, enabled=False):
        if chunk_size is None or n <= chunk_size:
            return torch.einsum(equation, left.float(), right32)
        chunks = []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            left_chunk = left[:, start:end] if direction == "outgoing" else left[:, :, start:end]
            chunks.append(torch.einsum(equation, left_chunk.float(), right32))
        return torch.cat(chunks, dim=1)


def trimul_post(
    contracted: Tensor,
    zn: Tensor,
    norm_out_weight: Tensor,
    norm_out_bias: Tensor,
    p_out_weight: Tensor,
    g_out_weight: Tensor,
    *,
    eps: float = TRIMUL_EPS,
) -> Tensor:
    """Stage 3 (pointwise): normalize the contraction in fp32, project out, apply the gate."""
    normed = _layer_norm_fp32(contracted, norm_out_weight, norm_out_bias, eps)
    mixed = F.linear(normed.to(zn.dtype), p_out_weight)
    return mixed * torch.sigmoid(F.linear(zn, g_out_weight))


def trimul_reference(
    z: Tensor,
    direction: Direction,
    mask: Tensor | None,
    norm_in_weight: Tensor,
    norm_in_bias: Tensor,
    p_in_weight: Tensor,
    g_in_weight: Tensor,
    norm_out_weight: Tensor,
    norm_out_bias: Tensor,
    p_out_weight: Tensor,
    g_out_weight: Tensor,
    *,
    eps: float = TRIMUL_EPS,
    chunk_size: int | None = 64,
) -> Tensor:
    """The differentiable staged reference (CPU path, backward oracle, fused-path oracle)."""
    left, right, zn = trimul_pre(
        z, mask, norm_in_weight, norm_in_bias, p_in_weight, g_in_weight, eps=eps
    )
    contracted = trimul_contract(left, right, direction, chunk_size=chunk_size)
    return trimul_post(
        contracted, zn, norm_out_weight, norm_out_bias, p_out_weight, g_out_weight, eps=eps
    )


class TriangleMultiplication(nn.Module):
    """One triangle multiplicative update, returning the delta.

    The residual add and row-shared dropout live in the pair-update block that
    owns this module (milestone 1), exactly as upstream's ``PairUpdateBlock``.
    """

    def __init__(
        self,
        width: int,
        direction: Direction,
        *,
        eps: float = TRIMUL_EPS,
        chunk_size: int | None = 64,
        backend: Backend = "auto",
    ) -> None:
        super().__init__()
        if direction not in _EINSUM:
            raise ValueError(f"direction must be 'outgoing' or 'incoming', got {direction!r}")
        if backend not in _VALID_BACKENDS:
            raise ValueError(f"backend must be one of {_VALID_BACKENDS}, got {backend!r}")
        self.width = width
        self.direction: Direction = direction
        self.eps = eps
        self.chunk_size = chunk_size
        self.backend: Backend = backend
        # Plain nn.LayerNorm holders keep the upstream state-dict exactly; the fp32
        # math lives in _layer_norm_fp32, which is OplmLayerNorm's contract.
        self.norm_start = nn.LayerNorm(width, eps=eps)
        self.norm_mix = nn.LayerNorm(width, eps=eps)
        self.proj_bundle = nn.Linear(width, 4 * width, bias=False)
        self.proj_emit = nn.Linear(width, width, bias=False)
        self.proj_gate = nn.Linear(width, width, bias=False)

    def kernel_weights(self) -> tuple[Tensor, ...]:
        """The eight functional/cuEquivariance operands, split ``proj_bundle``."""
        bundle = self.proj_bundle.weight
        return (
            self.norm_start.weight,
            self.norm_start.bias,
            bundle[: 2 * self.width],
            bundle[2 * self.width :],
            self.norm_mix.weight,
            self.norm_mix.bias,
            self.proj_emit.weight,
            self.proj_gate.weight,
        )

    def forward(self, z: Tensor, mask: Tensor | None = None) -> Tensor:
        weights = self.kernel_weights()
        needs_grad = torch.is_grad_enabled() and (
            z.requires_grad or any(w.requires_grad for w in weights)
        )
        path = resolve_trimul_path(z, needs_grad=needs_grad, backend=self.backend)
        if path == "reference":
            return trimul_reference(
                z, self.direction, mask, *weights, eps=self.eps, chunk_size=self.chunk_size
            )
        if path == "fused_autograd":
            return fused_trimul(z, self.direction, mask, *weights, eps=self.eps)
        return trimul_mixed(
            z, self.direction, mask, *weights, eps=self.eps, chunk_size=self.chunk_size
        )
