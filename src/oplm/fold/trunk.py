"""Pair stack: SwiGLU transitions, pair-update blocks, and the recurrence ("parcae").

Ported from Biohub's ESMFold2 ``Transition``/``PairUpdateBlock``/``FoldingTrunk`` and the
``parcae_*`` recurrence of ``EsmFold2Model`` (esm/models/esmfold2/{layers,model}.py,
Apache-2.0; see THIRD_PARTY_NOTICES.md). Parameter names follow the released HF checkpoint:
``layers.N.{tri_mul_out,tri_mul_in,pair_transition.{norm,mlp.gate_up_proj,mlp.down_proj}}`` and
``parcae.{input_norm,log_delta,log_state_decay,input_matrix_continuous,out_proj,output_stack}``.
Modifications: the triangle updates are milestone 0's :class:`TriangleMultiplication`; the
recurrence takes the loop count, the number of gradient-carrying loops and the initial state
as arguments (spec §5.6; upstream always runs ``num_loops + 1`` loops without gradient);
row-shared dropout is a real module where upstream has ``DropoutResidual(0.0)``; the pair
stack supports activation checkpointing per block.
"""

from __future__ import annotations

import contextlib
import math
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from oplm.fold.trimul import Backend, TriangleMultiplication

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from oplm.fold.configuration_fold import FoldConfig

__all__ = [
    "GatedMLP",
    "PairStack",
    "PairUpdateBlock",
    "Recurrence",
    "RowSharedDropout",
    "Transition",
    "cuda_bf16_autocast",
    "fp32_autocast_off",
    "pair_stack_kwargs",
    "unloaded",
]


def cuda_bf16_autocast(enabled: bool) -> contextlib.AbstractContextManager[Any]:
    """bf16 autocast on CUDA when ``enabled``; a null context otherwise (never warns on CPU)."""
    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def fp32_autocast_off(device: torch.device) -> contextlib.AbstractContextManager[Any]:
    """Disable any outer autocast for fp32-only math (coordinates, alignment, logits; spec §5.4)."""
    return torch.autocast(device_type=device.type, enabled=False)


def unloaded(p: Tensor) -> bool:
    """True unless transformers already loaded ``p`` (its ``_is_hf_initialized`` flag is set)."""
    return not getattr(p, "_is_hf_initialized", False)


class GatedMLP(nn.Module):
    """``down_proj(silu(a) * b)``, ``a, b = gate_up_proj(x).chunk(2)`` (upstream SwiGLU order)."""

    def __init__(self, width: int, hidden: int) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(width, 2 * hidden, bias=False)
        self.down_proj = nn.Linear(hidden, width, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        a, b = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(a) * b)


class Transition(nn.Module):
    """Pre-norm gated MLP returning the delta; the caller adds the residual."""

    def __init__(self, width: int, expansion: int = 4, *, eps: float = 1e-5) -> None:
        super().__init__()
        if expansion < 1:
            raise ValueError(f"expansion must be >= 1; got {expansion!r}")
        self.norm = nn.LayerNorm(width, eps=eps)
        self.mlp = GatedMLP(width, expansion * width)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(self.norm(x))


class RowSharedDropout(nn.Module):
    """Dropout with one mask per (batch, column, channel), shared along the row axis.

    AF2-style "row-wise" dropout for the triangle updates; identity in eval or at ``p == 0``.
    """

    def __init__(self, p: float) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"dropout must be in [0, 1); got {p!r}")
        self.p = p

    def forward(self, delta: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return delta
        shape = (delta.shape[0], 1, *delta.shape[2:])
        keep = torch.rand(shape, device=delta.device) >= self.p
        return delta * keep.to(delta.dtype) / (1.0 - self.p)


class PairUpdateBlock(nn.Module):
    """``z += drop(tri_out(z)); z += drop(tri_in(z)); z += transition(z)`` (upstream block)."""

    def __init__(
        self,
        width: int,
        *,
        expansion: int = 4,
        dropout: float = 0.0,
        eps: float = 1e-5,
        chunk_size: int | None = 64,
        trimul_backend: Backend = "auto",
    ) -> None:
        super().__init__()
        self.tri_mul_out = TriangleMultiplication(
            width, "outgoing", eps=eps, chunk_size=chunk_size, backend=trimul_backend
        )
        self.tri_mul_in = TriangleMultiplication(
            width, "incoming", eps=eps, chunk_size=chunk_size, backend=trimul_backend
        )
        self.pair_transition = Transition(width, expansion, eps=eps)
        self.dropout = RowSharedDropout(dropout)

    def forward(self, z: Tensor, pair_mask: Tensor | None = None) -> Tensor:
        z = z + self.dropout(self.tri_mul_out(z, pair_mask))
        z = z + self.dropout(self.tri_mul_in(z, pair_mask))
        return z + self.pair_transition(z)


class PairStack(nn.Module):
    """A sequence of :class:`PairUpdateBlock` (checkpoint name ``layers.N``)."""

    def __init__(self, num_blocks: int, width: int, **block_kwargs: Any) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [PairUpdateBlock(width, **block_kwargs) for _ in range(num_blocks)]
        )
        self.gradient_checkpointing = False

    def forward(self, z: Tensor, pair_mask: Tensor | None = None) -> Tensor:
        for layer in self.layers:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                z = checkpoint(layer, z, pair_mask, use_reentrant=False)
            else:
                z = layer(z, pair_mask)
        return z


def pair_stack_kwargs(config: FoldConfig) -> dict[str, Any]:
    """Block kwargs shared by every pair stack built from ``config``."""
    return {
        "expansion": config.transition_expansion,
        "dropout": config.trunk_dropout,
        "eps": config.layer_norm_eps,
        "chunk_size": config.trimul_chunk_size,
        "trimul_backend": config.trimul_backend,
    }


def _inverse_softplus(y: float) -> float:
    return math.log(math.expm1(y))


class Recurrence(nn.Module):
    """``z_t = trunk(a ⊙ z_{t-1} + input_norm(inject_t) @ Bᵀ)`` and the readout + coda.

    ``delta = softplus(log_delta)``, ``a = exp(-delta · exp(log_state_decay))``,
    ``B = delta[:, None] · input_matrix_continuous`` (upstream ``_discretized_dynamics``). The
    state has the dtype of ``z0`` (bf16 under autocast, fp32 otherwise), as upstream.
    """

    def __init__(self, width: int, *, coda_blocks: int, **block_kwargs: Any) -> None:
        super().__init__()
        eps = block_kwargs.get("eps", 1e-5)
        self.width = width
        self.input_norm = nn.LayerNorm(width, eps=eps)
        self.log_delta = nn.Parameter(torch.empty(width))
        self.log_state_decay = nn.Parameter(torch.empty(width))
        self.input_matrix_continuous = nn.Parameter(torch.empty(width, width))
        self.out_proj = nn.Linear(width, width, bias=False)
        self.out_proj._init_identity = True  # ty: ignore[unresolved-attribute]  # read by _init_weights
        self.output_stack = PairStack(coda_blocks, width, **block_kwargs)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Upstream init: ``delta0 = 0.5 ln 5`` (``a = sqrt(1/5)``), ``B = delta0 I``, ``out = I``.

        Skips tensors transformers already loaded, so ``_init_weights`` may call it freely.
        """
        delta0 = 0.5 * math.log(5.0)
        if unloaded(self.log_delta):
            nn.init.constant_(self.log_delta, _inverse_softplus(delta0))
        if unloaded(self.log_state_decay):
            nn.init.zeros_(self.log_state_decay)
        for p in (self.input_matrix_continuous, self.out_proj.weight):
            if unloaded(p):
                with torch.no_grad():
                    p.copy_(torch.eye(self.width, device=p.device, dtype=p.dtype))

    def dynamics(self) -> tuple[Tensor, Tensor]:
        """``(a, B)``: the per-channel decay ``(D,)`` and the input matrix ``(D, D)``."""
        delta = F.softplus(self.log_delta)
        a = torch.exp(-delta * torch.exp(self.log_state_decay))
        return a, delta[:, None] * self.input_matrix_continuous

    def init_state(self, like: Tensor, generator: torch.Generator | None = None) -> Tensor:
        """Truncated-normal ``z_0`` (std ``sqrt(2 / (5 D))``, clipped at 3σ), drawn in fp32."""
        std = math.sqrt(2.0 / (5.0 * like.shape[-1]))
        state = torch.empty(like.shape, dtype=torch.float32, device=like.device)
        nn.init.trunc_normal_(state, 0.0, std, -3 * std, 3 * std, generator=generator)
        return state.to(like.dtype)

    def run(
        self,
        trunk: PairStack,
        inject: Callable[[int], Tensor],
        *,
        z0: Tensor,
        pair_mask: Tensor | None,
        num_loops: int,
        grad_loops: int | None = None,
        return_states: bool = False,
    ) -> tuple[Tensor, list[Tensor]]:
        """Iterate the recurrence ``num_loops`` times from ``z0``.

        Args:
            trunk: The shared pair stack applied after every state update.
            inject: ``inject(t)`` returns the pair injected at loop ``t`` (``z_init`` plus the
                LM pair encoding); it is called inside the loop so per-loop dropout and the LM
                encoder run under the loop's gradient mode.
            z0: Initial state; dtype sets the state dtype.
            pair_mask: ``(B, L, L)`` float pair validity, or ``None``.
            num_loops: Iterations executed (upstream's ``num_loops + 1``).
            grad_loops: Only the last ``grad_loops`` iterations build a graph (truncated
                BPTT, spec §5.6); ``None`` keeps the caller's gradient mode throughout.
            return_states: Also return the state after every loop (parity fixtures).
        """
        if num_loops < 1:
            raise ValueError(f"num_loops must be >= 1; got {num_loops!r}")
        a, b_mat = self.dynamics()
        a = a.to(z0.dtype).view(1, 1, 1, -1)
        b_mat = b_mat.to(z0.dtype)
        no_grad_loops = 0 if grad_loops is None else max(num_loops - grad_loops, 0)
        z, states = z0, []
        for t in range(num_loops):
            ctx = torch.no_grad() if t < no_grad_loops else contextlib.nullcontext()
            with ctx:
                injected = self.input_norm(inject(t))
                z = a * z + F.linear(injected.to(z.dtype), b_mat)
                z = trunk(z, pair_mask)
            if return_states:
                states.append(z)
        return z, states

    def readout(self, z: Tensor, pair_mask: Tensor | None = None) -> Tensor:
        """``output_stack(out_proj(z))`` — the pair every head consumes."""
        return self.output_stack(self.out_proj(z), pair_mask)
