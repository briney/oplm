"""Core-boundary loop modules: input injection, per-cycle gating, step conditioning."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .norm import make_norm

__all__ = ["LoopCycleGate", "LoopInputInjector", "LoopStepConditioner"]

_SQRT2_INV = 2.0**-0.5


def _inverse_softplus(y: float) -> float:
    return math.log(math.expm1(y))


class LoopInputInjector(nn.Module):
    """Re-inject the prelude representation ``e`` into the recurrent state ``z`` on pass ``k >= 1``.

    Modes:
        * ``"concat_proj"``: ``proj(cat([norm_z(z), norm_e(e)], dim=-1) / sqrt(2))``.
        * ``"decayed"``: continuous-time decayed state + prelude projection matching
          ``oplm.fold.trunk.Recurrence`` (``a * z + linear(norm(e), B)``).
        * ``"orthogonal"``: project out the component of ``z`` parallel to ``e`` in fp32,
          then add ``inject_scale * e``.
    """

    def __init__(
        self,
        mode: str,
        hidden_size: int,
        norm_type: str = "layernorm",
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if mode not in ("concat_proj", "decayed", "orthogonal"):
            raise ValueError(
                f"LoopInputInjector mode must be 'concat_proj', 'decayed', or 'orthogonal'; "
                f"got {mode!r}."
            )
        self.mode = mode
        self.hidden_size = hidden_size

        if mode == "concat_proj":
            self.z_norm: nn.Module = make_norm(norm_type, hidden_size, eps=norm_eps)
            self.e_norm: nn.Module = make_norm(norm_type, hidden_size, eps=norm_eps)
            self.proj = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        elif mode == "decayed":
            self.input_norm: nn.Module = make_norm(norm_type, hidden_size, eps=norm_eps)
            self.log_delta = nn.Parameter(torch.empty(hidden_size))
            self.log_state_decay = nn.Parameter(torch.empty(hidden_size))
            self.input_matrix_continuous = nn.Parameter(torch.empty(hidden_size, hidden_size))
            self.reset_parameters()
        else:  # "orthogonal"
            self.inject_scale = nn.Parameter(torch.ones(hidden_size))

    def reset_parameters(self) -> None:
        """Initialize custom parameters (idempotent; skips already-loaded weights if desired)."""
        if self.mode == "decayed":
            delta0 = 0.5 * math.log(5.0)
            with torch.no_grad():
                self.log_delta.fill_(_inverse_softplus(delta0))
                self.log_state_decay.zero_()
                self.input_matrix_continuous.copy_(
                    torch.eye(
                        self.hidden_size,
                        device=self.input_matrix_continuous.device,
                        dtype=self.input_matrix_continuous.dtype,
                    )
                )
        elif self.mode == "orthogonal":
            with torch.no_grad():
                self.inject_scale.fill_(1.0)

    def forward(self, z: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """Combine recurrent core state ``z`` with prelude anchor ``e``."""
        if self.mode == "concat_proj":
            h = torch.cat((self.z_norm(z), self.e_norm(e)), dim=-1) * _SQRT2_INV
            return self.proj(h).to(z.dtype)
        if self.mode == "decayed":
            delta = F.softplus(self.log_delta.float())
            a = torch.exp(-delta * torch.exp(self.log_state_decay.float()))
            b_mat = delta[:, None] * self.input_matrix_continuous.float()
            e_normed = self.input_norm(e).to(z.dtype)
            return a.to(z.dtype) * z + F.linear(e_normed, b_mat.to(z.dtype))
        # "orthogonal": project z orthogonal to e in fp32 with zero-norm-safe denominator.
        z_f = z.float()
        e_f = e.float()
        e_norm = torch.sqrt((e_f * e_f).sum(dim=-1, keepdim=True).clamp(min=1e-12))
        e_hat = e_f / e_norm
        z_orth = z_f - (z_f * e_hat).sum(dim=-1, keepdim=True) * e_hat
        out = z_orth + self.inject_scale.float() * e_f
        return out.to(z.dtype)


class LoopCycleGate(nn.Module):
    """Per-pass 1-D residual gate: ``z_k = z_{k-1} + g_k * (core_out - z_{k-1})``.

    Stores ``num_loops - 1`` separate 1-D parameters inside an ``nn.ParameterList`` so
    every gate lands in AdamW no-decay at ``lr * 1`` and avoids 2-D Muon/FSDP2 edge cases.
    """

    def __init__(self, mode: str, num_loops: int, hidden_size: int, init_val: float = 1.0) -> None:
        super().__init__()
        if mode not in ("scalar", "channel"):
            raise ValueError(f"LoopCycleGate mode must be 'scalar' or 'channel'; got {mode!r}.")
        if num_loops < 2:
            raise ValueError(f"LoopCycleGate requires num_loops >= 2; got {num_loops}.")
        self.mode = mode
        shape = (1,) if mode == "scalar" else (hidden_size,)
        self.gates = nn.ParameterList(
            [nn.Parameter(torch.full(shape, float(init_val))) for _ in range(num_loops - 1)]
        )

    def forward(self, z_prev: torch.Tensor, core_out: torch.Tensor, gate_idx: int) -> torch.Tensor:
        """Blend ``core_out`` into ``z_prev`` using pass gate ``gate_idx = pass_idx - 1``."""
        g = self.gates[gate_idx].to(z_prev.dtype)
        return z_prev + g * (core_out - z_prev)


def _sinusoidal_encoding(
    t: float,
    dim: int,
    *,
    device: torch.device,
    max_period: float = 10000.0,
) -> torch.Tensor:
    """Compute a ``(dim,)`` fp32 sinusoidal encoding of scalar progress ``t in [0, 1]``."""
    half = dim // 2
    if half == 0:
        return torch.zeros(dim, device=device, dtype=torch.float32)
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=device, dtype=torch.float32) / half
    )
    args = freqs * float(t)
    emb = torch.cat((torch.sin(args), torch.cos(args)), dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class LoopStepConditioner(nn.Module):
    """Sinusoidal pass-progress encoding with zero-initialized residual FiLM at core entry.

    Uses encoding width ``D = hidden_size`` so both ``fc1: Linear(D, D)`` and
    ``fc2: Linear(D, 2 * D)`` have hidden fan-in ``D`` and obey standard μP/Muon routing
    with zero special cases. ``fc2`` is tagged ``_init_zero = True`` so at step 0
    ``gamma = beta = 0`` and the module is an exact identity.
    """

    def __init__(
        self,
        hidden_size: int,
        norm_type: str = "layernorm",
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.norm: nn.Module = make_norm(norm_type, hidden_size, eps=norm_eps)
        self.fc1 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        self.fc2._init_zero = True  # ty: ignore[unresolved-attribute]  # read by _init_weights
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, z: torch.Tensor, pass_idx: int, num_loops: int) -> torch.Tensor:
        """Apply residual FiLM ``z + gamma * norm(z) + beta`` conditioned on loop progress."""
        t = pass_idx / max(num_loops - 1, 1)
        enc_dtype = z.dtype if torch.is_autocast_enabled(z.device.type) else self.fc1.weight.dtype
        enc = _sinusoidal_encoding(t, self.hidden_size, device=z.device).to(enc_dtype)
        mod = self.fc2(F.silu(self.fc1(enc)))
        gamma, beta = mod.chunk(2, dim=-1)
        gamma = gamma.to(z.dtype).view(1, 1, -1)
        beta = beta.to(z.dtype).view(1, 1, -1)
        return z + gamma * self.norm(z).to(z.dtype) + beta
