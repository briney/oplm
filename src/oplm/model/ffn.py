"""SwiGLU / GEGLU / squared-ReLU feed-forward blocks and the make_ffn factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from .configuration_oplm import OplmConfig

__all__ = ["GEGLU", "SquaredReLU", "SwiGLU", "make_ffn", "round_up_to"]


class SwiGLU(nn.Module):
    """SwiGLU feed-forward block: ``down(silu(gate(x)) * up(x))``.

    Three linear projections share no parameters and all use the same `bias`
    setting. The gated branch is `silu(gate_proj(x))`; it modulates `up_proj(x)`
    elementwise before being projected back to the model dim by `down_proj`.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        # Writes back into the residual stream: picked up by
        # OplmPreTrainedModel._init_weights for the 1/sqrt(2L) scaling (§15.1).
        self.down_proj._is_residual_writer = True  # ty: ignore[unresolved-attribute]  # nn.Module setattr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, D) -> (B, T, F) -> (B, T, D)
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class GEGLU(nn.Module):
    """GEGLU feed-forward block: ``down(gelu(gate(x)) * up(x))``.

    Identical in shape and `bias` handling to :class:`SwiGLU`; the only
    difference is the gating nonlinearity (GELU instead of SiLU). The gated
    branch is `gelu(gate_proj(x))`; it modulates `up_proj(x)` elementwise before
    being projected back to the model dim by `down_proj`.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        # Writes back into the residual stream: picked up by
        # OplmPreTrainedModel._init_weights for the 1/sqrt(2L) scaling (§15.1).
        self.down_proj._is_residual_writer = True  # ty: ignore[unresolved-attribute]  # nn.Module setattr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, D) -> (B, T, F) -> (B, T, D)
        return self.down_proj(F.gelu(self.gate_proj(x)) * self.up_proj(x))


class SquaredReLU(nn.Module):
    """Squared-ReLU feed-forward block: ``down(relu(up(x)) ** 2)``.

    Non-gated Primer-style FFN (arXiv:2109.08668): two linear projections
    (`up_proj`, `down_proj`) sharing the same `bias` setting, with an
    elementwise ``relu(.) ** 2`` nonlinearity in between. When
    `intermediate_size` is derived from the config, it uses ~4*D (vs ~8/3*D for
    the gated variants) so total FFN params stay at ~8*D^2 either way.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        # Writes back into the residual stream: picked up by
        # OplmPreTrainedModel._init_weights for the 1/sqrt(2L) scaling (§15.1).
        self.down_proj._is_residual_writer = True  # ty: ignore[unresolved-attribute]  # nn.Module setattr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, D) -> (B, T, F) -> (B, T, D)
        return self.down_proj(F.relu(self.up_proj(x)).square())


def round_up_to(value: int, multiple: int) -> int:
    """Round `value` up to the nearest non-zero positive `multiple`.

    Used to align `intermediate_size` to a tensor-core / memory-friendly
    boundary when the config does not pin it explicitly.
    """
    return ((value + multiple - 1) // multiple) * multiple


def make_ffn(config: OplmConfig) -> nn.Module:
    """Construct the FFN operator selected by `config.ffn_activation`.

    New FFN variants should be added here; the rest of the model stays
    agnostic to which activation is in use.

    Args:
        config: Carries `hidden_size`, `intermediate_size`, `ffn_activation`,
            and `ffn_bias`.

    Returns:
        A `nn.Module` mapping `(B, T, D) -> (B, T, D)`.

    Raises:
        ValueError: For an unrecognized activation string.
    """
    activation = config.ffn_activation
    if activation == "swiglu":
        return SwiGLU(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=config.ffn_bias,
        )
    if activation == "geglu":
        return GEGLU(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=config.ffn_bias,
        )
    if activation == "relu2":
        return SquaredReLU(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=config.ffn_bias,
        )
    raise ValueError(
        f"Unknown ffn_activation {activation!r}; expected 'swiglu', 'geglu', or 'relu2'."
    )
