"""SwiGLU feed-forward block and the make_ffn factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from .configuration_oplm import OplmConfig

__all__ = ["SwiGLU", "make_ffn", "round_up_to"]


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
        self.down_proj._is_residual_writer = True  # type: ignore[attr-defined]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, D) -> (B, T, F) -> (B, T, D)
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def round_up_to(value: int, multiple: int) -> int:
    """Round `value` up to the nearest non-zero positive `multiple`.

    Used to align `intermediate_size` to a tensor-core / memory-friendly
    boundary when the config does not pin it explicitly.
    """
    return ((value + multiple - 1) // multiple) * multiple


def make_ffn(config: OplmConfig) -> nn.Module:
    """Construct the FFN operator selected by `config.ffn_activation`.

    New gated-FFN variants should be added here; the rest of the model stays
    agnostic to which activation is in use.

    Args:
        config: Carries `hidden_size`, `intermediate_size`, `ffn_activation`,
            and `ffn_bias`.

    Returns:
        A `nn.Module` mapping `(B, T, D) -> (B, T, D)`.

    Raises:
        NotImplementedError: For activations that are accepted by config
            validation but not yet implemented (currently `"geglu"`).
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
        raise NotImplementedError("ffn_activation='geglu' is reserved but not yet implemented.")
    raise ValueError(f"Unknown ffn_activation {activation!r}; expected 'swiglu' or 'geglu'.")
