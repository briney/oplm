"""Feed-forward network with configurable activation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn.functional as F
from torch import Tensor, nn

from oplm.model.conv import BidirectionalDepthwiseConv

if TYPE_CHECKING:
    from oplm.config import ModelConfig


class FFN(nn.Module):
    """Feed-forward network with configurable activation.

    Supports three activation variants:

    - **SwiGLU** (default): ``down(silu(gate(x)) * up(x))`` with ``ffn_dim = round(8/3 * D, 256)``
    - **ReLU squared**: ``down(relu(up(x))^2)`` with ``ffn_dim = 4 * D``
    - **GELU**: ``down(gelu(up(x)))`` with ``ffn_dim = 4 * D``

    Optionally applies a Canon-D depthwise convolution in the expanded space
    when ``"D"`` appears in ``conv_positions``.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden_dim = config.hidden_dim
        ffn_dim = config.ffn_dim or 4 * hidden_dim
        self.activation = config.ffn_activation

        if self.activation == "swiglu":
            self.gate_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
            self.up_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        else:
            self.up_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)

        self.down_proj = nn.Linear(ffn_dim, hidden_dim, bias=False)

        # Optional Canon-D convolution in expanded space
        self.conv_d = (
            BidirectionalDepthwiseConv(
                ffn_dim,
                kernel_size=config.conv_kernel_size,
                activation=config.conv_activation,
            )
            if "D" in config.conv_positions
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply feed-forward transformation.

        Args:
            x: Input tensor of shape ``(B, T, D)``.

        Returns:
            Output tensor of shape ``(B, T, D)``.
        """
        if self.activation == "swiglu":
            h = F.silu(self.gate_proj(x)) * self.up_proj(x)  # (B, T, ffn_dim)
        elif self.activation == "relu_squared":
            h = F.relu(self.up_proj(x)).square()  # (B, T, ffn_dim)
        else:  # gelu
            h = F.gelu(self.up_proj(x))  # (B, T, ffn_dim)

        if self.conv_d is not None:
            h = self.conv_d(h)

        return self.down_proj(h)  # (B, T, D)
