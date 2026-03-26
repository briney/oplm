"""Bidirectional depthwise convolution for local context mixing."""

from __future__ import annotations

from torch import Tensor, nn


class BidirectionalDepthwiseConv(nn.Module):
    """Depthwise Conv1d with symmetric (bidirectional) padding.

    Adaptation of Proust's Canon layers for the encoder: uses "same" padding
    instead of causal (left-only) padding, and odd kernel sizes for symmetry.

    Each channel has its own kernel (``groups=dim``), so the layer has only
    ``dim * kernel_size`` parameters.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        activation: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            groups=dim,
            padding=kernel_size // 2,
        )
        self.act = nn.SiLU() if activation else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Apply depthwise convolution with symmetric padding.

        Args:
            x: Input tensor of shape ``(B, T, D)``.

        Returns:
            Output tensor of shape ``(B, T, D)``.
        """
        # (B, T, D) -> (B, D, T) -> Conv1d -> (B, D, T) -> (B, T, D)
        return self.act(self.conv(x.transpose(1, 2)).transpose(1, 2))
