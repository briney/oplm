"""Root Mean Square Layer Normalization."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Single learnable scale parameter per dimension, no bias.
    Uses ``torch.nn.functional.rms_norm`` (fused by torch.compile).
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """Apply RMS normalization.

        Args:
            x: Input tensor of shape ``(..., dim)``.

        Returns:
            Normalized tensor of the same shape.
        """
        # Match the affine scale to the activation dtype so mixed-precision
        # runs can use the fused kernel while keeping the master weight in FP32.
        weight = self.weight if self.weight.dtype == x.dtype else self.weight.to(dtype=x.dtype)
        return F.rms_norm(x, (self.dim,), weight, self.eps)
