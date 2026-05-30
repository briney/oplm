"""LayerNorm and RMSNorm with fp32 internals, plus the make_norm factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["OplmLayerNorm", "OplmRMSNorm", "make_norm"]


def _as_shape_tuple(normalized_shape: int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(normalized_shape, int):
        return (normalized_shape,)
    return tuple(normalized_shape)


class OplmLayerNorm(nn.Module):
    """LayerNorm with forced fp32 internal compute.

    Numerically equivalent to `torch.nn.LayerNorm` but always computes the mean,
    variance, and normalization in fp32 regardless of input dtype. The learnable
    affine parameters are stored in fp32 and the result is cast back to the
    input dtype on the way out.
    """

    def __init__(
        self,
        normalized_shape: int | Sequence[int],
        eps: float = 1e-6,
        bias: bool = True,
    ) -> None:
        super().__init__()
        shape = _as_shape_tuple(normalized_shape)
        self.normalized_shape = shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(shape))
        if bias:
            self.bias = nn.Parameter(torch.zeros(shape))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        mean = x_fp32.mean(-1, keepdim=True)
        var = x_fp32.var(-1, keepdim=True, unbiased=False)
        x_norm = (x_fp32 - mean) * torch.rsqrt(var + self.eps)
        x_norm = x_norm * self.weight.to(torch.float32)
        if self.bias is not None:
            x_norm = x_norm + self.bias.to(torch.float32)
        return x_norm.to(input_dtype)

    def extra_repr(self) -> str:
        return (
            f"normalized_shape={self.normalized_shape}, eps={self.eps}, "
            f"bias={self.bias is not None}"
        )


class OplmRMSNorm(nn.Module):
    """Root-mean-square normalization with forced fp32 internal compute.

    Computes `weight * x / sqrt(mean(x**2, dim=-1) + eps)` in fp32 and casts
    back to the input dtype. No bias term.
    """

    def __init__(
        self,
        normalized_shape: int | Sequence[int],
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        shape = _as_shape_tuple(normalized_shape)
        self.normalized_shape = shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        rms = torch.sqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        x_norm = x_fp32 / rms
        x_norm = x_norm * self.weight.to(torch.float32)
        return x_norm.to(input_dtype)

    def extra_repr(self) -> str:
        return f"normalized_shape={self.normalized_shape}, eps={self.eps}"


def make_norm(
    norm_type: str,
    normalized_shape: int | Sequence[int],
    eps: float = 1e-6,
) -> nn.Module:
    """Construct the norm operator selected by `config.norm_type`.

    A single config flag controls every norm site in the model (pre-norm,
    sandwich/hybrid/post-sdpa norms, QK-norm, post-embedding norm, final norm,
    MLM-head intermediate norm, pre-head norm).

    Args:
        norm_type: One of `"layernorm"` or `"rmsnorm"`.
        normalized_shape: Shape of the trailing dim(s) to normalize over.
        eps: Numerical stability term.

    Returns:
        An `OplmLayerNorm` or `OplmRMSNorm` instance.

    Raises:
        ValueError: If `norm_type` is not recognized.
    """
    if norm_type == "layernorm":
        return OplmLayerNorm(normalized_shape, eps=eps)
    if norm_type == "rmsnorm":
        return OplmRMSNorm(normalized_shape, eps=eps)
    raise ValueError(f"Unknown norm_type {norm_type!r}; expected 'layernorm' or 'rmsnorm'.")
