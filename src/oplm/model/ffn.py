"""SwiGLU / GEGLU / squared-ReLU feed-forward blocks and the make_ffn factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

from .conv import CanonConv

if TYPE_CHECKING:
    from .configuration_oplm import OplmConfig

__all__ = ["GEGLU", "SquaredReLU", "SwiGLU", "make_ffn", "round_up_to"]


def _apply_canon_d(
    branch: torch.Tensor,
    conv_d: CanonConv | None,
    canon_residual: bool,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Apply Canon-D to an FFN activation branch (no-op when `conv_d is None`).

    Canon-D (Physics of LM 4.1, arXiv:2512.17351) is a depthwise conv inside the
    FFN, before the nonlinearity, at `intermediate_size` width. It is residual by
    default (`branch + conv(branch)`); replacement (`conv(branch)`) is only the
    explicit `canon_residual=False` ablation.

    Args:
        branch: `(B, T, F)` pre-activation FFN branch (gate for SwiGLU/GEGLU, the
            up-projection for squared-ReLU).
        conv_d: The Canon-D conv at `intermediate_size`, or `None` when D is off.
        canon_residual: Add the conv output back to the branch when `True`.
        attention_mask: `(B, T)` pad mask, required when `conv_d` is set so pad
            positions are zeroed before the kernel runs.

    Returns:
        The (optionally) convolved branch, same shape as `branch`.
    """
    if conv_d is None:
        return branch
    if attention_mask is None:
        raise ValueError("Canon-D requires an attention_mask to zero pad positions.")
    convolved = conv_d(branch, attention_mask)
    return branch + convolved if canon_residual else convolved


class SwiGLU(nn.Module):
    """SwiGLU feed-forward block: ``down(silu(gate(x)) * up(x))``.

    Three linear projections share no parameters and all use the same `bias`
    setting. The gated branch is `silu(gate_proj(x))`; it modulates `up_proj(x)`
    elementwise before being projected back to the model dim by `down_proj`.

    When `conv_d` is provided (Canon-D enabled), the depthwise conv runs on the
    gate branch at `intermediate_size` before the SiLU nonlinearity.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
        conv_d: CanonConv | None = None,
        canon_residual: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.conv_d = conv_d
        self.canon_residual = canon_residual
        # Writes back into the residual stream: picked up by
        # OplmPreTrainedModel._init_weights for the 1/sqrt(2L) scaling (§15.1).
        self.down_proj._is_residual_writer = True  # ty: ignore[unresolved-attribute]  # nn.Module setattr

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # (B, T, D) -> (B, T, F) -> (B, T, D)
        gate = _apply_canon_d(self.gate_proj(x), self.conv_d, self.canon_residual, attention_mask)
        return self.down_proj(F.silu(gate) * self.up_proj(x))


class GEGLU(nn.Module):
    """GEGLU feed-forward block: ``down(gelu(gate(x)) * up(x))``.

    Identical in shape and `bias` handling to :class:`SwiGLU`; the only
    difference is the gating nonlinearity (GELU instead of SiLU). The gated
    branch is `gelu(gate_proj(x))`; it modulates `up_proj(x)` elementwise before
    being projected back to the model dim by `down_proj`.

    When `conv_d` is provided (Canon-D enabled), the depthwise conv runs on the
    gate branch at `intermediate_size` before the GELU nonlinearity.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
        conv_d: CanonConv | None = None,
        canon_residual: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.conv_d = conv_d
        self.canon_residual = canon_residual
        # Writes back into the residual stream: picked up by
        # OplmPreTrainedModel._init_weights for the 1/sqrt(2L) scaling (§15.1).
        self.down_proj._is_residual_writer = True  # ty: ignore[unresolved-attribute]  # nn.Module setattr

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # (B, T, D) -> (B, T, F) -> (B, T, D)
        gate = _apply_canon_d(self.gate_proj(x), self.conv_d, self.canon_residual, attention_mask)
        return self.down_proj(F.gelu(gate) * self.up_proj(x))


class SquaredReLU(nn.Module):
    """Squared-ReLU feed-forward block: ``down(relu(up(x)) ** 2)``.

    Non-gated Primer-style FFN (arXiv:2109.08668): two linear projections
    (`up_proj`, `down_proj`) sharing the same `bias` setting, with an
    elementwise ``relu(.) ** 2`` nonlinearity in between. When
    `intermediate_size` is derived from the config, it uses ~4*D (vs ~8/3*D for
    the gated variants) so total FFN params stay at ~8*D^2 either way.

    When `conv_d` is provided (Canon-D enabled), the depthwise conv runs on the
    single up-projection branch at `intermediate_size` before the ReLU.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
        conv_d: CanonConv | None = None,
        canon_residual: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.conv_d = conv_d
        self.canon_residual = canon_residual
        # Writes back into the residual stream: picked up by
        # OplmPreTrainedModel._init_weights for the 1/sqrt(2L) scaling (§15.1).
        self.down_proj._is_residual_writer = True  # ty: ignore[unresolved-attribute]  # nn.Module setattr

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # (B, T, D) -> (B, T, F) -> (B, T, D)
        hidden = _apply_canon_d(self.up_proj(x), self.conv_d, self.canon_residual, attention_mask)
        return self.down_proj(F.relu(hidden).square())


def round_up_to(value: int, multiple: int) -> int:
    """Round `value` up to the nearest non-zero positive `multiple`.

    Used to align `intermediate_size` to a tensor-core / memory-friendly
    boundary when the config does not pin it explicitly.
    """
    return ((value + multiple - 1) // multiple) * multiple


def _make_canon_d(config: OplmConfig, layer_idx: int) -> CanonConv | None:
    """Build the per-layer Canon-D conv at `intermediate_size`, or `None`.

    Canon-D lives inside the FFN (paper-exact placement: before the activation,
    at `intermediate_size`). The per-layer kernel size is read from the resolved
    `canon_kernel_sizes` list — validated independently here because `make_ffn`
    runs during `OplmBlock.__init__` before the block's own Canon validation.
    """
    if not getattr(config, "canon_enabled", False) or "D" not in set(config.canon_positions or []):
        return None
    kernel_sizes = config.canon_kernel_sizes
    if not isinstance(kernel_sizes, list) or len(kernel_sizes) != config.num_hidden_layers:
        raise ValueError(
            "config.canon_kernel_sizes must be a list of length num_hidden_layers; "
            "resolve it via resolve_canon_kernel_sizes() before instantiating the FFN."
        )
    return CanonConv(
        config.intermediate_size,
        kernel_sizes[layer_idx],
        activation=getattr(config, "canon_activation", "none"),
    )


def make_ffn(config: OplmConfig, layer_idx: int = 0) -> nn.Module:
    """Construct the FFN operator selected by `config.ffn_activation`.

    New FFN variants should be added here; the rest of the model stays
    agnostic to which activation is in use.

    Args:
        config: Carries `hidden_size`, `intermediate_size`, `ffn_activation`,
            `ffn_bias`, and the Canon settings used to build the optional
            Canon-D conv.
        layer_idx: Block index, used only to pick this layer's Canon-D kernel
            size from the resolved `canon_kernel_sizes` list.

    Returns:
        A `nn.Module` mapping `(B, T, D) -> (B, T, D)`; `forward` accepts an
        optional `attention_mask` consumed by Canon-D when enabled.

    Raises:
        ValueError: For an unrecognized activation string.
    """
    activation = config.ffn_activation
    conv_d = _make_canon_d(config, layer_idx)
    canon_residual = bool(getattr(config, "canon_residual", True))
    if activation == "swiglu":
        return SwiGLU(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=config.ffn_bias,
            conv_d=conv_d,
            canon_residual=canon_residual,
        )
    if activation == "geglu":
        return GEGLU(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=config.ffn_bias,
            conv_d=conv_d,
            canon_residual=canon_residual,
        )
    if activation == "relu2":
        return SquaredReLU(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=config.ffn_bias,
            conv_d=conv_d,
            canon_residual=canon_residual,
        )
    raise ValueError(
        f"Unknown ffn_activation {activation!r}; expected 'swiglu', 'geglu', or 'relu2'."
    )
