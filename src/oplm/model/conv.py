"""Bidirectional depthwise 1D convolution for Canon insertion points."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .masking import zero_pad_positions

__all__ = ["CanonConv", "resolve_canon_kernel_sizes"]


_ACTIVATIONS = {
    "none": None,
    "silu": F.silu,
    "gelu": F.gelu,
}


class CanonConv(nn.Module):
    """Per-channel 1D conv along the sequence axis for Canon insertion sites.

    The conv is depthwise (`groups == hidden_size`) and bidirectional: the
    encoder sees both past and future tokens. Pad positions are zeroed before
    the kernel runs so that pad content cannot leak into real-token channels
    via the receptive field.

    For odd `kernel_size`, `nn.Conv1d`'s symmetric `padding=k//2` already
    produces a same-length output. For even `kernel_size`, `padding` is fixed
    to `0` and the input is asymmetrically padded `(k//2, k//2 - 1)` on the
    time axis before the conv runs — this preserves the same-length output
    while keeping the bias of the receptive field consistent across layers.
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        activation: str = "none",
    ) -> None:
        super().__init__()
        if kernel_size < 2:
            raise ValueError(f"kernel_size must be >= 2; got {kernel_size}.")
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation {activation!r}; expected one of {sorted(_ACTIVATIONS)}."
            )

        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.activation_name = activation
        self._activation = _ACTIVATIONS[activation]
        self._even_kernel = kernel_size % 2 == 0

        # Odd kernels: built-in symmetric padding produces same-length output.
        # Even kernels: pad manually below — F.pad with (k//2, k//2 - 1).
        conv_padding = 0 if self._even_kernel else kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            padding=conv_padding,
            bias=False,
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # (B, T, D) -> (B, T, D)
        x = zero_pad_positions(x, attention_mask)
        x = x.transpose(1, 2)  # (B, T, D) -> (B, D, T)
        if self._even_kernel:
            # Asymmetric pad keeps output length == input length for even k.
            x = F.pad(x, (self.kernel_size // 2, self.kernel_size // 2 - 1))
        x = self.conv(x)
        x = x.transpose(1, 2)  # (B, D, T) -> (B, T, D)
        if self._activation is not None:
            x = self._activation(x)
        return x


def resolve_canon_kernel_sizes(spec: Any, num_hidden_layers: int) -> list[int]:
    """Expand `canon_kernel_sizes` into a per-layer `list[int]`.

    Accepts the three forms documented in the architecture doc:

    * `int` — broadcast across every layer.
    * `list[int]` — must already have length `num_hidden_layers`.
    * `dict` with `schedule="linear"` (keys `min`, `max`) or
      `schedule="constant"` (key `value`).

    The returned list is validated to ensure every entry is `>= 2`; even
    entries are allowed (the `CanonConv` asymmetric-pad path handles them).

    Args:
        spec: The raw `canon_kernel_sizes` value from the config.
        num_hidden_layers: Target list length.

    Returns:
        A `list[int]` of length `num_hidden_layers`.

    Raises:
        ValueError: For a length mismatch, an unknown schedule, a missing
            schedule key, an unsupported `spec` type, or any resolved entry
            `< 2`.
    """
    if isinstance(spec, bool):
        # bool is a subclass of int; reject before the int branch fires.
        raise ValueError(f"canon_kernel_sizes must be int/list/dict; got bool {spec!r}.")
    if isinstance(spec, int):
        resolved = [spec] * num_hidden_layers
    elif isinstance(spec, list):
        if len(spec) != num_hidden_layers:
            raise ValueError(
                f"canon_kernel_sizes list has length {len(spec)}; expected {num_hidden_layers}."
            )
        resolved = [int(v) for v in spec]
    elif isinstance(spec, dict):
        schedule = spec.get("schedule")
        if schedule == "linear":
            if "min" not in spec or "max" not in spec:
                raise ValueError(
                    "canon_kernel_sizes linear schedule requires 'min' and 'max' keys."
                )
            min_k = int(spec["min"])
            max_k = int(spec["max"])
            resolved = np.linspace(min_k, max_k, num_hidden_layers).round().astype(int).tolist()
        elif schedule == "constant":
            if "value" not in spec:
                raise ValueError("canon_kernel_sizes constant schedule requires a 'value' key.")
            resolved = [int(spec["value"])] * num_hidden_layers
        else:
            raise ValueError(
                f"Unknown canon_kernel_sizes schedule {schedule!r}; "
                "expected 'linear' or 'constant'."
            )
    else:
        raise ValueError(
            f"canon_kernel_sizes must be int, list[int], or dict; got {type(spec).__name__}."
        )

    bad = [v for v in resolved if v < 2]
    if bad:
        raise ValueError(f"canon_kernel_sizes must be >= 2 at every layer; got entries {bad}.")
    return resolved
