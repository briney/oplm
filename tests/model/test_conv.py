"""Tests for `oplm.model.conv` — CanonConv, resolve_canon_kernel_sizes."""

from __future__ import annotations

import pytest
import torch

from oplm.model.conv import CanonConv, resolve_canon_kernel_sizes

# ---------------------------------------------------------------------------
# CanonConv shape / construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel_size", [2, 3, 4, 5, 8])
def test_canon_conv_preserves_shape(kernel_size: int):
    conv = CanonConv(hidden_size=16, kernel_size=kernel_size)
    x = torch.randn(2, 7, 16)
    mask = torch.ones(2, 7, dtype=torch.long)
    out = conv(x, mask)
    assert out.shape == x.shape


def test_canon_conv_is_depthwise():
    conv = CanonConv(hidden_size=12, kernel_size=3)
    # Depthwise: groups == in_channels == out_channels; weight has shape (D, 1, k).
    assert conv.conv.groups == 12
    assert conv.conv.weight.shape == (12, 1, 3)
    assert conv.conv.bias is None


def test_canon_conv_rejects_kernel_one():
    with pytest.raises(ValueError, match="kernel_size"):
        CanonConv(hidden_size=4, kernel_size=1)


def test_canon_conv_rejects_unknown_activation():
    with pytest.raises(ValueError, match="activation"):
        CanonConv(hidden_size=4, kernel_size=3, activation="tanh")


# ---------------------------------------------------------------------------
# CanonConv pad zeroing
# ---------------------------------------------------------------------------


def test_canon_conv_zeros_pad_inputs_before_conv():
    """A non-zero value at a pad position must not influence the conv output.

    Compared against a parallel run where the same input is pre-zeroed at the
    pad rows; the two outputs should be bit-identical.
    """
    torch.manual_seed(0)
    conv = CanonConv(hidden_size=8, kernel_size=3)
    x = torch.randn(1, 6, 8)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.long)

    # Inject garbage into pad positions; the conv must ignore it.
    x_with_garbage = x.clone()
    x_with_garbage[:, 4:, :] = 9999.0

    out_garbage = conv(x_with_garbage, mask)
    out_clean = conv(x * mask.unsqueeze(-1), mask)
    assert torch.allclose(out_garbage, out_clean, atol=1e-6)


def test_canon_conv_even_kernel_runs_and_preserves_length():
    conv = CanonConv(hidden_size=8, kernel_size=4)
    x = torch.randn(2, 5, 8)
    mask = torch.ones(2, 5, dtype=torch.long)
    out = conv(x, mask)
    assert out.shape == (2, 5, 8)


def test_canon_conv_odd_kernel_uses_symmetric_padding():
    conv = CanonConv(hidden_size=8, kernel_size=3)
    # Symmetric padding: conv's own padding is k//2 == 1; we add no manual pad.
    assert conv.conv.padding == (1,)
    assert conv._even_kernel is False  # type: ignore[attr-defined]


def test_canon_conv_even_kernel_disables_builtin_padding():
    conv = CanonConv(hidden_size=8, kernel_size=4)
    # Even kernels: conv padding is 0; we pad manually before the conv runs.
    assert conv.conv.padding == (0,)
    assert conv._even_kernel is True  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# CanonConv activations
# ---------------------------------------------------------------------------


def test_canon_conv_no_activation_by_default():
    """With activation='none', a linear conv equals the raw conv output."""
    torch.manual_seed(1)
    conv = CanonConv(hidden_size=8, kernel_size=3, activation="none")
    x = torch.randn(2, 4, 8)
    mask = torch.ones(2, 4, dtype=torch.long)

    out = conv(x, mask)
    # Equivalent manual path: zero pads -> transpose -> conv -> transpose.
    expected = conv.conv((x * mask.unsqueeze(-1)).transpose(1, 2)).transpose(1, 2)
    assert torch.allclose(out, expected, atol=1e-6)


def test_canon_conv_silu_activation_applied():
    torch.manual_seed(2)
    conv = CanonConv(hidden_size=8, kernel_size=3, activation="silu")
    x = torch.randn(1, 4, 8)
    mask = torch.ones(1, 4, dtype=torch.long)

    out = conv(x, mask)
    pre_act = conv.conv((x * mask.unsqueeze(-1)).transpose(1, 2)).transpose(1, 2)
    assert torch.allclose(out, torch.nn.functional.silu(pre_act), atol=1e-6)


def test_canon_conv_gelu_activation_applied():
    torch.manual_seed(3)
    conv = CanonConv(hidden_size=8, kernel_size=3, activation="gelu")
    x = torch.randn(1, 4, 8)
    mask = torch.ones(1, 4, dtype=torch.long)

    out = conv(x, mask)
    pre_act = conv.conv((x * mask.unsqueeze(-1)).transpose(1, 2)).transpose(1, 2)
    assert torch.allclose(out, torch.nn.functional.gelu(pre_act), atol=1e-6)


# ---------------------------------------------------------------------------
# CanonConv gradient flow
# ---------------------------------------------------------------------------


def test_canon_conv_grad_flows_to_weight_and_input():
    conv = CanonConv(hidden_size=8, kernel_size=3)
    x = torch.randn(2, 4, 8, requires_grad=True)
    mask = torch.ones(2, 4, dtype=torch.long)
    conv(x, mask).sum().backward()
    assert conv.conv.weight.grad is not None
    assert conv.conv.weight.grad.abs().sum() > 0
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# resolve_canon_kernel_sizes
# ---------------------------------------------------------------------------


def test_resolve_scalar_broadcasts():
    assert resolve_canon_kernel_sizes(4, num_hidden_layers=6) == [4, 4, 4, 4, 4, 4]


def test_resolve_list_passthrough():
    assert resolve_canon_kernel_sizes([2, 3, 4, 5], num_hidden_layers=4) == [2, 3, 4, 5]


def test_resolve_list_wrong_length_raises():
    with pytest.raises(ValueError, match="length"):
        resolve_canon_kernel_sizes([3, 3, 3], num_hidden_layers=4)


def test_resolve_constant_schedule():
    out = resolve_canon_kernel_sizes({"schedule": "constant", "value": 5}, num_hidden_layers=4)
    assert out == [5, 5, 5, 5]


def test_resolve_linear_schedule_endpoints_and_length():
    out = resolve_canon_kernel_sizes(
        {"schedule": "linear", "min": 2, "max": 8}, num_hidden_layers=4
    )
    assert len(out) == 4
    assert out[0] == 2
    assert out[-1] == 8
    # Linear interpolation rounded to ints: monotonic non-decreasing.
    assert all(b >= a for a, b in zip(out, out[1:], strict=False))


def test_resolve_linear_schedule_single_layer():
    out = resolve_canon_kernel_sizes(
        {"schedule": "linear", "min": 3, "max": 9}, num_hidden_layers=1
    )
    # Endpoint of a single-element linspace is `min` (np.linspace convention).
    assert out == [3]


def test_resolve_linear_schedule_decreasing():
    out = resolve_canon_kernel_sizes(
        {"schedule": "linear", "min": 8, "max": 2}, num_hidden_layers=4
    )
    assert out[0] == 8
    assert out[-1] == 2
    assert all(b <= a for a, b in zip(out, out[1:], strict=False))


def test_resolve_rejects_kernel_below_two():
    with pytest.raises(ValueError, match=">= 2"):
        resolve_canon_kernel_sizes([2, 1, 3, 4], num_hidden_layers=4)
    with pytest.raises(ValueError, match=">= 2"):
        resolve_canon_kernel_sizes(1, num_hidden_layers=3)


def test_resolve_unknown_schedule_raises():
    with pytest.raises(ValueError, match="schedule"):
        resolve_canon_kernel_sizes(
            {"schedule": "exponential", "min": 2, "max": 8}, num_hidden_layers=4
        )


def test_resolve_linear_missing_keys_raises():
    with pytest.raises(ValueError, match="min"):
        resolve_canon_kernel_sizes({"schedule": "linear", "max": 8}, num_hidden_layers=4)


def test_resolve_constant_missing_value_raises():
    with pytest.raises(ValueError, match="value"):
        resolve_canon_kernel_sizes({"schedule": "constant"}, num_hidden_layers=4)


def test_resolve_rejects_unsupported_type():
    with pytest.raises(ValueError, match="int"):
        resolve_canon_kernel_sizes("4", num_hidden_layers=3)  # type: ignore[arg-type]


def test_resolve_rejects_bool():
    # bool is a subclass of int; we explicitly reject it so a stray True/False
    # doesn't silently become an all-1s or all-0s kernel schedule.
    with pytest.raises(ValueError, match="bool"):
        resolve_canon_kernel_sizes(True, num_hidden_layers=3)  # type: ignore[arg-type]
