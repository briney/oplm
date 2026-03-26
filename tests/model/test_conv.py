"""Tests for the BidirectionalDepthwiseConv module."""

from __future__ import annotations

import torch

from oplm.model.conv import BidirectionalDepthwiseConv


B, T, D = 2, 16, 64


# ---------------------------------------------------------------------------
# Output shape tests
# ---------------------------------------------------------------------------


class TestConvOutputShape:
    """Verify output shape equals input shape (same padding)."""

    def test_default_kernel(self) -> None:
        conv = BidirectionalDepthwiseConv(D)
        x = torch.randn(B, T, D)
        assert conv(x).shape == (B, T, D)

    def test_kernel_3(self) -> None:
        conv = BidirectionalDepthwiseConv(D, kernel_size=3)
        x = torch.randn(B, T, D)
        assert conv(x).shape == (B, T, D)

    def test_kernel_15(self) -> None:
        conv = BidirectionalDepthwiseConv(D, kernel_size=15)
        x = torch.randn(B, T, D)
        assert conv(x).shape == (B, T, D)

    def test_single_token_sequence(self) -> None:
        conv = BidirectionalDepthwiseConv(D, kernel_size=7)
        x = torch.randn(B, 1, D)
        assert conv(x).shape == (B, 1, D)

    def test_various_seq_lengths(self) -> None:
        conv = BidirectionalDepthwiseConv(D, kernel_size=7)
        for seq_len in [1, 3, 7, 8, 32, 128]:
            x = torch.randn(B, seq_len, D)
            assert conv(x).shape == (B, seq_len, D)


# ---------------------------------------------------------------------------
# Depthwise property
# ---------------------------------------------------------------------------


class TestConvDepthwise:
    """Verify depthwise (groups=dim) parameterization."""

    def test_groups_equals_dim(self) -> None:
        conv = BidirectionalDepthwiseConv(D, kernel_size=7)
        assert conv.conv.groups == D

    def test_parameter_count(self) -> None:
        """Depthwise conv should have dim * kernel_size + dim (bias) params."""
        k = 7
        conv = BidirectionalDepthwiseConv(D, kernel_size=k)
        # weight: (D, 1, k), bias: (D,)
        total = sum(p.numel() for p in conv.conv.parameters())
        assert total == D * k + D


# ---------------------------------------------------------------------------
# Bidirectional (symmetric) padding
# ---------------------------------------------------------------------------


class TestConvBidirectional:
    """Verify symmetric padding produces bidirectional context."""

    def test_symmetric_padding_amount(self) -> None:
        """Padding should be kernel_size // 2 on each side."""
        k = 7
        conv = BidirectionalDepthwiseConv(D, kernel_size=k)
        assert conv.conv.padding == (k // 2,)

    def test_center_token_sees_both_sides(self) -> None:
        """The middle token's output should depend on tokens on both sides."""
        conv = BidirectionalDepthwiseConv(D, kernel_size=7, activation=False)
        x = torch.zeros(1, 16, D)

        # Set left and right neighbors of center differently
        x[:, 6, :] = 1.0  # left of center
        x[:, 8, :] = 1.0  # right of center
        out = conv(x)

        # Center token (idx 7) should be nonzero due to neighbors
        assert out[:, 7, :].abs().sum() > 0

    def test_first_and_last_tokens_have_output(self) -> None:
        """Edge tokens should produce valid (nonzero) output."""
        conv = BidirectionalDepthwiseConv(D, kernel_size=7, activation=False)
        x = torch.randn(B, T, D)
        out = conv(x)
        assert out[:, 0, :].abs().sum() > 0
        assert out[:, -1, :].abs().sum() > 0


# ---------------------------------------------------------------------------
# Activation toggle
# ---------------------------------------------------------------------------


class TestConvActivation:
    """Verify the optional SiLU activation."""

    def test_activation_enabled_by_default(self) -> None:
        conv = BidirectionalDepthwiseConv(D)
        assert isinstance(conv.act, torch.nn.SiLU)

    def test_activation_disabled(self) -> None:
        conv = BidirectionalDepthwiseConv(D, activation=False)
        assert isinstance(conv.act, torch.nn.Identity)

    def test_activation_changes_output(self) -> None:
        """With vs without activation should differ."""
        torch.manual_seed(42)
        x = torch.randn(B, T, D)

        torch.manual_seed(0)
        conv_act = BidirectionalDepthwiseConv(D, activation=True)
        torch.manual_seed(0)
        conv_no = BidirectionalDepthwiseConv(D, activation=False)

        out_act = conv_act(x)
        out_no = conv_no(x)
        assert not torch.allclose(out_act, out_no, atol=1e-6)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


class TestConvGradient:
    """Verify gradients flow through the conv."""

    def test_gradient_flow(self) -> None:
        conv = BidirectionalDepthwiseConv(D, kernel_size=7)
        x = torch.randn(B, T, D, requires_grad=True)
        loss = conv(x).sum()
        loss.backward()
        for name, param in conv.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"
