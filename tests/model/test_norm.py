"""Tests for RMSNorm."""

from __future__ import annotations

import warnings

import torch

from oplm.model.norm import RMSNorm


class TestRMSNorm:
    def test_output_shape(self) -> None:
        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_unit_rms(self) -> None:
        """After normalization (before scale), RMS should be ~1."""
        dim = 128
        norm = RMSNorm(dim)
        # Set weight to ones so we're just testing normalization
        norm.weight.data.fill_(1.0)
        x = torch.randn(4, 16, dim)
        out = norm(x)
        rms = out.pow(2).mean(dim=-1).sqrt()
        torch.testing.assert_close(rms, torch.ones_like(rms), atol=1e-4, rtol=1e-4)

    def test_learnable_weight(self) -> None:
        norm = RMSNorm(32)
        assert norm.weight.shape == (32,)
        assert norm.weight.requires_grad

    def test_2d_input(self) -> None:
        norm = RMSNorm(64)
        x = torch.randn(8, 64)
        out = norm(x)
        assert out.shape == (8, 64)

    def test_4d_input(self) -> None:
        norm = RMSNorm(64)
        x = torch.randn(2, 4, 8, 64)
        out = norm(x)
        assert out.shape == (2, 4, 8, 64)

    def test_bf16_input_avoids_dtype_mismatch_warning(self) -> None:
        norm = RMSNorm(64)
        x = torch.randn(2, 8, 64, dtype=torch.bfloat16)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = norm(x)

        assert out.dtype == torch.bfloat16
        assert norm.weight.dtype == torch.float32
        assert not any("Mismatch dtype between input and weight" in str(w.message) for w in caught)

    def test_bf16_input_backpropagates_to_fp32_weight(self) -> None:
        norm = RMSNorm(64)
        x = torch.randn(2, 8, 64, dtype=torch.bfloat16, requires_grad=True)

        loss = norm(x).float().square().mean()
        loss.backward()

        assert norm.weight.grad is not None
        assert norm.weight.grad.dtype == torch.float32
        assert torch.isfinite(norm.weight.grad).all()
