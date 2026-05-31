"""Tests for `oplm.model.masking` — prepare_attention_mask, zero_pad_positions."""

from __future__ import annotations

import pytest
import torch

from oplm.model.masking import prepare_attention_mask, zero_pad_positions

# ---------------------------------------------------------------------------
# prepare_attention_mask
# ---------------------------------------------------------------------------


def test_prepare_attention_mask_defaults_to_all_ones():
    mask = prepare_attention_mask(None, batch_size=2, seq_len=5, device="cpu")
    assert mask.shape == (2, 5)
    assert mask.dtype == torch.long
    assert mask.device.type == "cpu"
    assert torch.equal(mask, torch.ones(2, 5, dtype=torch.long))


def test_prepare_attention_mask_honors_dtype_for_default():
    mask = prepare_attention_mask(None, batch_size=2, seq_len=3, device="cpu", dtype=torch.bool)
    assert mask.dtype == torch.bool
    assert mask.all()


def test_prepare_attention_mask_returns_caller_mask_as_is():
    supplied = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.long)
    out = prepare_attention_mask(supplied, batch_size=2, seq_len=3, device="cpu")
    assert out is supplied


def test_prepare_attention_mask_rejects_bad_shape():
    bad = torch.ones(2, 4, dtype=torch.long)
    with pytest.raises(ValueError, match="expected"):
        prepare_attention_mask(bad, batch_size=2, seq_len=5, device="cpu")


def test_prepare_attention_mask_rejects_wrong_rank():
    bad = torch.ones(2, 3, 5, dtype=torch.long)
    with pytest.raises(ValueError):
        prepare_attention_mask(bad, batch_size=2, seq_len=3, device="cpu")


# ---------------------------------------------------------------------------
# zero_pad_positions
# ---------------------------------------------------------------------------


def test_zero_pad_positions_zeros_pad_rows():
    x = torch.randn(2, 4, 8)
    mask = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.long)
    out = zero_pad_positions(x, mask)
    # Pad rows are zero.
    assert torch.equal(out[0, 2], torch.zeros(8))
    assert torch.equal(out[0, 3], torch.zeros(8))
    assert torch.equal(out[1, 1], torch.zeros(8))
    assert torch.equal(out[1, 2], torch.zeros(8))
    assert torch.equal(out[1, 3], torch.zeros(8))


def test_zero_pad_positions_leaves_real_rows_untouched():
    x = torch.randn(2, 4, 8)
    mask = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.long)
    out = zero_pad_positions(x, mask)
    assert torch.equal(out[0, 0], x[0, 0])
    assert torch.equal(out[0, 1], x[0, 1])
    assert torch.equal(out[1, 0], x[1, 0])


def test_zero_pad_positions_preserves_input_dtype():
    x = torch.randn(2, 3, 4, dtype=torch.bfloat16)
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.long)
    out = zero_pad_positions(x, mask)
    assert out.dtype == torch.bfloat16
    assert out.shape == x.shape


def test_zero_pad_positions_with_float_mask():
    x = torch.randn(2, 3, 4)
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    out = zero_pad_positions(x, mask)
    assert torch.equal(out[0, 2], torch.zeros(4))
    assert torch.equal(out[1, 1], torch.zeros(4))
    assert torch.equal(out[0, 0], x[0, 0])
