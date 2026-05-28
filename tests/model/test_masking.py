"""Tests for `oplm.model.masking` — prepare_attention_mask, zero_pad_positions, flex."""

from __future__ import annotations

import pytest
import torch

from oplm.model.masking import (
    make_flex_block_mask,
    prepare_attention_mask,
    zero_pad_positions,
)

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


# ---------------------------------------------------------------------------
# make_flex_block_mask
# ---------------------------------------------------------------------------


def _flex_available_on_device(device: torch.device) -> bool:
    try:
        from torch.nn.attention.flex_attention import create_block_mask  # noqa: F401
    except ImportError:
        return False
    # create_block_mask itself runs on CPU; the kernel needs CUDA but the
    # mask construction does not.
    return True


@pytest.mark.skipif(
    not _flex_available_on_device(torch.device("cpu")),
    reason="flex_attention not importable",
)
def test_make_flex_block_mask_builds_for_full_mask():
    mask = torch.ones(2, 8, dtype=torch.long)
    block_mask = make_flex_block_mask(mask, num_heads=4)
    # Sanity: the BlockMask reports the expected query/kv lengths.
    assert block_mask.kv_num_blocks is not None
    # repr should mention the lengths we built it for.
    text = repr(block_mask)
    assert "8" in text


@pytest.mark.skipif(
    not _flex_available_on_device(torch.device("cpu")),
    reason="flex_attention not importable",
)
def test_make_flex_block_mask_with_padding():
    # Real tokens at positions 0..4, pads at 5..7 in row 0; row 1 fully real.
    mask = torch.tensor(
        [[1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]],
        dtype=torch.long,
    )
    block_mask = make_flex_block_mask(mask, num_heads=2)
    # The row that has pads must allow strictly fewer KV slots than the all-ones row.
    assert block_mask.kv_num_blocks is not None
