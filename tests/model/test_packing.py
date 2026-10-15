"""Sequence packing: a packed row must give each sequence exactly what its own padded row gives."""

from __future__ import annotations

import pytest
import torch

from oplm import OplmConfig, OplmForMaskedLM
from oplm.model.masking import segment_attention_mask

LENGTHS = (5, 7, 3)


def test_segment_mask_is_block_diagonal_with_no_empty_rows() -> None:
    mask = segment_attention_mask(torch.tensor([[1, 1, 1, 2, 2, 3, 0, 0]]))[0, 0]
    assert mask.shape == (8, 8) and bool(mask.any(-1).all())
    assert mask[3].tolist() == [False, False, False, True, True, False, False, False]
    assert mask[7].tolist() == [False] * 6 + [True, True]


def _model(checkpointing: str | None) -> OplmForMaskedLM:
    torch.manual_seed(0)
    cfg = OplmConfig(
        hidden_size=32,
        num_attention_heads=4,
        intermediate_size=64,
        num_hidden_layers=3,
        max_position_embeddings=64,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        rope_dim=4,  # partial RoPE: half of each 8-channel head rotates
        nope_dim=4,
        value_residual="learnable",
        canon_enabled=True,
        canon_positions=["A", "B", "C", "D"],
        canon_kernel_sizes=[3, 4, 7],  # odd and even kernels, one wider than a sequence
    )
    model = OplmForMaskedLM(cfg)
    if checkpointing is not None:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"mode": checkpointing})
    return model.train()


def _batches() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """The same three labeled sequences, one per padded row and packed into one row."""
    gen = torch.Generator().manual_seed(1)
    seqs = [torch.randint(5, 25, (n,), generator=gen) for n in LENGTHS]
    labels = [
        torch.where(torch.rand(n, generator=gen) < 0.5, s, -100)
        for s, n in zip(seqs, LENGTHS, strict=True)
    ]
    width = max(LENGTHS) + 1
    padded = {
        "input_ids": torch.ones(len(seqs), width, dtype=torch.long),
        "attention_mask": torch.zeros(len(seqs), width, dtype=torch.long),
        "labels": torch.full((len(seqs), width), -100),
    }
    for row, (s, lab) in enumerate(zip(seqs, labels, strict=True)):
        padded["input_ids"][row, : len(s)] = s
        padded["attention_mask"][row, : len(s)] = 1
        padded["labels"][row, : len(s)] = lab
    total = sum(LENGTHS)
    pad = 2
    packed = {
        "input_ids": torch.cat([*seqs, torch.ones(pad, dtype=torch.long)])[None],
        "attention_mask": torch.cat([torch.ones(total), torch.zeros(pad)]).long()[None],
        "labels": torch.cat([*labels, torch.full((pad,), -100)])[None],
        "segment_ids": torch.cat(
            [torch.full((n,), i + 1) for i, n in enumerate(LENGTHS)]
            + [torch.zeros(pad, dtype=torch.long)]
        )[None],
    }
    return padded, packed


def _unpack(packed_logits: torch.Tensor) -> list[torch.Tensor]:
    starts = [sum(LENGTHS[:i]) for i in range(len(LENGTHS))]
    return [packed_logits[0, s : s + n] for s, n in zip(starts, LENGTHS, strict=True)]


@pytest.mark.parametrize("output_attentions", [False, True], ids=["sdpa", "manual"])
def test_packed_logits_match_padded_rows(output_attentions: bool) -> None:
    model = _model(None)
    padded, packed = _batches()
    ref = model(**padded, output_attentions=output_attentions)
    out = model(**packed, output_attentions=output_attentions)
    for row, got in enumerate(_unpack(out.logits)):
        torch.testing.assert_close(got, ref.logits[row, : LENGTHS[row]], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(out.loss, ref.loss, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("checkpointing", [None, "full", "selective"])
def test_packed_gradients_match_padded_rows(checkpointing: str | None) -> None:
    padded, packed = _batches()
    grads = []
    for batch in (padded, packed):
        model = _model(checkpointing)
        model(**batch).loss.backward()
        grads.append({n: p.grad for n, p in model.named_parameters() if p.grad is not None})
    assert grads[0].keys() == grads[1].keys()
    for name, ref in grads[0].items():
        torch.testing.assert_close(grads[1][name], ref, rtol=1e-4, atol=1e-6, msg=name)


def test_without_segment_ids_a_single_sequence_row_is_unchanged() -> None:
    """One sequence per row, packed or not, is today's computation: same ids, same output."""
    model = _model(None).eval()
    padded, _ = _batches()
    seg = padded["attention_mask"].clone()  # one segment per row, pads 0
    with torch.no_grad():
        ref = model(**padded).logits
        got = model(**padded, segment_ids=seg).logits
    mask = padded["attention_mask"].bool()
    torch.testing.assert_close(got[mask], ref[mask], rtol=1e-5, atol=1e-6)
